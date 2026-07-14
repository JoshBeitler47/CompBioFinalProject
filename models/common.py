"""
common.py
---------
Shared data loading and training helpers for the comparison scripts in this
folder (catastrophic_forgetting.py, compare_and_visualize_v2.py,
multi_seed_variability.py, fashion_mnist_variability.py). Pulling these out
means a fix made once -- e.g. mean-centering the data, or the conscience
mechanism below -- applies everywhere instead of needing to be copy-pasted
into four near-identical copies of the same functions.

Two things worth flagging up front, both raised in
context/project_notes_backprop_vs_hebbian.md:

1. CENTERING (section 6). All data here is mean-centered on the TRAINING
   set's average image before being unit-normalized. Uncentered data wastes
   representational capacity on "the average digit" -- every input shares a
   big component pointing in the same direction, which is exactly the kind
   of thing a variance-seeking / competitive-similarity rule shouldn't have
   to spend units on.

2. THE HEBBIAN RULE HERE IS NOT LITERALLY OJA'S DELTA RULE. The math in the
   project notes (section 4c) derives Delta w = eta * y * (x - y*w), applied
   to EVERY unit on every step. That rule, applied to many units at once
   with no other change, has a known failure mode the notes call out in
   section 4f: every unit converges to the SAME top principal component,
   which is useless for a multi-unit hidden layer, unless you add a
   deflation step (Sanger's rule) or a subspace formulation. `hebbian_W1`
   instead defaults to mode="lateral": Foldiak-style (1990) anti-Hebbian
   lateral inhibition + Turrigiano-style homeostatic firing-rate thresholds.
   Units settle to an activity level under mutual inhibition (a soft
   population code, not one hard winner), templates update toward what each
   active unit saw, lateral weights between co-active units strengthen
   (pushing them apart next time), and each unit's threshold rises or falls
   to hold its firing rate near a small target -- which also makes the code
   sparse by construction. mode="competitive" (hard winner-take-all + a
   frequency-bias "conscience" heuristic) is kept as a simpler fallback/
   comparison baseline, not the default. Neither is literally Oja's rule --
   call the default "Foldiak-style Hebbian learning," the fallback
   "competitive Hebbian learning."
"""

import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State for mode="lateral" hebbian_W1: feedforward templates W, lateral
# inhibitory weights L, and per-unit homeostatic thresholds theta. Kept as a
# named bundle (instead of three loose return values) so it can be threaded
# through hebbian_features/redundancy_check/continuation calls as one object.
HebbianState = namedtuple("HebbianState", ["W", "L", "theta"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(dataset_cls, root="./data"):
    """Loads train/test splits of `dataset_cls` (MNIST, FashionMNIST, ...),
    mean-centers both on the TRAINING set's average image, then unit-normalizes
    each image. Returns X_train, y_train, X_test, y_test, raw_test, mean.

    `raw_test` is the *uncentered* [0,1] pixel tensor (for plotting example
    images); `mean` is returned so callers who need to re-derive uncentered
    data or center other splits (e.g. task-split subsets) can reuse it.
    """
    train_set = dataset_cls(root=root, train=True, download=True)
    test_set = dataset_cls(root=root, train=False, download=True)

    raw_train = train_set.data.float() / 255.0
    raw_test = test_set.data.float() / 255.0

    flat_train = raw_train.view(len(train_set), -1)
    flat_test = raw_test.view(len(test_set), -1)

    mean = flat_train.mean(dim=0, keepdim=True)  # [1, 784], TRAIN-set mean only

    X_train = F.normalize(flat_train - mean, dim=1).to(device)
    X_test = F.normalize(flat_test - mean, dim=1).to(device)
    y_train = train_set.targets.clone().to(device)
    y_test = test_set.targets.clone().to(device)

    return X_train, y_train, X_test, y_test, raw_test, mean.to(device)


def center(X_flat, mean):
    """Apply an already-computed training-set mean to another split (e.g. a
    Task-A/Task-B subset), then unit-normalize. Keeps every split consistent
    with the same centering reference."""
    return F.normalize(X_flat - mean, dim=1)


# ---------------------------------------------------------------------------
# Hebbian hidden layer (see module docstring, point 2)
# ---------------------------------------------------------------------------
def hebbian_W1(
    X_train,
    width,
    epochs=3,
    lr_start=0.05,
    lr_end=0.05,
    seed=0,
    batch_size=256,
    mode="lateral",
    # mode="lateral" params (Foldiak anti-Hebbian lateral inhibition + homeostasis)
    target_rate=0.08,
    lateral_lr=0.1,
    threshold_lr=0.3,
    relax_iters=4,
    lateral_cap=2.0,
    state_init=None,
    # mode="competitive" params (legacy winner-take-all + conscience fallback)
    conscience=True,
    conscience_strength=10.0,
    conscience_beta=0.05,
    W_init=None,
    freq_init=None,
    return_state=False,
):
    """Trains the Hebbian hidden layer. See module docstring, point 2, for
    which mode is which and why "lateral" is the default."""
    if mode == "lateral":
        return _hebbian_W1_lateral(
            X_train, width, epochs, lr_start, lr_end, seed, batch_size,
            target_rate, lateral_lr, threshold_lr, relax_iters, lateral_cap,
            state_init,
        )
    return _hebbian_W1_competitive(
        X_train, width, epochs, lr_start, lr_end, seed, batch_size,
        conscience, conscience_strength, conscience_beta,
        W_init, freq_init, return_state,
    )


def _hebbian_W1_lateral(
    X_train, width, epochs, lr_start, lr_end, seed, batch_size,
    target_rate, lateral_lr, threshold_lr, relax_iters, lateral_cap, state_init,
):
    """Foldiak-style (1990) anti-Hebbian lateral inhibition + Turrigiano-style
    homeostatic thresholds. Replaces hard winner-take-all with a soft
    population code: on each input, activity settles under mutual inhibition
    (`y <- relu(a - theta - y @ L.T)`, iterated a few times) instead of one
    unit winning outright. Three local updates follow, each using only
    information the unit (or unit pair, for the lateral term) already has:

      - feedforward W: each unit moves its template toward what it saw,
        SCALED BY ITS OWN ACTIVITY LEVEL (mean_y below) -- a unit that barely
        crossed threshold moves its template only slightly; a confidently,
        strongly active unit moves it fully. This is the actual "y" term
        every Hebbian rule has (Delta w ~ y * x): dropping it (using a flat
        step for every unit that fired at all, regardless of how strongly)
        was an earlier bug in this file -- it let marginal, low-confidence
        activations drag many templates toward the same generic, low-
        information patterns, collapsing effective dimensionality to a
        handful of directions out of `width`. After renormalizing (the
        Oja-style stability trick used elsewhere in this file), templates
        stay diverse instead of collapsing.
      - lateral L: units that fire together MORE than chance (target_rate^2)
        get their mutual inhibition strengthened -- the actual anti-Hebbian
        rule, and the mechanism that decorrelates templates instead of the
        old conscience frequency-bias heuristic.
      - threshold theta: each unit's firing rate is pulled toward
        target_rate -- fire too often, get harder to activate; fire too
        rarely, get easier. This is homeostatic synaptic scaling (Turrigiano)
        in its simplest form, and since target_rate is small, it makes the
        resulting code sparse by construction. threshold_lr is deliberately
        NOT tiny: theta starts at 0 (no threshold at all), so if it ramps up
        too slowly, far too many units fire on every input for the first
        stretch of training, before homeostasis has caught up -- another
        earlier bug, which let templates collapse before any competitive
        pressure existed to stop them.
    """
    torch.manual_seed(seed)
    n = X_train.size(0)
    device_ = X_train.device
    if state_init is not None:
        W, L, theta = state_init.W.clone(), state_init.L.clone(), state_init.theta.clone()
    else:
        W = F.normalize(X_train[torch.randint(0, n, (width,))].clone(), dim=1)
        L = torch.zeros(width, width, device=device_)
        theta = torch.zeros(width, device=device_)

    off_diag = ~torch.eye(width, dtype=torch.bool, device=device_)

    for e in range(epochs):
        lr = lr_start + (lr_end - lr_start) * (e / max(epochs - 1, 1))
        perm = torch.randperm(n, device=device_)
        for i in range(0, n, batch_size):
            x = X_train[perm[i : i + batch_size]]
            a = x @ W.t()  # [B, width] feedforward drive

            y = F.relu(a - theta)
            for _ in range(relax_iters):
                y = F.relu(a - theta - y @ L.t())  # settle under lateral inhibition

            b = x.size(0)
            activity_sum = y.sum(dim=0)  # [width]
            mean_y = activity_sum / b  # [width], each unit's average activity this batch
            has_activity = mean_y > 1e-6

            # Feedforward Hebbian update: move toward the activity-weighted mean
            # input direction, but scale the STEP SIZE by mean_y (confidence) --
            # see docstring above for why this matters -- then renormalize
            # (Oja-style stability trick).
            weighted_x = y.t() @ x  # [width, 784], sum over batch of y_i * x
            target = torch.zeros_like(W)
            target[has_activity] = weighted_x[has_activity] / activity_sum[has_activity].unsqueeze(1)
            step = lr * mean_y.unsqueeze(1) * (target - W)
            W[has_activity] = F.normalize(W[has_activity] + step[has_activity], dim=1)

            # Anti-Hebbian lateral update: co-activity above chance (target_rate^2)
            # strengthens mutual inhibition; below chance, it relaxes (but never
            # goes negative -- L is purely inhibitory).
            co_activity = (y.t() @ y) / b  # [width, width]
            L = (L + lateral_lr * (co_activity - target_rate**2)).clamp(0.0, lateral_cap)
            L = L * off_diag  # no self-inhibition; theta already handles that

            # Homeostatic threshold update: pull each unit's firing rate toward target_rate.
            batch_rate = (y > 0).float().mean(dim=0)  # [width]
            theta = (theta + threshold_lr * (batch_rate - target_rate)).clamp(min=0.0)

    return HebbianState(W=W, L=L, theta=theta)


def _hebbian_W1_competitive(
    X_train, width, epochs, lr_start, lr_end, seed, batch_size,
    conscience, conscience_strength, conscience_beta,
    W_init, freq_init, return_state,
):
    """Winner-take-all competitive learning with Oja-style renormalization.
    Legacy fallback -- see mode="lateral" (the default) for the mechanism
    this project actually uses now.

    conscience: if True, applies a frequency-bias ("conscience mechanism",
    DeSieno 1988, "Adding a conscience to competitive learning") -- units
    that have been winning more than their fair share get a penalty in the
    competition, and chronically-losing ("dead") units get a fairer shot.
    """
    torch.manual_seed(seed)
    n = X_train.size(0)
    if W_init is not None:
        W = W_init.clone()  # continue training an existing template matrix
    else:
        W = F.normalize(X_train[torch.randint(0, n, (width,))].clone(), dim=1)
    if freq_init is not None:
        freq = freq_init.clone()  # carry over win-frequency state across tasks
    else:
        freq = torch.full((width,), 1.0 / width, device=X_train.device)

    for e in range(epochs):
        lr = lr_start + (lr_end - lr_start) * (e / max(epochs - 1, 1))
        perm = torch.randperm(n, device=X_train.device)
        for i in range(0, n, batch_size):
            x = X_train[perm[i : i + batch_size]]
            scores = x @ W.t()
            if conscience:
                bias = conscience_strength * (1.0 / width - freq)
                scores = scores + bias
            winners = scores.argmax(dim=1)

            sum_x = torch.zeros_like(W)
            counts = torch.zeros(width, device=X_train.device)
            sum_x.index_add_(0, winners, x)
            counts.index_add_(0, winners, torch.ones_like(winners, dtype=torch.float))

            won = counts > 0
            W[won] = F.normalize(
                W[won] + lr * (sum_x[won] / counts[won].unsqueeze(1) - W[won]), dim=1
            )

            if conscience:
                batch_share = counts / x.size(0)
                freq = (1 - conscience_beta) * freq + conscience_beta * batch_share

    if return_state:
        return W, freq
    return W


def random_W1(width, in_dim=784, seed=0):
    torch.manual_seed(seed)
    return F.normalize(torch.randn(width, in_dim, device=device), dim=1)


def redundancy_check(W, name="", seed=None, verbose=True):
    """W rows are assumed unit-length, so W @ W.T is cosine similarity
    between every pair of templates. Returns (mean pairwise similarity,
    effective dimensionality via SVD participation ratio). Accepts either a
    raw template tensor or a HebbianState (mode="lateral") -- the templates
    are what's being compared either way; L/theta don't factor in here."""
    if isinstance(W, HebbianState):
        W = W.W
    with torch.no_grad():
        sim = W @ W.t()
        n = W.shape[0]
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=W.device)
        mean_sim = sim[off_diag_mask].mean().item()

        s = torch.linalg.svdvals(W)
        eff_dim = ((s.sum() ** 2) / (s**2).sum()).item()

    if verbose:
        tag = f"[seed {seed}] " if seed is not None else ""
        print(
            f"  {tag}{name}: mean pairwise similarity={mean_sim:.3f}, "
            f"effective dimensionality={eff_dim:.1f}/{n}"
        )
    return mean_sim, eff_dim


# ---------------------------------------------------------------------------
# Sparsity control (context notes section 1, "the rigorous version": disentangle
# the learning RULE from representation SPARSITY when measuring forgetting)
# ---------------------------------------------------------------------------
def topk_mask(h, k):
    """Zero out every activation except the top-k per row. A simple,
    purely-postsynaptic way to force a sparse code -- k active units out of
    however many there are -- independent of what learning rule produced the
    activations in the first place, so it can be applied identically to a
    backprop MLP's hidden layer or to Hebbian/random features."""
    if k is None or k >= h.size(1):
        return h
    topk = torch.topk(h, k, dim=1)
    mask = torch.zeros_like(h)
    mask.scatter_(1, topk.indices, 1.0)
    return h * mask


# ---------------------------------------------------------------------------
# Backprop MLP
# ---------------------------------------------------------------------------
def build_backprop_net(width, seed=0, in_dim=784, out_dim=10):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, out_dim)).to(
        device
    )


def backprop_forward(net, x, sparsity_k=None):
    """Runs `net` (as built by build_backprop_net) manually so a top-k sparsity
    mask can be inserted between the hidden ReLU and the output layer."""
    h = net[1](net[0](x))  # ReLU(Linear1(x))
    if sparsity_k is not None:
        h = topk_mask(h, sparsity_k)
    return net[2](h)


def train_backprop(
    net, X, y, epochs=8, lr=1e-3, batch_size=128, sparsity_k=None, track_curve=False,
    X_test=None, y_test=None,
):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    curve = []
    for _ in range(epochs):
        perm = torch.randperm(X.size(0), device=X.device)
        for i in range(0, X.size(0), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            lossf(backprop_forward(net, X[idx], sparsity_k), y[idx]).backward()
            opt.step()
        if track_curve and X_test is not None:
            with torch.no_grad():
                acc = (
                    backprop_forward(net, X_test, sparsity_k).argmax(1).cpu()
                    == y_test.cpu()
                ).float().mean().item() * 100
            curve.append(acc)
    return net, curve


# ---------------------------------------------------------------------------
# Linear readout on top of frozen hidden-layer features
# ---------------------------------------------------------------------------
def hebbian_features(X, W1, sparsity_k=None, relax_iters=4):
    """Extracts hidden-layer activity for `X`. Accepts either a raw template
    tensor (Random's frozen projection, or a mode="competitive" W) -- in
    which case this is just relu(X @ W1.T) -- or a HebbianState (mode=
    "lateral"), in which case it runs the same settling relaxation used
    during training (no learning, just inference) so features reflect what
    the trained lateral-inhibition circuit actually represents."""
    if isinstance(W1, HebbianState):
        W, L, theta = W1.W, W1.L, W1.theta
        a = X @ W.t()
        h = F.relu(a - theta)
        for _ in range(relax_iters):
            h = F.relu(a - theta - h @ L.t())
    else:
        h = F.relu(X @ W1.t())
    if sparsity_k is not None:
        h = topk_mask(h, sparsity_k)
    return h


def build_readout(width, seed=0, out_dim=10):
    torch.manual_seed(seed)
    return nn.Linear(width, out_dim).to(device)


def train_readout(ro, H, y, epochs=100, lr=0.01, track_curve=False, curve_every=5,
                   H_test=None, y_test=None):
    opt = torch.optim.Adam(ro.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        lossf(ro(H), y).backward()
        opt.step()
        if track_curve and H_test is not None and (
            epoch % curve_every == 0 or epoch == epochs - 1
        ):
            with torch.no_grad():
                acc = (
                    ro(H_test).argmax(1).cpu() == y_test.cpu()
                ).float().mean().item() * 100
            curve.append((epoch, acc))
    return ro, curve


@torch.no_grad()
def eval_acc(predict_fn, X, y):
    preds = predict_fn(X).argmax(1)
    return (preds.cpu() == y.cpu()).float().mean().item() * 100


# ---------------------------------------------------------------------------
# The 3-condition (Backprop / Hebbian / Random) seed-variability sweep.
# multi_seed_variability.py and fashion_mnist_variability.py are the SAME
# experiment on two datasets -- this is that shared logic, parameterized
# by dataset class and output filename, so a change to the experiment only
# needs to be made once.
# ---------------------------------------------------------------------------
def run_seed_variability_experiment(
    dataset_cls,
    output_name,
    title,
    num_seeds=5,
    width=400,
    backprop_epochs=8,
    readout_epochs=100,
):
    os.makedirs("output", exist_ok=True)
    X_train, y_train, X_test, y_test, _raw_test, _mean = load_dataset(dataset_cls)

    conditions = [
        "Backprop\n(end-to-end)",
        "Hebbian\n+ readout",
        "Random\n+ readout",
    ]
    results = {c: [] for c in conditions}

    for seed in range(num_seeds):
        print(f"\n=== Seed {seed} ({seed + 1}/{num_seeds}) ===")

        net = build_backprop_net(width, seed=seed)
        net, _ = train_backprop(net, X_train, y_train, epochs=backprop_epochs)
        acc = eval_acc(lambda X: net(X), X_test, y_test)
        print(f"  Backprop:           {acc:.2f}%")
        results[conditions[0]].append(acc)

        Wh = hebbian_W1(X_train, width, epochs=3, lr_start=0.05, lr_end=0.05, seed=seed)
        ro = build_readout(width, seed=seed)
        H = hebbian_features(X_train, Wh)
        ro, _ = train_readout(ro, H, y_train, epochs=readout_epochs)
        acc = eval_acc(lambda X: ro(hebbian_features(X, Wh)), X_test, y_test)
        print(f"  Hebbian (baseline): {acc:.2f}%")
        results[conditions[1]].append(acc)

        Wr = random_W1(width, seed=seed)
        ro_r = build_readout(width, seed=seed)
        H_r = hebbian_features(X_train, Wr)
        ro_r, _ = train_readout(ro_r, H_r, y_train, epochs=readout_epochs)
        acc = eval_acc(lambda X: ro_r(hebbian_features(X, Wr)), X_test, y_test)
        print(f"  Random:             {acc:.2f}%")
        results[conditions[2]].append(acc)

    print(f"\n=== Summary across {num_seeds} seeds ===")
    stats = {}
    for c in conditions:
        t = torch.tensor(results[c])
        stats[c] = (t.mean().item(), t.std().item())
        print(
            f"  {c.replace(chr(10), ' ')}: {t.mean():.2f}% +/- {t.std():.2f}%  "
            f"(runs: {[round(v, 1) for v in results[c]]})"
        )

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, c in enumerate(conditions):
        jitter = torch.linspace(-0.08, 0.08, num_seeds).tolist()
        ax.scatter(
            [i + j for j in jitter],
            results[c],
            color=colors[i],
            alpha=0.5,
            zorder=2,
            label="individual seeds" if i == 0 else None,
        )
        mean, std = stats[c]
        ax.errorbar(
            i, mean, yerr=std, fmt="D", color=colors[i], markersize=9, capsize=6,
            zorder=3, markeredgecolor="black", markeredgewidth=0.8,
            label="mean +/- 1 std" if i == 0 else None,
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace("\n", " ") for c in conditions])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(f"output/{output_name}", dpi=150)
    plt.close()
    print(f"\nSaved: output/{output_name}")

    return results, stats
