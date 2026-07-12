"""
compare_and_visualize.py (v2)
------------------------------
Extends the original three-model comparison (Backprop / Hebbian+readout /
Random+readout) with three additions:

  1. MULTI-SEED AVERAGING: each model is trained several times with different
     random seeds, and we report mean +/- std accuracy instead of one number.
  2. SAMPLE-EFFICIENCY SWEEP: retrain the SUPERVISED parts (backprop, and each
     readout) using only a fraction of the labeled training data. The Hebbian
     feature layer itself always trains on the FULL, unlabeled training set --
     that's the whole point of the comparison (it never needed labels to begin
     with), so only the supervised steps are label-starved.
  3. TRAINING CURVES: backprop and the two readouts now record test accuracy
     across epochs, so we can plot how fast each one converges.

Outputs (PNG files in this folder):
  1. accuracy_comparison.png     -- bar chart with error bars (multi-seed)
  2. confusion_matrices.png      -- unchanged from before, single representative run
  3. misclassified.png           -- unchanged from before, single representative run
  4. filters.png                 -- unchanged from before, single representative run
  5. sample_efficiency.png       -- NEW: accuracy vs. fraction of labeled data
  6. training_curves.png         -- NEW: accuracy vs. epoch, backprop vs Hebbian readout

Requires: torch, torchvision, matplotlib
Runtime: meaningfully longer than v1 -- see SEEDS / EFFICIENCY_FRACTIONS below.
Turn those numbers down for a quick first pass, then back up for final results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

WIDTH = 400                          # shared hidden-layer width, identical for all 3 models
SEEDS = [0, 1, 2]                    # how many times to repeat the main comparison
EFFICIENCY_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
EFFICIENCY_SEEDS = [0, 1]            # fewer seeds for the sweep, to keep runtime sane
BACKPROP_EPOCHS = 8
READOUT_EPOCHS = 100

# ---------------------------------------------------------------------------
# Data. Loaded once. All models see the same unit-normalized 784-vectors.
# ---------------------------------------------------------------------------
train_set = torchvision.datasets.MNIST(root="./data", train=True,  download=True)
test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True)

def prep(ds):
    raw = ds.data.float() / 255.0
    X = F.normalize(raw.view(len(ds), -1), dim=1)
    return X.to(device), ds.targets.clone().to(device), raw

X_train_full, y_train_full, _        = prep(train_set)
X_test,       y_test,       raw_test = prep(test_set)
y_test_cpu = y_test.cpu()
N_TRAIN = X_train_full.size(0)


def labeled_subset(fraction, seed):
    """Returns a random subset of (X_train_full, y_train_full) of the given
    fraction. Used ONLY for the supervised steps (backprop, readout)."""
    if fraction >= 1.0:
        return X_train_full, y_train_full
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(N_TRAIN * fraction)
    idx = torch.randperm(N_TRAIN, generator=g)[:n].to(device)
    return X_train_full[idx], y_train_full[idx]


# ===========================================================================
# Model A: backprop, end-to-end. Now records a per-epoch test-accuracy curve.
# ===========================================================================
def train_backprop(X, y, epochs=BACKPROP_EPOCHS, seed=0, track_curve=False):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(784, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 10)).to(device)
    opt, lossf = torch.optim.Adam(net.parameters(), lr=1e-3), nn.CrossEntropyLoss()

    curve = []
    for epoch in range(epochs):
        perm = torch.randperm(X.size(0), device=device)
        for i in range(0, X.size(0), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lossf(net(X[idx]), y[idx]).backward()
            opt.step()
        if track_curve:
            with torch.no_grad():
                acc = (net(X_test).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100
            curve.append(acc)
    return net, curve


# ===========================================================================
# Hebbian hidden layer. ALWAYS trains on the full, unlabeled training set --
# label_fraction never touches this function, on purpose (see module docstring).
# ===========================================================================
def hebbian_W1(epochs=3, lr=0.05, seed=0):
    torch.manual_seed(seed)
    W = F.normalize(X_train_full[torch.randint(0, N_TRAIN, (WIDTH,))].clone(), dim=1)
    for _ in range(epochs):
        perm = torch.randperm(N_TRAIN, device=device)
        for i in range(0, N_TRAIN, 256):
            x = X_train_full[perm[i:i + 256]]
            winners = (x @ W.t()).argmax(1)
            sum_x = torch.zeros_like(W)
            counts = torch.zeros(WIDTH, device=device)
            sum_x.index_add_(0, winners, x)
            counts.index_add_(0, winners, torch.ones_like(winners, dtype=torch.float))
            won = counts > 0
            W[won] = F.normalize(W[won] + lr * (sum_x[won] / counts[won].unsqueeze(1) - W[won]), dim=1)
    with torch.no_grad():
        all_winners = (X_train_full @ W.t()).argmax(dim=1)
        n_active = all_winners.unique().numel()
    print(f"  [seed {seed}] {n_active}/{WIDTH} Hebbian units won at least once across full training set")

    return W


def random_W1(seed=0):
    torch.manual_seed(seed)
    return F.normalize(torch.randn(WIDTH, 784, device=device), dim=1)


def train_readout(W1, X, y, epochs=READOUT_EPOCHS, track_curve=False, curve_every=5):
    """X, y here are the (possibly label-starved) supervised subset."""
    Htr = F.relu(X @ W1.t())
    Hte = F.relu(X_test @ W1.t())
    ro, lossf = nn.Linear(WIDTH, 10).to(device), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(ro.parameters(), lr=0.01)

    curve = []
    for epoch in range(epochs):
        opt.zero_grad()
        lossf(ro(Htr), y).backward()
        opt.step()
        if track_curve and (epoch % curve_every == 0 or epoch == epochs - 1):
            with torch.no_grad():
                acc = (ro(Hte).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100
            curve.append((epoch, acc))
    return ro, curve

def redundancy_check(W, name, seed):
    """W rows are already unit-length (both hebbian_W1 and random_W1 normalize
    their rows), so W @ W.T directly gives cosine similarity between every
    pair of templates."""
    with torch.no_grad():
        sim = W @ W.t()
        n = W.shape[0]
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=W.device)
        mean_sim = sim[off_diag_mask].mean().item()

        s = torch.linalg.svdvals(W)
        eff_dim = ((s.sum() ** 2) / (s ** 2).sum()).item()

    print(f"  [seed {seed}] {name}: mean pairwise similarity={mean_sim:.3f}, "
          f"effective dimensionality={eff_dim:.1f}/{n}")
    return mean_sim, eff_dim

# ===========================================================================
# One full pipeline run: all three models, given a seed and a label fraction.
# Returns just the three test accuracies (used by both the seed-averaging
# and the sample-efficiency experiments below).
# ===========================================================================
def run_pipeline(seed, label_fraction, track_curve=False):
    X_sup, y_sup = labeled_subset(label_fraction, seed)

    net, curveA = train_backprop(X_sup, y_sup, seed=seed, track_curve=track_curve)
    with torch.no_grad():
        accA = (net(X_test).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100

    Wh = hebbian_W1(seed=seed)   # full unlabeled data, independent of label_fraction
    redundancy_check(Wh, "Hebbian", seed)

    roB, curveB = train_readout(Wh, X_sup, y_sup, track_curve=track_curve)
    with torch.no_grad():
        accB = (roB(F.relu(X_test @ Wh.t())).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100

    Wr = random_W1(seed=seed)
    redundancy_check(Wr, "Random", seed)
    roC, curveC = train_readout(Wr, X_sup, y_sup, track_curve=track_curve)
    with torch.no_grad():
        accC = (roC(F.relu(X_test @ Wr.t())).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100
    

    extras = dict(net=net, Wh=Wh, Wr=Wr, roB=roB, roC=roC,
                   predsA=net(X_test).argmax(1).cpu() if not track_curve else None)
    return dict(backprop=accA, hebbian=accB, random=accC,
                curves=dict(backprop=curveA, hebbian=curveB, random=curveC),
                extras=extras)


# ===========================================================================
# Experiment 1: multi-seed averaged accuracy (replaces the single-run bar chart)
# ===========================================================================
print(f"\n=== Multi-seed comparison ({len(SEEDS)} seeds) ===")
seed_results = {"backprop": [], "hebbian": [], "random": []}
representative_run = None   # we'll keep seed 0's full run for the other figures

for seed in SEEDS:
    r = run_pipeline(seed, label_fraction=1.0, track_curve=(seed == SEEDS[0]))
    seed_results["backprop"].append(r["backprop"])
    seed_results["hebbian"].append(r["hebbian"])
    seed_results["random"].append(r["random"])
    print(f"  seed {seed}: backprop={r['backprop']:.2f}%  hebbian={r['hebbian']:.2f}%  random={r['random']:.2f}%")
    if seed == SEEDS[0]:
        representative_run = r

means = {k: float(np.mean(v)) for k, v in seed_results.items()}
stds  = {k: float(np.std(v))  for k, v in seed_results.items()}
print("Means:", means)
print("Stds: ", stds)

# ---- Figure 1: accuracy bar chart with error bars ----
labels = ["Backprop\n(end-to-end)", "Hebbian\n+ readout", "Random\n+ readout"]
keys = ["backprop", "hebbian", "random"]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, [means[k] for k in keys], yerr=[stds[k] for k in keys],
                capsize=6, color=["#4C72B0", "#55A868", "#C44E52"])
for b, k in zip(bars, keys):
    plt.text(b.get_x() + b.get_width() / 2, means[k] + stds[k] + 1,
              f"{means[k]:.1f}%", ha="center", fontweight="bold")
plt.ylabel("Test accuracy (%)"); plt.ylim(0, 100)
plt.title(f"MNIST: does it matter HOW the hidden layer learns?\n(mean \u00b1 std over {len(SEEDS)} seeds)")
plt.tight_layout(); plt.savefig("accuracy_comparison.png", dpi=150); plt.close()


# ===========================================================================
# Experiment 2: sample-efficiency sweep
# ===========================================================================
print(f"\n=== Sample-efficiency sweep ({EFFICIENCY_FRACTIONS}) ===")
sweep = {"backprop": {}, "hebbian": {}, "random": {}}
for frac in EFFICIENCY_FRACTIONS:
    accs = {"backprop": [], "hebbian": [], "random": []}
    for seed in EFFICIENCY_SEEDS:
        r = run_pipeline(seed, label_fraction=frac)
        accs["backprop"].append(r["backprop"])
        accs["hebbian"].append(r["hebbian"])
        accs["random"].append(r["random"])
    for k in accs:
        sweep[k][frac] = float(np.mean(accs[k]))
    print(f"  fraction={frac}: " + "  ".join(f"{k}={sweep[k][frac]:.2f}%" for k in accs))

plt.figure(figsize=(6, 4))
for k, color, label in zip(keys, ["#4C72B0", "#55A868", "#C44E52"], labels):
    fracs = sorted(sweep[k].keys())
    plt.plot(fracs, [sweep[k][f] for f in fracs], marker="o", color=color,
              label=label.replace("\n", " "))
plt.xlabel("Fraction of labeled training data used")
plt.ylabel("Test accuracy (%)")
plt.title("Sample efficiency: how much labeled data does each model need?")
plt.legend(); plt.ylim(0, 100)
plt.tight_layout(); plt.savefig("sample_efficiency.png", dpi=150); plt.close()


# ===========================================================================
# Experiment 3: training curves (from the representative seed-0 run above)
# ===========================================================================
plt.figure(figsize=(6, 4))
backprop_curve = representative_run["curves"]["backprop"]           # list of acc per epoch
hebbian_curve  = representative_run["curves"]["hebbian"]            # list of (epoch, acc)

plt.plot(range(1, len(backprop_curve) + 1), backprop_curve, marker="o",
          color="#4C72B0", label="Backprop (per epoch)")
if hebbian_curve:
    epochs_h, accs_h = zip(*hebbian_curve)
    plt.plot(epochs_h, accs_h, marker="s", color="#55A868", label="Hebbian readout")

plt.xlabel("Training epoch")
plt.ylabel("Test accuracy (%)")
plt.title("How fast does each model converge?")
plt.legend(); plt.ylim(0, 100)
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()


# ===========================================================================
# Figures 4-6: confusion matrices / misclassified / filters, from the SAME
# representative (seed 0) run, so all figures describe one consistent run.
# ===========================================================================
extras = representative_run["extras"]
net, Wh, Wr, roB, roC = extras["net"], extras["Wh"], extras["Wr"], extras["roB"], extras["roC"]

with torch.no_grad():
    predsA = net(X_test).argmax(1).cpu()
    predsB = roB(F.relu(X_test @ Wh.t())).argmax(1).cpu()
    predsC = roC(F.relu(X_test @ Wr.t())).argmax(1).cpu()

results = {
    "Backprop\n(end-to-end)": dict(preds=predsA, W1=net[0].weight.detach().cpu()),
    "Hebbian\n+ readout":     dict(preds=predsB, W1=Wh.cpu()),
    "Random\n+ readout":      dict(preds=predsC, W1=Wr.cpu()),
}
for name, r in results.items():
    r["acc"] = (r["preds"] == y_test_cpu).float().mean().item() * 100

names = list(results.keys())
row_labels = ["Backprop", "Hebbian", "Random"]

# ---- confusion matrices ----
def error_matrix(preds):
    cm = torch.bincount((y_test_cpu * 10 + preds), minlength=100).reshape(10, 10).float()
    cm.fill_diagonal_(0)
    return cm

err = {n: error_matrix(results[n]["preds"]) for n in names}
vmax = max(m.max().item() for m in err.values())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, n in zip(axes, names):
    im = ax.imshow(err[n], cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(f"{n}  ({results[n]['acc']:.1f}%)")
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
fig.colorbar(im, ax=axes, shrink=0.8, label="# of mistakes")
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight"); plt.close()

# ---- misclassified examples ----
N = 8
fig, axes = plt.subplots(3, N, figsize=(N * 1.15, 5.0))
for row, n in enumerate(names):
    preds = results[n]["preds"]
    wrong = (preds != y_test_cpu).nonzero(as_tuple=True)[0][:N]
    for col in range(N):
        ax = axes[row, col]; ax.set_xticks([]); ax.set_yticks([])
        if col < len(wrong):
            wi = wrong[col].item()
            ax.imshow(raw_test[wi], cmap="gray")
            ax.set_title(f"{y_test_cpu[wi].item()}\u2192{preds[wi].item()}", fontsize=9)
    axes[row, 0].set_ylabel(row_labels[row], fontsize=10, labelpad=8)
plt.suptitle("Digits each model gets wrong  (true \u2192 predicted)")
plt.subplots_adjust(hspace=0.6)
plt.tight_layout(); plt.savefig("misclassified.png", dpi=150); plt.close()

# ---- learned filters ----
M = 10
fig, axes = plt.subplots(3, M, figsize=(M * 1.1, 5.0))
for row, n in enumerate(names):
    W = results[n]["W1"]
    for col in range(M):
        f = W[col].view(28, 28)
        v = f.abs().max()
        ax = axes[row, col]; ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(f, cmap="seismic", vmin=-v, vmax=v)
    axes[row, 0].set_ylabel(row_labels[row], fontsize=10, labelpad=8)
plt.suptitle("What each hidden unit 'looks for' (its weight pattern)")
plt.subplots_adjust(hspace=0.6)
plt.tight_layout(); plt.savefig("filters.png", dpi=150); plt.close()

print("\nSaved: accuracy_comparison.png, sample_efficiency.png, training_curves.png,")
print("       confusion_matrices.png, misclassified.png, filters.png")
