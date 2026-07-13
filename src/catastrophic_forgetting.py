"""
catastrophic_forgetting.py
---------------------------
Tests whether backprop or Hebbian-learned representations "forget" old
knowledge when trained on new data they never revisit.

THE EXPERIMENT:
  Task A = digits 0-4.  Task B = digits 5-9.

  For each model (Backprop, Hebbian+readout, Random+readout):
    1. Train on Task A only. Measure test accuracy on Task A digits.
    2. Continue training the SAME model (no reset!) on Task B only --
       Task A images/labels are never shown again.
    3. Re-measure test accuracy on Task A digits. The drop is "forgetting."

WHY RANDOM IS A BUILT-IN CONTROL FOR THIS EXPERIMENT:
  Random's hidden layer is frozen and never trained at all -- only its
  small readout layer trains. So ANY forgetting Random shows comes
  purely from the readout retraining on new classes. If Hebbian or
  Backprop forget MORE than Random, that extra forgetting must be coming
  from their hidden-layer representations actually shifting -- which is
  the real, interesting result.

  For Hebbian specifically, we also save a snapshot of its readout right
  before Task B, and test that OLD readout against the NEW (post-Task-B)
  Hebbian features. That isolates "did the features themselves drift"
  from "did the readout just forget," which the combined before/after
  number can't tell you on its own.

Requires: torch, torchvision, matplotlib, numpy
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output", exist_ok=True)
print("Using device:", device)

WIDTH = 400
SEEDS = [0, 1]  # bump this up if you have time for more rigor
BACKPROP_EPOCHS = 5
HEBB_EPOCHS = 3
READOUT_EPOCHS = 60

TASK_A_DIGITS = [0, 1, 2, 3, 4]
TASK_B_DIGITS = [5, 6, 7, 8, 9]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True)
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True)


def prep(ds):
    raw = ds.data.float() / 255.0
    X = F.normalize(raw.view(len(ds), -1), dim=1)
    return X.to(device), ds.targets.clone().to(device)


X_train_full, y_train_full = prep(train_set)
X_test_full, y_test_full = prep(test_set)


def split_by_digit(X, y, digits):
    mask = torch.isin(y, torch.tensor(digits, device=y.device))
    return X[mask], y[mask]


X_train_A, y_train_A = split_by_digit(X_train_full, y_train_full, TASK_A_DIGITS)
X_train_B, y_train_B = split_by_digit(X_train_full, y_train_full, TASK_B_DIGITS)
X_test_A, y_test_A = split_by_digit(X_test_full, y_test_full, TASK_A_DIGITS)

print(f"Task A (0-4): {X_train_A.size(0)} train, {X_test_A.size(0)} test images")
print(f"Task B (5-9): {X_train_B.size(0)} train images\n")


@torch.no_grad()
def eval_acc(predict_fn, X, y):
    preds = predict_fn(X).argmax(1)
    return (preds.cpu() == y.cpu()).float().mean().item() * 100


def linear_probe(
    feature_fn,
    X_train_probe,
    y_train_probe,
    X_test_probe,
    y_test_probe,
    epochs=READOUT_EPOCHS,
    lr=0.01,
):
    """Freeze whatever the current representation is (post-Task-B), and train
    a FRESH small linear classifier using ONLY Task A data. This measures how
    much Task-A information the representation still contains, independent of
    whatever the model's actual deployed readout currently says -- which lets
    us compare feature retention fairly across all three models, not just Hebbian."""
    with torch.no_grad():
        Htr = feature_fn(X_train_probe)
    probe = nn.Linear(Htr.shape[1], 10).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(probe(Htr), y_train_probe).backward()
        opt.step()
    with torch.no_grad():
        Hte = feature_fn(X_test_probe)
        acc = (
            probe(Hte).argmax(1).cpu() == y_test_probe.cpu()
        ).float().mean().item() * 100
    return acc


# ---------------------------------------------------------------------------
# Backprop: one network, trained end-to-end, continued (not reset) across tasks
# ---------------------------------------------------------------------------
def build_backprop_net(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 10)).to(
        device
    )


def continue_backprop(net, X, y, epochs=BACKPROP_EPOCHS, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(X.size(0), device=device)
        for i in range(0, X.size(0), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            lossf(net(X[idx]), y[idx]).backward()
            opt.step()
    return net


# ---------------------------------------------------------------------------
# Hebbian hidden layer + readout, both continued (not reset) across tasks
# ---------------------------------------------------------------------------
def init_hebbian(seed):
    torch.manual_seed(seed)
    idx = torch.randint(0, X_train_full.size(0), (WIDTH,))
    return F.normalize(X_train_full[idx].clone(), dim=1)


def continue_hebbian(W, X, epochs=HEBB_EPOCHS, lr=0.05):
    """Unsupervised -- never touches y. Used for both A and B."""
    for _ in range(epochs):
        perm = torch.randperm(X.size(0), device=device)
        for i in range(0, X.size(0), 256):
            x = X[perm[i : i + 256]]
            winners = (x @ W.t()).argmax(1)
            sum_x = torch.zeros_like(W)
            counts = torch.zeros(WIDTH, device=device)
            sum_x.index_add_(0, winners, x)
            counts.index_add_(0, winners, torch.ones_like(winners, dtype=torch.float))
            won = counts > 0
            W[won] = F.normalize(
                W[won] + lr * (sum_x[won] / counts[won].unsqueeze(1) - W[won]), dim=1
            )
    return W


def build_readout(seed):
    torch.manual_seed(seed)
    return nn.Linear(WIDTH, 10).to(device)


def continue_readout(ro, W, X, y, epochs=READOUT_EPOCHS, lr=0.01):
    H = F.relu(X @ W.t())
    opt = torch.optim.Adam(ro.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(ro(H), y).backward()
        opt.step()
    return ro


# ---------------------------------------------------------------------------
# Random: hidden layer is frozen from the start -- our built-in control
# ---------------------------------------------------------------------------
def random_init(seed):
    torch.manual_seed(seed)
    return F.normalize(torch.randn(WIDTH, 784, device=device), dim=1)


# ===========================================================================
# Run the experiment across seeds
# ===========================================================================
records = {
    "backprop": [],
    "hebbian": [],
    "hebbian_repr_only": [],
    "random": [],
    "backprop_probe": [],
    "hebbian_probe": [],
    "random_probe": [],
}

for seed in SEEDS:
    print(f"--- seed {seed} ---")

    # ---------------- Backprop ----------------
    net = build_backprop_net(seed)
    net = continue_backprop(net, X_train_A, y_train_A)
    acc_before = eval_acc(lambda X: net(X), X_test_A, y_test_A)
    net = continue_backprop(net, X_train_B, y_train_B)
    acc_after = eval_acc(lambda X: net(X), X_test_A, y_test_A)
    records["backprop"].append((acc_before, acc_after))
    print(
        f"  Backprop : before={acc_before:.2f}%  after={acc_after:.2f}%  "
        f"forgotten={acc_before - acc_after:.2f} pts"
    )

    backprop_hidden = lambda X: net[1](
        net[0](X)
    )  # ReLU(Linear1(X)) -- the hidden representation
    acc_probe_b = linear_probe(
        backprop_hidden, X_train_A, y_train_A, X_test_A, y_test_A
    )
    records["backprop_probe"].append(acc_probe_b)
    print(
        f"             (fresh linear probe on post-Task-B hidden layer: {acc_probe_b:.2f}% "
        f"-- how much Task A info is still recoverable from the representation)"
    )

    # ---------------- Hebbian ----------------
    Wh = init_hebbian(seed)
    Wh = continue_hebbian(Wh, X_train_A)
    roH = build_readout(seed)
    roH = continue_readout(roH, Wh, X_train_A, y_train_A)
    acc_before_h = eval_acc(lambda X: roH(F.relu(X @ Wh.t())), X_test_A, y_test_A)

    roH_snapshot = copy.deepcopy(roH)  # readout as it was, BEFORE seeing Task B
    Wh = continue_hebbian(Wh, X_train_B)  # features keep adapting, unsupervised
    roH = continue_readout(
        roH, Wh, X_train_B, y_train_B
    )  # readout retrains on new classes

    acc_after_h = eval_acc(lambda X: roH(F.relu(X @ Wh.t())), X_test_A, y_test_A)
    # Isolates feature drift: OLD readout, but on the NEW (post-Task-B) features
    acc_repr_h = eval_acc(
        lambda X: roH_snapshot(F.relu(X @ Wh.t())), X_test_A, y_test_A
    )

    records["hebbian"].append((acc_before_h, acc_after_h))
    records["hebbian_repr_only"].append(acc_repr_h)
    print(
        f"  Hebbian  : before={acc_before_h:.2f}%  after={acc_after_h:.2f}%  "
        f"forgotten={acc_before_h - acc_after_h:.2f} pts"
    )
    print(
        f"             (old readout + drifted features only: {acc_repr_h:.2f}% "
        f"-- isolates feature drift from readout forgetting)"
    )

    hebbian_feat = lambda X: F.relu(X @ Wh.t())
    acc_probe_h = linear_probe(hebbian_feat, X_train_A, y_train_A, X_test_A, y_test_A)
    records["hebbian_probe"].append(acc_probe_h)
    print(
        f"             (fresh linear probe on post-Task-B Hebbian features: {acc_probe_h:.2f}%)"
    )

    # ---------------- Random ----------------
    Wr = random_init(seed)  # never trained, ever -- the control
    roR = build_readout(seed)
    roR = continue_readout(roR, Wr, X_train_A, y_train_A)
    acc_before_r = eval_acc(lambda X: roR(F.relu(X @ Wr.t())), X_test_A, y_test_A)
    roR = continue_readout(roR, Wr, X_train_B, y_train_B)
    acc_after_r = eval_acc(lambda X: roR(F.relu(X @ Wr.t())), X_test_A, y_test_A)
    records["random"].append((acc_before_r, acc_after_r))
    print(
        f"  Random   : before={acc_before_r:.2f}%  after={acc_after_r:.2f}%  "
        f"forgotten={acc_before_r - acc_after_r:.2f} pts"
    )

    random_feat = lambda X: F.relu(
        X @ Wr.t()
    )  # Wr never trains, so this is unchanged from before
    acc_probe_r = linear_probe(random_feat, X_train_A, y_train_A, X_test_A, y_test_A)
    records["random_probe"].append(acc_probe_r)
    print(
        f"             (fresh linear probe on Random's [unchanged] features: {acc_probe_r:.2f}% "
        f"-- sanity check, should be close to 'before')\n"
    )


# ===========================================================================
# Average across seeds and plot
# ===========================================================================
def avg_before_after(key):
    arr = np.array(records[key])  # shape (n_seeds, 2)
    return arr[:, 0].mean(), arr[:, 1].mean()


b_before, b_after = avg_before_after("backprop")
h_before, h_after = avg_before_after("hebbian")
r_before, r_after = avg_before_after("random")
repr_only_mean = float(np.mean(records["hebbian_repr_only"]))

print("=== Averaged over seeds ===")
print(
    f"Backprop : before={b_before:.2f}%  after={b_after:.2f}%  forgotten={b_before - b_after:.2f} pts"
)
print(
    f"Hebbian  : before={h_before:.2f}%  after={h_after:.2f}%  forgotten={h_before - h_after:.2f} pts"
)
print(f"  (feature-drift-only accuracy: {repr_only_mean:.2f}%)")
print(
    f"Random   : before={r_before:.2f}%  after={r_after:.2f}%  forgotten={r_before - r_after:.2f} pts"
)

print("\n=== Linear probe comparison ===")
print(
    "How much Task A info is still recoverable from each representation after Task B,"
)
print("using a FRESH classifier -- independent of readout collapse:")
print(f"  Backprop probe : {np.mean(records['backprop_probe']):.2f}%")
print(f"  Hebbian probe  : {np.mean(records['hebbian_probe']):.2f}%")
print(
    f"  Random probe   : {np.mean(records['random_probe']):.2f}%  (sanity check, ~= Random 'before')"
)

models = ["Backprop", "Hebbian", "Random"]
before = [b_before, h_before, r_before]
after = [b_after, h_after, r_after]
forgotten = [b_before - b_after, h_before - h_after, r_before - r_after]

x = np.arange(len(models))
width = 0.3
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7, 6.5), gridspec_kw={"height_ratios": [2, 1]}
)

VISIBLE_MIN = 3.0  # minimum bar height for visibility of tiny after bars

# --- Top panel: before vs after ---
ax1.bar(
    x - width / 2,
    before,
    width,
    label="Before learning Task B",
    color="#4C72B0",
)
after_bars = ax1.bar(
    x + width / 2,
    [max(v, VISIBLE_MIN) for v in after],
    width,
    label="After learning Task B",
    color="#C44E52",
    hatch="////",
    edgecolor="#C44E52",
    linewidth=0.5,
)
for i, (bv, av) in enumerate(zip(before, after)):
    ax1.text(i - width / 2, bv + 1, f"{bv:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.text(
        i + width / 2,
        max(av, VISIBLE_MIN) + 1,
        f"{av:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel("Test accuracy on Task A (%)")
ax1.set_ylim(0, 110)
ax1.set_title(
    f"Catastrophic forgetting: does learning new digits erase old ones?\n(mean over {len(SEEDS)} seeds)"
)
ax1.legend(fontsize=8)

# --- Bottom panel: forgotten amount ---
bars = ax2.bar(x, forgotten, width * 1.5, color="#C44E52")
for bar, val in zip(bars, forgotten):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f} pts",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel("Accuracy forgotten (pts)")
ax2.set_ylim(0, 105)
ax2.set_title("Accuracy drop (before − after)")

plt.tight_layout()
plt.savefig("output/catastrophic_forgetting.png", dpi=150)
plt.close()

print("\nSaved: output/catastrophic_forgetting.png")

# ===========================================================================
# Second figure: zoomed-in view of feature retention (linear probe)
# ===========================================================================
bp_probe = float(np.mean(records["backprop_probe"]))
hb_probe = float(np.mean(records["hebbian_probe"]))
rn_probe = float(np.mean(records["random_probe"]))

fig2, ax = plt.subplots(figsize=(7, 5))
x2 = np.arange(len(models))
width2 = 0.25

ax.bar(
    x2 - width2 / 2, before, width2, label="Before Task B (baseline)", color="#4C72B0"
)
ax.bar(
    x2 + width2 / 2,
    [bp_probe, hb_probe, rn_probe],
    width2,
    label="Linear probe on post-Task-B features",
    color="#55A868",
)

for i, (bv, pv) in enumerate(zip(before, [bp_probe, hb_probe, rn_probe])):
    ax.text(
        i - width2 / 2, bv + 0.3, f"{bv:.1f}%", ha="center", va="bottom", fontsize=9
    )
    ax.text(
        i + width2 / 2,
        pv + 0.3,
        f"{pv:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax.set_xticks(x2)
ax.set_xticklabels(models)
ax.set_ylabel("Test accuracy on Task A (%)")
ax.set_ylim(85, 101)
ax.set_title(
    "Feature retention: how much Task A info survives in the\nrepresentation after Task B training?"
)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("output/catastrophic_forgetting_feature_retention.png", dpi=150)
plt.close()
print("Saved: output/catastrophic_forgetting_feature_retention.png")
