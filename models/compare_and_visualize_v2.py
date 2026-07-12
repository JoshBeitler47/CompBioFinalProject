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

Shared data loading (with mean-centering) and the Backprop/Hebbian/Random
training routines live in common.py -- see that file's module docstring for
why the data is centered and why the Hebbian rule here isn't literally Oja's
rule.

Outputs (PNG files in output/):
   1. output/accuracy_comparison.png     -- bar chart with error bars (multi-seed)
   2. output/confusion_matrices.png      -- unchanged from before, single representative run
   3. output/misclassified.png           -- unchanged from before, single representative run
   4. output/filters.png                 -- unchanged from before, single representative run
   5. output/sample_efficiency.png       -- accuracy vs. fraction of labeled data
   6. output/training_curves.png         -- accuracy vs. epoch, backprop vs Hebbian readout

Requires: torch, torchvision, matplotlib
Runtime: meaningfully longer than v1 -- see SEEDS / EFFICIENCY_FRACTIONS below.
Turn those numbers down for a quick first pass, then back up for final results.
"""

import os
import sys

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C

device = C.device
os.makedirs("output", exist_ok=True)
print("Using device:", device)

WIDTH = 400  # shared hidden-layer width, identical for all 3 models
SEEDS = [0, 1, 2]  # how many times to repeat the main comparison
EFFICIENCY_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
EFFICIENCY_SEEDS = [0, 1]  # fewer seeds for the sweep, to keep runtime sane
BACKPROP_EPOCHS = 8
READOUT_EPOCHS = 100

# ---------------------------------------------------------------------------
# Data. Loaded once. All models see the same mean-centered, unit-normalized
# 784-vectors (see common.py for why centering matters here).
# ---------------------------------------------------------------------------
X_train_full, y_train_full, X_test, y_test, raw_test, _mean = C.load_dataset(
    torchvision.datasets.MNIST
)
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
# One full pipeline run: all three models, given a seed and a label fraction.
# Returns just the three test accuracies (used by both the seed-averaging
# and the sample-efficiency experiments below).
# ===========================================================================
def run_pipeline(seed, label_fraction, track_curve=False):
    X_sup, y_sup = labeled_subset(label_fraction, seed)

    net = C.build_backprop_net(WIDTH, seed=seed)
    net, curveA = C.train_backprop(
        net, X_sup, y_sup, epochs=BACKPROP_EPOCHS,
        track_curve=track_curve, X_test=X_test, y_test=y_test,
    )
    with torch.no_grad():
        accA = (net(X_test).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100

    Wh = C.hebbian_W1(X_train_full, WIDTH, seed=seed)  # full unlabeled data, independent of label_fraction
    C.redundancy_check(Wh, "Hebbian", seed)

    roB = C.build_readout(WIDTH, seed=seed)
    HtrB = C.hebbian_features(X_sup, Wh)
    HteB = C.hebbian_features(X_test, Wh)
    roB, curveB_raw = C.train_readout(
        roB, HtrB, y_sup, epochs=READOUT_EPOCHS,
        track_curve=track_curve, H_test=HteB, y_test=y_test,
    )
    curveB = curveB_raw  # list of (epoch, acc)
    with torch.no_grad():
        accB = (roB(HteB).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100

    Wr = C.random_W1(WIDTH, seed=seed)
    C.redundancy_check(Wr, "Random", seed)
    roC = C.build_readout(WIDTH, seed=seed)
    HtrC = C.hebbian_features(X_sup, Wr)
    HteC = C.hebbian_features(X_test, Wr)
    roC, curveC = C.train_readout(roC, HtrC, y_sup, epochs=READOUT_EPOCHS)
    with torch.no_grad():
        accC = (roC(HteC).argmax(1).cpu() == y_test_cpu).float().mean().item() * 100

    extras = dict(
        net=net,
        Wh=Wh,
        Wr=Wr,
        roB=roB,
        roC=roC,
    )
    return dict(
        backprop=accA,
        hebbian=accB,
        random=accC,
        curves=dict(backprop=curveA, hebbian=curveB, random=curveC),
        extras=extras,
    )


# ===========================================================================
# Experiment 1: multi-seed averaged accuracy (replaces the single-run bar chart)
# ===========================================================================
print(f"\n=== Multi-seed comparison ({len(SEEDS)} seeds) ===")
seed_results = {"backprop": [], "hebbian": [], "random": []}
representative_run = None  # we'll keep seed 0's full run for the other figures

for seed in SEEDS:
    r = run_pipeline(seed, label_fraction=1.0, track_curve=(seed == SEEDS[0]))
    seed_results["backprop"].append(r["backprop"])
    seed_results["hebbian"].append(r["hebbian"])
    seed_results["random"].append(r["random"])
    print(
        f"  seed {seed}: backprop={r['backprop']:.2f}%  hebbian={r['hebbian']:.2f}%  random={r['random']:.2f}%"
    )
    if seed == SEEDS[0]:
        representative_run = r

means = {k: float(np.mean(v)) for k, v in seed_results.items()}
stds = {k: float(np.std(v)) for k, v in seed_results.items()}
print("Means:", means)
print("Stds: ", stds)

# ---- Figure 1: accuracy bar chart with error bars ----
labels = ["Backprop\n(end-to-end)", "Hebbian\n+ readout", "Random\n+ readout"]
keys = ["backprop", "hebbian", "random"]
plt.figure(figsize=(6, 4))
bars = plt.bar(
    labels,
    [means[k] for k in keys],
    yerr=[stds[k] for k in keys],
    capsize=6,
    color=["#4C72B0", "#55A868", "#C44E52"],
)
for b, k in zip(bars, keys):
    plt.text(
        b.get_x() + b.get_width() / 2,
        means[k] + stds[k] + 1,
        f"{means[k]:.1f}%",
        ha="center",
        fontweight="bold",
    )
plt.ylabel("Test accuracy (%)")
plt.ylim(0, 100)
plt.title(
    f"MNIST: does it matter HOW the hidden layer learns?\n(mean ± std over {len(SEEDS)} seeds)"
)
plt.tight_layout()
plt.savefig("output/accuracy_comparison.png", dpi=150)
plt.close()


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
    print(
        f"  fraction={frac}: " + "  ".join(f"{k}={sweep[k][frac]:.2f}%" for k in accs)
    )

plt.figure(figsize=(6, 4))
for k, color, label in zip(keys, ["#4C72B0", "#55A868", "#C44E52"], labels):
    fracs = sorted(sweep[k].keys())
    plt.plot(
        fracs,
        [sweep[k][f] for f in fracs],
        marker="o",
        color=color,
        label=label.replace("\n", " "),
    )
plt.xlabel("Fraction of labeled training data used")
plt.ylabel("Test accuracy (%)")
plt.title("Sample efficiency: how much labeled data does each model need?")
plt.legend()
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("output/sample_efficiency.png", dpi=150)
plt.close()


# ===========================================================================
# Experiment 3: training curves (from the representative seed-0 run above)
# ===========================================================================
plt.figure(figsize=(6, 4))
backprop_curve = representative_run["curves"]["backprop"]  # list of acc per epoch
hebbian_curve = representative_run["curves"]["hebbian"]  # list of (epoch, acc)

plt.plot(
    range(1, len(backprop_curve) + 1),
    backprop_curve,
    marker="o",
    color="#4C72B0",
    label="Backprop (per epoch)",
)
if hebbian_curve:
    epochs_h, accs_h = zip(*hebbian_curve)
    plt.plot(epochs_h, accs_h, marker="s", color="#55A868", label="Hebbian readout")

plt.xlabel("Training epoch")
plt.ylabel("Test accuracy (%)")
plt.title("How fast does each model converge?")
plt.legend()
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("output/training_curves.png", dpi=150)
plt.close()


# ===========================================================================
# Figures 4-6: confusion matrices / misclassified / filters, from the SAME
# representative (seed 0) run, so all figures describe one consistent run.
# ===========================================================================
extras = representative_run["extras"]
net, Wh, Wr, roB, roC = (
    extras["net"],
    extras["Wh"],
    extras["Wr"],
    extras["roB"],
    extras["roC"],
)

with torch.no_grad():
    predsA = net(X_test).argmax(1).cpu()
    predsB = roB(C.hebbian_features(X_test, Wh)).argmax(1).cpu()
    predsC = roC(C.hebbian_features(X_test, Wr)).argmax(1).cpu()

Wh_templates = Wh.W if isinstance(Wh, C.HebbianState) else Wh  # mode="lateral" -> unwrap templates
results = {
    "Backprop\n(end-to-end)": dict(preds=predsA, W1=net[0].weight.detach().cpu()),
    "Hebbian\n+ readout": dict(preds=predsB, W1=Wh_templates.cpu()),
    "Random\n+ readout": dict(preds=predsC, W1=Wr.cpu()),
}
for name, r in results.items():
    r["acc"] = (r["preds"] == y_test_cpu).float().mean().item() * 100

names = list(results.keys())
row_labels = ["Backprop", "Hebbian", "Random"]


# ---- confusion matrices ----
def error_matrix(preds):
    cm = (
        torch.bincount((y_test_cpu * 10 + preds), minlength=100).reshape(10, 10).float()
    )
    cm.fill_diagonal_(0)
    return cm


err = {n: error_matrix(results[n]["preds"]) for n in names}
vmax = max(m.max().item() for m in err.values())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, n in zip(axes, names):
    im = ax.imshow(err[n], cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(f"{n}  ({results[n]['acc']:.1f}%)")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
fig.colorbar(im, ax=axes, shrink=0.8, label="# of mistakes")
plt.savefig("output/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- misclassified examples ----
N = 8
fig, axes = plt.subplots(3, N, figsize=(N * 1.15, 5.0))
for row, n in enumerate(names):
    preds = results[n]["preds"]
    wrong = (preds != y_test_cpu).nonzero(as_tuple=True)[0][:N]
    for col in range(N):
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        if col < len(wrong):
            wi = wrong[col].item()
            ax.imshow(raw_test[wi], cmap="gray")
            ax.set_title(f"{y_test_cpu[wi].item()}→{preds[wi].item()}", fontsize=9)
    axes[row, 0].set_ylabel(row_labels[row], fontsize=10, labelpad=8)
plt.suptitle("Digits each model gets wrong  (true → predicted)")
plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.savefig("output/misclassified.png", dpi=150)
plt.close()

# ---- learned filters ----
M = 10
fig, axes = plt.subplots(3, M, figsize=(M * 1.1, 5.0))
for row, n in enumerate(names):
    W = results[n]["W1"]
    for col in range(M):
        f = W[col].view(28, 28)
        v = f.abs().max()
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(f, cmap="seismic", vmin=-v, vmax=v)
    axes[row, 0].set_ylabel(row_labels[row], fontsize=10, labelpad=8)
plt.suptitle("What each hidden unit 'looks for' (its weight pattern)")
plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.savefig("output/filters.png", dpi=150)
plt.close()

print(
    "\nSaved: output/accuracy_comparison.png, output/sample_efficiency.png, output/training_curves.png,"
)
print(
    "       output/confusion_matrices.png, output/misclassified.png, output/filters.png"
)
