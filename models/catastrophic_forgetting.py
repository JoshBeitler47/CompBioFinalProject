"""
catastrophic_forgetting.py
---------------------------
Tests whether backprop or Hebbian-learned representations "forget" old
knowledge when trained on new data they never revisit.

THE EXPERIMENT:
  Task A = digits 0-4.  Task B = digits 5-9.

  For each model (Backprop, Hebbian+readout, Random+readout, and sparse
  variants of Backprop/Hebbian -- see below):
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

SPARSE VS. DENSE CONTROL (the "rigorous version" from the project notes):
  It's not obvious that a Hebbian model forgets less BECAUSE it's Hebbian --
  what actually protects against forgetting in the brain is largely sparse,
  non-overlapping representations (so old and new tasks don't activate the
  same units), not the local learning rule per se. To disentangle "the rule"
  from "sparsity," we add sparse variants of Backprop and Hebbian that force
  only the top-K most active hidden units through to the readout (a k-winners
  -take-all mask, independent of which learning rule produced the
  activations). If sparsity -- not the rule -- is what's doing the work,
  Sparse Backprop should forget less than Dense Backprop by roughly the same
  margin Sparse Hebbian improves over Dense Hebbian.

Requires: torch, torchvision, matplotlib, numpy
"""

import copy
import os
import sys

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C

device = C.device
os.makedirs("output", exist_ok=True)
print("Using device:", device)

WIDTH = 400
SPARSITY_K = 40  # top-40-of-400 (10%) active units for the sparse conditions
SEEDS = [0, 1]  # bump this up if you have time for more rigor
BACKPROP_EPOCHS = 5
HEBB_EPOCHS = 3
READOUT_EPOCHS = 60

TASK_A_DIGITS = [0, 1, 2, 3, 4]
TASK_B_DIGITS = [5, 6, 7, 8, 9]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
X_train_full, y_train_full, X_test_full, y_test_full, _raw_test, _mean = C.load_dataset(
    torchvision.datasets.MNIST
)


def split_by_digit(X, y, digits):
    mask = torch.isin(y, torch.tensor(digits, device=y.device))
    return X[mask], y[mask]


X_train_A, y_train_A = split_by_digit(X_train_full, y_train_full, TASK_A_DIGITS)
X_train_B, y_train_B = split_by_digit(X_train_full, y_train_full, TASK_B_DIGITS)
X_test_A, y_test_A = split_by_digit(X_test_full, y_test_full, TASK_A_DIGITS)

print(f"Task A (0-4): {X_train_A.size(0)} train, {X_test_A.size(0)} test images")
print(f"Task B (5-9): {X_train_B.size(0)} train images\n")


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
    us compare feature retention fairly across all models, not just Hebbian."""
    with torch.no_grad():
        Htr = feature_fn(X_train_probe)
    probe = nn.Linear(Htr.shape[1], 10).to(device)
    probe, _ = C.train_readout(probe, Htr, y_train_probe, epochs=epochs, lr=lr)
    with torch.no_grad():
        Hte = feature_fn(X_test_probe)
        acc = (probe(Hte).argmax(1).cpu() == y_test_probe.cpu()).float().mean().item() * 100
    return acc


# ===========================================================================
# Run the experiment across seeds
# ===========================================================================
CONDITIONS = ["backprop", "backprop_sparse", "hebbian", "hebbian_sparse", "random"]
records = {c: [] for c in CONDITIONS}
records["hebbian_repr_only"] = []
records["hebbian_sparse_repr_only"] = []
probe_records = {c: [] for c in CONDITIONS}

for seed in SEEDS:
    print(f"--- seed {seed} ---")

    # ---------------- Backprop: dense and sparse ----------------
    for cond, k in [("backprop", None), ("backprop_sparse", SPARSITY_K)]:
        net = C.build_backprop_net(WIDTH, seed=seed)
        net, _ = C.train_backprop(net, X_train_A, y_train_A, epochs=BACKPROP_EPOCHS, sparsity_k=k)
        acc_before = C.eval_acc(lambda X: C.backprop_forward(net, X, k), X_test_A, y_test_A)
        net, _ = C.train_backprop(net, X_train_B, y_train_B, epochs=BACKPROP_EPOCHS, sparsity_k=k)
        acc_after = C.eval_acc(lambda X: C.backprop_forward(net, X, k), X_test_A, y_test_A)
        records[cond].append((acc_before, acc_after))
        print(
            f"  {cond:16s}: before={acc_before:.2f}%  after={acc_after:.2f}%  "
            f"forgotten={acc_before - acc_after:.2f} pts"
        )

        hidden_fn = lambda X, net=net, k=k: C.topk_mask(net[1](net[0](X)), k)
        acc_probe = linear_probe(hidden_fn, X_train_A, y_train_A, X_test_A, y_test_A)
        probe_records[cond].append(acc_probe)
        print(f"                    (fresh linear probe on post-Task-B hidden layer: {acc_probe:.2f}%)")

    # ---------------- Hebbian: dense and sparse (same trained state, different readout view) ----------------
    # The feature layer is unsupervised and shared between the dense and
    # sparse conditions -- only the readout differs -- so it's trained ONCE
    # per task stage and both readouts read off the SAME state. This matters:
    # if each condition re-ran hebbian_W1 independently, "before" and "after"
    # would silently be measured against different feature layers whose only
    # difference is incidental training-order noise. (hebbian_W1 defaults to
    # mode="lateral" -- Foldiak anti-Hebbian inhibition + homeostasis -- which
    # returns a single HebbianState bundling W/L/theta, carried across the
    # Task-A -> Task-B continuation via state_init below.)
    Wh_A = C.hebbian_W1(X_train_A, WIDTH, epochs=HEBB_EPOCHS, seed=seed)

    hebbian_readouts, hebbian_before, hebbian_snapshots = {}, {}, {}
    for cond, k in [("hebbian", None), ("hebbian_sparse", SPARSITY_K)]:
        ro = C.build_readout(WIDTH, seed=seed)
        Htr = C.hebbian_features(X_train_A, Wh_A, sparsity_k=k)
        ro, _ = C.train_readout(ro, Htr, y_train_A, epochs=READOUT_EPOCHS)
        hebbian_before[cond] = C.eval_acc(lambda X: ro(C.hebbian_features(X, Wh_A, k)), X_test_A, y_test_A)
        hebbian_readouts[cond] = ro
        hebbian_snapshots[cond] = copy.deepcopy(ro)  # readout as it was, BEFORE seeing Task B

    Wh_B = C.hebbian_W1(X_train_B, WIDTH, epochs=HEBB_EPOCHS, seed=seed, state_init=Wh_A)

    for cond, k in [("hebbian", None), ("hebbian_sparse", SPARSITY_K)]:
        ro = hebbian_readouts[cond]
        Htr_b = C.hebbian_features(X_train_B, Wh_B, sparsity_k=k)
        ro, _ = C.train_readout(ro, Htr_b, y_train_B, epochs=READOUT_EPOCHS)

        acc_before = hebbian_before[cond]
        acc_after = C.eval_acc(lambda X: ro(C.hebbian_features(X, Wh_B, k)), X_test_A, y_test_A)
        acc_repr = C.eval_acc(
            lambda X: hebbian_snapshots[cond](C.hebbian_features(X, Wh_B, k)), X_test_A, y_test_A
        )

        records[cond].append((acc_before, acc_after))
        records[f"{cond}_repr_only"].append(acc_repr)
        print(
            f"  {cond:16s}: before={acc_before:.2f}%  after={acc_after:.2f}%  "
            f"forgotten={acc_before - acc_after:.2f} pts  "
            f"(old readout + drifted features only: {acc_repr:.2f}%)"
        )

        feat_fn = lambda X, Wh_B=Wh_B, k=k: C.hebbian_features(X, Wh_B, k)
        acc_probe = linear_probe(feat_fn, X_train_A, y_train_A, X_test_A, y_test_A)
        probe_records[cond].append(acc_probe)
        print(f"                    (fresh linear probe on post-Task-B features: {acc_probe:.2f}%)")

    # ---------------- Random ----------------
    Wr = C.random_W1(WIDTH, seed=seed)  # never trained, ever -- the control
    roR = C.build_readout(WIDTH, seed=seed)
    HtrR = C.hebbian_features(X_train_A, Wr)
    roR, _ = C.train_readout(roR, HtrR, y_train_A, epochs=READOUT_EPOCHS)
    acc_before_r = C.eval_acc(lambda X: roR(C.hebbian_features(X, Wr)), X_test_A, y_test_A)
    HtrR_b = C.hebbian_features(X_train_B, Wr)
    roR, _ = C.train_readout(roR, HtrR_b, y_train_B, epochs=READOUT_EPOCHS)
    acc_after_r = C.eval_acc(lambda X: roR(C.hebbian_features(X, Wr)), X_test_A, y_test_A)
    records["random"].append((acc_before_r, acc_after_r))
    print(
        f"  random          : before={acc_before_r:.2f}%  after={acc_after_r:.2f}%  "
        f"forgotten={acc_before_r - acc_after_r:.2f} pts"
    )

    random_feat = lambda X: C.hebbian_features(X, Wr)  # Wr never trains -- unchanged from before
    acc_probe_r = linear_probe(random_feat, X_train_A, y_train_A, X_test_A, y_test_A)
    probe_records["random"].append(acc_probe_r)
    print(
        f"                    (fresh linear probe on Random's [unchanged] features: {acc_probe_r:.2f}% "
        f"-- sanity check, should be close to 'before')\n"
    )


# ===========================================================================
# Average across seeds and plot
# ===========================================================================
def avg_before_after(key):
    arr = np.array(records[key])  # shape (n_seeds, 2)
    return arr[:, 0].mean(), arr[:, 1].mean()


print("=== Averaged over seeds ===")
means = {}
for c in CONDITIONS:
    before, after = avg_before_after(c)
    means[c] = (before, after)
    print(f"{c:16s}: before={before:.2f}%  after={after:.2f}%  forgotten={before - after:.2f} pts")

print(f"  (Hebbian feature-drift-only accuracy: {np.mean(records['hebbian_repr_only']):.2f}%)")
print(
    f"  (Hebbian-sparse feature-drift-only accuracy: {np.mean(records['hebbian_sparse_repr_only']):.2f}%)"
)

print("\n=== Sparse vs. dense: does sparsity, not the rule, explain forgetting? ===")
bp_forgot = means["backprop"][0] - means["backprop"][1]
bp_sparse_forgot = means["backprop_sparse"][0] - means["backprop_sparse"][1]
heb_forgot = means["hebbian"][0] - means["hebbian"][1]
heb_sparse_forgot = means["hebbian_sparse"][0] - means["hebbian_sparse"][1]
print(f"Backprop forgetting:        dense={bp_forgot:.2f} pts   sparse={bp_sparse_forgot:.2f} pts")
print(f"Hebbian forgetting:         dense={heb_forgot:.2f} pts   sparse={heb_sparse_forgot:.2f} pts")
print(
    "If sparsity is doing the work, both rows should show sparse < dense by a similar margin;\n"
    "if it's the learning rule, only the Hebbian row should improve substantially with sparsity."
)

print("\n=== Linear probe comparison ===")
print("How much Task A info is still recoverable from each representation after Task B,")
print("using a FRESH classifier -- independent of readout collapse:")
for c in CONDITIONS:
    print(f"  {c:16s}: {np.mean(probe_records[c]):.2f}%")

# ---- Figure 1: before/after bars for all 5 conditions ----
labels = ["Backprop", "Backprop\n(sparse)", "Hebbian", "Hebbian\n(sparse)", "Random"]
before = [means[c][0] for c in CONDITIONS]
after = [means[c][1] for c in CONDITIONS]

x = np.arange(len(CONDITIONS))
width = 0.35
plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, before, width, label="Before learning Task B (digits 5-9)", color="#4C72B0")
plt.bar(x + width / 2, after, width, label="After learning Task B (digits 5-9)", color="#C44E52")
plt.xticks(x, labels)
plt.ylabel("Test accuracy on Task A (digits 0-4), %")
plt.ylim(0, 100)
plt.title(
    f"Catastrophic forgetting: rule vs. sparsity\n(mean over {len(SEEDS)} seeds)"
)
plt.legend()
plt.tight_layout()
plt.savefig("output/catastrophic_forgetting.png", dpi=150)
plt.close()

print("\nSaved: output/catastrophic_forgetting.png")
