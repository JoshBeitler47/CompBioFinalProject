"""
accuracy_fashion.py
--------------------
Same 3-condition (Backprop / Hebbian / Random) multi-seed bar chart as
compare_and_visualize_v2.py's accuracy_comparison.png, but on Fashion-MNIST.

Why a separate script: compare_and_visualize_v2.py's other figures (sample
efficiency, training curves, confusion matrices, misclassified examples) are
all MNIST-specific and not part of what was asked for here -- this is just
the accuracy bar chart, on the harder dataset, using the same methodology
(3 seeds, same epoch counts) so it's a fair visual counterpart to
accuracy_comparison.png.

Outputs: output/accuracy_fashion.png
Requires: torch, torchvision, matplotlib, numpy (same as the rest of this folder).
"""

import os
import sys

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C

device = C.device
os.makedirs("output", exist_ok=True)
print("Using device:", device)

WIDTH = 400
SEEDS = [0, 1, 2]
BACKPROP_EPOCHS = 8
READOUT_EPOCHS = 100

X_train, y_train, X_test, y_test, raw_test, mean = C.load_dataset(torchvision.datasets.FashionMNIST)

results = {"backprop": [], "hebbian": [], "random": []}

for seed in SEEDS:
    net = C.build_backprop_net(WIDTH, seed=seed)
    net, _ = C.train_backprop(net, X_train, y_train, epochs=BACKPROP_EPOCHS)
    acc_b = C.eval_acc(lambda X: net(X), X_test, y_test)

    Wh = C.hebbian_W1(X_train, WIDTH, seed=seed)
    roH = C.build_readout(WIDTH, seed=seed)
    Htr = C.hebbian_features(X_train, Wh)
    roH, _ = C.train_readout(roH, Htr, y_train, epochs=READOUT_EPOCHS)
    acc_h = C.eval_acc(lambda X: roH(C.hebbian_features(X, Wh)), X_test, y_test)

    Wr = C.random_W1(WIDTH, seed=seed)
    roR = C.build_readout(WIDTH, seed=seed)
    HtrR = C.hebbian_features(X_train, Wr)
    roR, _ = C.train_readout(roR, HtrR, y_train, epochs=READOUT_EPOCHS)
    acc_r = C.eval_acc(lambda X: roR(C.hebbian_features(X, Wr)), X_test, y_test)

    results["backprop"].append(acc_b)
    results["hebbian"].append(acc_h)
    results["random"].append(acc_r)
    print(f"  seed {seed}: backprop={acc_b:.2f}%  hebbian={acc_h:.2f}%  random={acc_r:.2f}%")

means = {k: float(np.mean(v)) for k, v in results.items()}
stds = {k: float(np.std(v)) for k, v in results.items()}
print("Means:", means)
print("Stds: ", stds)

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
    f"Fashion-MNIST: does it matter HOW the hidden layer learns?\n(mean ± std over {len(SEEDS)} seeds)"
)
plt.tight_layout()
plt.savefig("output/accuracy_fashion.png", dpi=150)
plt.close()

print("\nSaved: output/accuracy_fashion.png")
