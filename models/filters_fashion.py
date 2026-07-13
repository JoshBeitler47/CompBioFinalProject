"""
filters_fashion.py
-------------------
Same "what does each hidden unit look for" visualization as the filters
panel in compare_and_visualize_v2.py, but on Fashion-MNIST instead of MNIST.

Why a separate script: compare_and_visualize_v2.py's figures are all MNIST,
and duplicating its whole multi-seed/sample-efficiency pipeline just to get
one more figure on a second dataset would be a lot of unnecessary runtime.
This is a single representative run (one seed, like compare_and_visualize_v2's
own filters.png) of Backprop / Hebbian / Random on Fashion-MNIST, using the
exact same plotting code, via the same common.py helpers.

Point of the comparison: on MNIST (filters.png), each Hebbian unit converges
on a distinct digit-shaped stroke. Here, several units converge on nearly the
SAME generic garment silhouette (e.g. multiple "shirt blob" or "boot blob"
filters) instead of 400 visibly different things. That's a direct visual
version of why Hebbian's edge over Random shrinks on Fashion-MNIST (see
fashion_mnist_variability.py's docstring): several Fashion-MNIST classes
(shirt/pullover/coat/t-shirt) share almost the same silhouette, so the rule
-- which only ever chases the most common recurring pixel pattern, never the
class label -- ends up rediscovering that shared silhouette repeatedly
instead of the finer details that would actually separate the classes.

Outputs: output/filters_fashion.png
Requires: torch, torchvision, matplotlib (same as the rest of this folder).
"""

import os
import sys

import torchvision
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import common as C

device = C.device
os.makedirs("output", exist_ok=True)
print("Using device:", device)

WIDTH = 400
SEED = 0
BACKPROP_EPOCHS = 8

X_train, y_train, X_test, y_test, raw_test, mean = C.load_dataset(torchvision.datasets.FashionMNIST)

print("Training Backprop...")
net = C.build_backprop_net(WIDTH, seed=SEED)
net, _ = C.train_backprop(net, X_train, y_train, epochs=BACKPROP_EPOCHS)

print("Training Hebbian (mode=lateral, the default)...")
Wh = C.hebbian_W1(X_train, WIDTH, seed=SEED)

Wr = C.random_W1(WIDTH, seed=SEED)

results = {
    "Backprop\n(end-to-end)": net[0].weight.detach().cpu(),
    "Hebbian\n+ readout": Wh.W.cpu(),
    "Random\n+ readout": Wr.cpu(),
}
row_labels = ["Backprop", "Hebbian", "Random"]

M = 10
fig, axes = plt.subplots(3, M, figsize=(M * 1.1, 5.0))
for row, (name, W) in enumerate(results.items()):
    for col in range(M):
        f = W[col].view(28, 28)
        v = f.abs().max()
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(f, cmap="seismic", vmin=-v, vmax=v)
    axes[row, 0].set_ylabel(row_labels[row], fontsize=10, labelpad=8)
plt.suptitle("What each hidden unit 'looks for' on Fashion-MNIST (its weight pattern)")
plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.savefig("output/filters_fashion.png", dpi=150)
plt.close()

print("\nSaved: output/filters_fashion.png")
