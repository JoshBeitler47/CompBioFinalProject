"""
fashion_mnist_variability.py
-----------------------------
Same 4-condition, multi-seed comparison as multi_seed_variability.py, but on
Fashion-MNIST instead of MNIST.

Why: on MNIST, competitive Hebbian learning consistently scored a couple
points BELOW a frozen-random hidden layer, and neither more epochs nor an
annealed learning rate closed that gap (see multi_seed_variability.py). A
diagnostic ruled out "dead units" as the cause -- all 400 templates get
used roughly evenly (see common.hebbian_W1's conscience mechanism, added to
push further on exactly this). That leaves task difficulty as the likely
explanation: MNIST is "linearly easy," so raw-pixel cosine-similarity
clusters, learned features, and random projections can all land in the same
ballpark, with only backprop clearly ahead. Fashion-MNIST (same 28x28
grayscale shape, harder classes -- e.g. shirt vs. pullover vs. coat) is the
drop-in swap that gives feature quality more room to matter.

The actual training/plotting logic is shared with multi_seed_variability.py
via common.run_seed_variability_experiment -- this file only supplies the
dataset class and output filename, so this is a genuine apples-to-apples
check of "does the SAME rule behave differently on a harder task."

Outputs: fashion_seed_variability.png
Runtime: ~1.5-2 min per seed x NUM_SEEDS.
"""

import os
import sys

import torchvision

sys.path.insert(0, os.path.dirname(__file__))
import common as C

NUM_SEEDS = 5

C.run_seed_variability_experiment(
    torchvision.datasets.FashionMNIST,
    output_name="fashion_seed_variability.png",
    title=f"Fashion-MNIST, {NUM_SEEDS} seeds -- does a harder task separate the methods?",
    num_seeds=NUM_SEEDS,
)
