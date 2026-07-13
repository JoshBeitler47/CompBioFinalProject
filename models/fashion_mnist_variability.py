"""
fashion_mnist_variability.py
-----------------------------
Same 4-condition, multi-seed comparison as multi_seed_variability.py, but on
Fashion-MNIST instead of MNIST.

History: an earlier version of the Hebbian hidden layer (winner-take-all
competition, see common.hebbian_W1's mode="competitive") consistently scored
a couple points BELOW a frozen-random hidden layer on MNIST, and neither
more epochs, an annealed learning rate, nor a conscience mechanism aimed at
"dead units" closed that gap. The current lateral-inhibition + homeostasis
rule (mode="lateral", the default) fixed that on MNIST -- Hebbian now beats
Random by a few points, consistently across seeds. This script is the harder
apples-to-apples check: does the same rule hold up on a task where raw-pixel
similarity is less informative (Fashion-MNIST's classes -- e.g. shirt vs.
pullover vs. coat -- overlap more in pixel space than MNIST digits do)?
Current result: Hebbian and Random come out roughly tied here, not a clear
win either way -- a more honest read than "fixed everywhere."

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
