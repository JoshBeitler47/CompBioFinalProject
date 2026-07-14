"""
multi_seed_variability.py
--------------------------
Answers a question raised while looking at visuals.py:

  "Isn't this supposed to be random? Why do I get the exact same numbers
  every time?" -- Because earlier scripts pinned torch.manual_seed(0), so
  every random draw (weight init, batch order, template init, ...) follows
  the identical sequence on every run. That's deliberate, so the number
  you write in your report is reproducible. This script removes that
  fixed point on purpose: it reruns the SAME Backprop/Hebbian/Random
  comparison under several different seeds and reports the mean +/-
  spread, not one number.

The actual training/plotting logic is shared with fashion_mnist_variability.py
(the same experiment on a harder dataset) via common.run_seed_variability_experiment.

Outputs:
  seed_variability.png -- one faint dot per seed per condition (the actual
  spread) plus a bold mean +/- 1 std error bar on top.

Runtime: ~1.5-2 min per seed x NUM_SEEDS. Lower NUM_SEEDS or the epoch counts
while iterating.
"""

import os
import sys

import torchvision

sys.path.insert(0, os.path.dirname(__file__))
import common as C

NUM_SEEDS = 5  # how many independent random seeds to sample

C.run_seed_variability_experiment(
    torchvision.datasets.MNIST,
    output_name="seed_variability.png",
    title=f"Same experiment, {NUM_SEEDS} different random seeds -- the spread is the point",
    num_seeds=NUM_SEEDS,
)
