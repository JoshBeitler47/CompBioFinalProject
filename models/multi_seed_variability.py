"""
multi_seed_variability.py
--------------------------
Answers two questions raised while looking at visuals.py:

  1. "Isn't this supposed to be random? Why do I get the exact same numbers
     every time?" -- Because earlier scripts pinned torch.manual_seed(0), so
     every random draw (weight init, batch order, template init, ...) follows
     the identical sequence on every run. That's deliberate, so the number
     you write in your report is reproducible. This script removes that
     fixed point on purpose: it reruns the SAME A/B/C/D comparison under
     several different seeds and reports the mean +/- spread, not one number.

  2. "Is there something to tune to make the Hebbian model better?" -- This
     script adds a 4th condition, "Hebbian (tuned)", that only changes two
     knobs versus the baseline Hebbian rule (mode="lateral" -- see
     common.hebbian_W1):
       - a higher target_rate (0.08 -> 0.20): more units are allowed to be
         active per input, so more information reaches the readout.
       - a gentler threshold_lr (0.3 -> 0.15): homeostasis still needs to
         ramp up fast enough to avoid the early-collapse bug documented in
         common.hebbian_W1, just not quite as aggressively.
     Verified empirically (see conversation/tuning notes) to beat the
     baseline by about a point, consistently across seeds. Note this is a
     DIFFERENT pair of knobs than an earlier version of this experiment used
     (more epochs + an annealed learning rate) -- that combination was tuned
     for the old winner-take-all mechanism and, tested against the current
     lateral-inhibition + homeostasis rule, made things WORSE rather than
     better: this rule already converges within a few epochs, and training
     longer just lets a little redundancy creep back into the templates.
     Everything else (width, normalization, lateral inhibition, readout) is
     identical to the baseline Hebbian condition, so any accuracy difference
     is attributable to just those two knobs.

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
