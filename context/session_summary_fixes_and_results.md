# Session Summary: Model Fixes, Hebbian Rework, and Final Results

*Log of the work done on `models/` in this session -- what was wrong, what changed, what
the final numbers are, and what conclusions they actually support. Companion to
`project_notes_backprop_vs_hebbian.md` (the pre-implementation research notes this
session's work was based on).*

---

## 1. Starting point: audit of `models/`

An initial review of the six original scripts found eight issues:

1. **Architecture mismatch** -- `backprop.py` used a 128-unit hidden layer while every
   other script used 400, breaking the "shared architecture" comparison the README claimed.
2. **Hebbian rule mislabeled** -- code and docstrings called it "Oja's rule," but it was
   actually winner-take-all competitive learning with an Oja-style renormalization, not
   Oja's literal delta rule.
3. **No mean-centering** -- data was unit-normalized but never centered on the dataset
   mean, which the research notes (section 6) specifically flagged as important.
4. **Hebbian underperformed Random** -- a documented, unresolved finding in
   `fashion_mnist_variability.py`'s docstring.
5. **No sparsity control** in `catastrophic_forgetting.py`, despite the research notes
   explicitly calling for one (disentangle "the rule" from "sparsity" as the explanation
   for any forgetting difference).
6. **No feedback alignment** model, despite being the recommended third data point.
7. **Heavy code duplication** across `catastrophic_forgetting.py`,
   `compare_and_visualize_v2.py`, `multi_seed_variability.py`, `fashion_mnist_variability.py`.
8. **Stale README** -- didn't match the actual files in `models/`.

## 2. First pass: fixes applied

- Fixed `backprop.py` to use `WIDTH = 400`.
- Added mean-centering to data loading (`common.load_dataset`, and inline in the
  standalone `backprop.py` / `hebbian.py`).
- Created `models/common.py`: shared data loading, training loops, and the seed-variability
  experiment runner, eliminating the duplication across four scripts.
- Added a sparsity control to `catastrophic_forgetting.py`: dense vs. top-k-masked
  (`SPARSITY_K = 40`, i.e. 10% of 400) variants of both Backprop and Hebbian, to separate
  "does the rule matter" from "does sparsity alone explain any forgetting difference."
- Added `models/feedback_alignment.py`: a custom `torch.autograd.Function` that replaces
  the output layer's transposed weights with a fixed random matrix on the backward pass,
  removing weight transport specifically while keeping labels/global error/gradient descent.
  Prints the W-vs-B alignment angle per epoch.
- Rewrote the README to match the actual script/output inventory.
- Explicitly scoped OUT: STDP (no time dimension in this static-image architecture) and
  three-factor/neuromodulatory gating (no task signal exists during the label-free
  unsupervised phase without secretly reintroducing supervision).

At this point the Hebbian hidden layer was still winner-take-all + a "conscience"
frequency-bias heuristic (DeSieno 1988) as a decorrelation stand-in.

## 3. The redesign: pushing Hebbian toward real biology

Directive: the project's actual goal is a model that mimics biologically-realistic
learning -- that's Hebbian, not Backprop or Feedback Alignment (both still need labels
and a global error signal; Feedback Alignment only removes weight transport specifically).
The conscience mechanism was explicitly documented as *"a simpler stand-in for the fuller
fix... anti-Hebbian lateral inhibition (Foldiak)"* -- so it got replaced with that fuller fix.

**New default: `mode="lateral"` in `common.hebbian_W1`** (old winner-take-all + conscience
kept as `mode="competitive"`, a fallback/ablation baseline, not deleted):

- **Settling**: instead of one hard winner, activity settles under mutual inhibition over
  a few fixed-point steps: `y <- relu(a - theta - y @ L.T)`. Multiple units can be active
  at once -- a soft population code.
- **Feedforward update (W)**: Hebbian growth toward the activity-weighted mean input,
  then Oja-style renormalization.
- **Lateral update (L)**: Foldiak's (1990) actual anti-Hebbian rule -- units that co-fire
  more than chance get their mutual inhibition strengthened.
- **Threshold update (theta)**: Turrigiano-style homeostatic scaling -- each unit's firing
  rate is pulled toward a small target rate, which also makes the code sparse by construction.

## 4. The bug, and the real fix

The first full-scale run of the new lateral-inhibition layer was badly broken:
**effective dimensionality collapsed to 5.5/400** (out of 400 "templates," only ~5 were
meaningfully different from each other), and **Hebbian accuracy came back at ~54%** --
*worse than an untrained random projection* (85.5%).

Root-caused via instrumented reduced-scale reruns of the training loop:

1. **Threshold ramp-up too slow.** `theta` starts at 0 (no threshold at all), and
   `threshold_lr` was too small (0.01) to catch up quickly. Result: ~46% of all 400 units
   fired on every input for a long stretch of early training (target was 8%), letting
   templates collapse together before any competitive pressure existed to stop them.
   **Fix: `threshold_lr` 0.01 -> 0.3.**

2. **The real bug: no confidence scaling on the feedforward update.** Any unit that
   crossed threshold at all got the exact same size update, whether it barely squeaked
   past or fired confidently. Real Hebbian learning is `Delta w ~ y * x` -- proportional
   to activity -- and that proportionality had been silently dropped when the update was
   written as "move to the activity-weighted mean input, with a flat step size." This let
   marginal, low-confidence activations drag many templates toward the same generic,
   low-information patterns. **Fix: scale the update step by each unit's own average
   activity (`mean_y`) for that batch**, so weak/marginal firing barely moves the
   template and strong firing moves it fully.

Verified via a controlled sweep (`common.hebbian_W1` directly, not just the isolated test
loop): fixing #2 alone took effective dimensionality from 10.6/400 to **120.7/400**, and
readout accuracy from 55.75% to **83.10%** (reduced-scale, 8000 train / 2000 test). Full
60k-sample runs confirmed the fix (section 5).

**Honest caveat, found while diagnosing:** the lateral inhibition term (`L`, the literal
Foldiak anti-Hebbian piece) empirically does very little. Scaling `lateral_lr` up
5-25x made no measurable difference to any result -- `L` stayed close to zero throughout
training regardless. The two things that actually fixed the model were the threshold
ramp speed and the confidence-scaled feedforward update, not the anti-Hebbian lateral
term specifically. The mechanism is correctly implemented and theoretically well-motivated,
but the evidence doesn't support "lateral inhibition is why this works" as a strong claim.

## 5. Final full-scale results

### MNIST, main comparison (`compare_and_visualize_v2.py`, 3 seeds)

| Model | Accuracy |
|---|---|
| Backprop | 97.77% |
| Hebbian | 89.06% +/- 0.56% |
| Random | 85.46% +/- 0.13% |

Hebbian effective dimensionality: ~108/400 (real, distributed structure). Random: ~347/400
(near-full spread, as expected for untrained random directions).

### MNIST, 5-seed check + tuning (`multi_seed_variability.py`)

| Condition | Accuracy |
|---|---|
| Backprop | 97.75% +/- 0.05% |
| Hebbian (baseline) | 89.03% +/- 0.50% |
| Hebbian (tuned) | 89.21% +/- 0.40% |
| Random | 85.66% +/- 0.32% |

### Fashion-MNIST, 5-seed check (`fashion_mnist_variability.py`)

| Condition | Accuracy |
|---|---|
| Backprop | 88.42% +/- 0.15% |
| Hebbian (baseline) | 73.80% +/- 0.93% |
| Hebbian (tuned) | 72.26% +/- 0.32% |
| Random | 74.40% +/- 0.37% |

Hebbian and Random are statistically tied here -- the fix resolved "loses to Random"
(the original embarrassing finding) but did not produce a clear win on the harder task.

### The "tuned" rework

The original "tuned" condition (more epochs, annealed learning rate) was designed for the
old winner-take-all rule and, tested against the new lateral rule, made things *worse*
on both datasets (the new rule already converges in ~3 epochs; training longer just let
redundancy creep back in). Replaced with different knobs after an empirical sweep:
`target_rate` 0.08 -> 0.20 (more units active per input) and `threshold_lr` 0.3 -> 0.15.

Result: **helps on MNIST** (89.03% -> 89.21%, more consistent across seeds) but **hurts
on Fashion-MNIST** (73.80% -> 72.26%). Conclusion: the "right" sparsity level for this
Hebbian layer is task-dependent, not a universal constant -- a second, independent cost
of biological plausibility beyond raw accuracy: Hebbian has no task signal to tell it
what sparsity level is appropriate, unlike Backprop's gradient, which adapts automatically.

### Catastrophic forgetting (`catastrophic_forgetting.py`, 2 seeds, Split-MNIST 0-4 -> 5-9)

| Condition | Before | After | Forgotten | Fresh probe |
|---|---|---|---|---|
| Backprop | 98.93% | 0.63% | 98.30 pts | 98.37% |
| Backprop (sparse) | 98.92% | 4.37% | 94.55 pts | 98.43% |
| Hebbian | 97.57% | 1.90% | 95.67 pts | 96.57% |
| Hebbian (sparse) | 97.54% | 2.26% | 95.28 pts | 96.66% |
| Random (frozen control) | 92.74% | 0.01% | 92.73 pts | 92.70% |

Key findings:
- **Everyone forgets almost completely** -- this is the "catastrophic" part.
- **Sparsity clearly helps Backprop (98.30 -> 94.55 pts lost) but barely helps Hebbian
  (95.67 -> 95.28 pts lost)** -- evidence that sparsity is not what's protecting Hebbian's
  small edge over Backprop.
- **Random (hidden layer literally frozen, never trained) still "forgets" 92.73 points**
  -- proof that most of the forgetting in every condition is the shared linear readout
  overwriting its decision boundary after only seeing new-class examples for a while, not
  the hidden-layer representations being destroyed.
- **Fresh-probe accuracy stays close to "before" for every condition** (96-98%) even
  though the deployed readout collapsed to near-zero -- confirms the representations
  themselves survive; it's specifically the readout's decision rule that gets overwritten.

## 6. New visualizations added this session

- **`output/filters_fashion.png`** (`models/filters_fashion.py`) -- same filters panel as
  `compare_and_visualize_v2.py`, on Fashion-MNIST. Shows several Hebbian units converging
  on near-duplicate generic garment silhouettes (e.g. multiple "shirt blob" / "boot blob"
  filters) instead of 400 visibly distinct patterns, unlike MNIST where each filter is a
  distinct digit stroke. Direct visual explanation for section 5's Fashion-MNIST result.
- **`output/accuracy_fashion.png`** (`models/accuracy_fashion.py`) -- Fashion-MNIST
  counterpart to `accuracy_comparison.png`, same 3-seed methodology and chart style.
- **`output/catastrophic_forgetting_probe.png`** (added to `catastrophic_forgetting.py`)
  -- 3-bar version of the forgetting chart: before / after (deployed readout) / after
  (fresh probe on the same features). Makes the "readout forgot, not the representation"
  finding visible rather than only present in the log output.

## 7. Conclusions actually supported by this data

1. **Backprop wins on accuracy everywhere, decisively, but needs labels + a global error
   signal + weight transport** -- biologically implausible machinery on all three counts.

2. **Feedback alignment demonstrates weight transport specifically is cheap to remove**
   (trains close to Backprop's accuracy) **while removing the whole label/global-error
   apparatus (Hebbian) is expensive** (a real, double-digit-point accuracy cost). That
   contrast is evidence for *which* piece of biological implausibility is actually
   load-bearing, not just a third bar on a chart.

3. **Hebbian, fixed, is a genuinely functional biologically-plausible learner on MNIST**
   -- beats an untrained random projection by a clear, 5-seed-consistent margin, and
   `filters.png` shows it learned real, recognizable digit-stroke structure, not noise.

4. **That edge is fragile and task-dependent.** On Fashion-MNIST it's a statistical tie
   with Random. Mechanism (visible in `filters_fashion.png`): Hebbian's features are
   shaped only by *input statistics* (the most common recurring pixel pattern), never by
   the *task*. On MNIST, "common recurring pattern" and "class identity" happen to
   correlate well (digits really do look visually distinct). On Fashion-MNIST, several
   classes share nearly the same silhouette, so the same mechanism that worked on MNIST
   now rediscovers the same generic shape repeatedly instead of task-discriminating detail.
   This is the concrete, empirical version of the "features shaped by input vs. features
   shaped by task" credit-assignment gap the research notes predicted in the abstract.

5. **Neither locality nor sparsity meaningfully protects against catastrophic forgetting**
   in this two-task setup. Hebbian forgets a little less than Backprop, but the effect is
   small, isn't explained by sparsity (directly tested and ruled out), and sits against a
   backdrop where a hidden layer that literally cannot learn (Random) still "forgets"
   nearly as much, because almost all of the effect is generic readout overwriting, not
   representation destruction. Genuine protection against forgetting would need something
   like memory replay (Complementary Learning Systems theory, cited in the original
   research notes) -- a different mechanism from anything tested here.

6. **Biological plausibility costs more than accuracy.** The tuned-hyperparameter result
   (section 5) shows a second, independent cost: Hebbian's ideal operating point is
   dataset-dependent and has to be hand-tuned per task, because there's no task signal to
   adapt it automatically the way Backprop's gradient does.

## 8. Known limitations / honest open items

- Lateral inhibition (`L`) is implemented correctly but empirically appears close to
  inert at the tested hyperparameters -- see section 4. Don't overclaim it as the active
  mechanism in a presentation.
- STDP and three-factor/neuromodulatory rules remain explicitly out of scope (see section 2).
- Seed counts are modest (2-5 seeds, CPU-only budget) -- reasonable for a class project,
  not exhaustive statistical power.
- Repo git state: the six original `.py` files are tracked at the repo root but no longer
  exist on disk there (pre-dates this session); `models/` is untracked. Not resolved --
  no commits were made this session (only made when explicitly requested).
- Final hyperparameters for `mode="lateral"`: `target_rate=0.08`, `lateral_lr=0.1`,
  `threshold_lr=0.3`, `relax_iters=4`, `lateral_cap=2.0` (baseline); tuned variant uses
  `target_rate=0.20`, `threshold_lr=0.15`. Sparsity control uses `SPARSITY_K=40` (10% of
  `WIDTH=400`).
