# CompBioFinalProject

Compares backpropagation vs. biologically-inspired learning on MNIST / Fashion-MNIST
using a shared `784 → 400 → 10` architecture across every model.

## Motivation

Backpropagation achieves high accuracy but requires a global error signal routed
through weights a real synapse has no physical access to (the "weight transport
problem") — biologically implausible. Hebbian learning is local and label-free
("cells that fire together wire together"), making it more biologically plausible,
but typically less accurate, and it's not obvious it protects against catastrophic
forgetting either. Feedback alignment sits in between: it removes weight transport
specifically while keeping a global loss signal. The accuracy/plausibility trade-offs
*are* the result. See `context/project_notes_backprop_vs_hebbian.md` for the full
research write-up behind these design choices.

## Scripts

<<<<<<< HEAD
All Python source is in `src/`:

| Script | What it does |
|---|---|
| `python src/backprop.py` | End-to-end backprop MLP (~97% test acc). Baseline. |
| `python src/hebbian.py` | Unsupervised competitive Hebbian hidden layer + supervised linear readout. |
| `python src/compare_and_visualize_v2.py` | Multi-model, multi-seed comparison with sample efficiency & training curves. |
| `python src/catastrophic_forgetting.py` | Tests forgetting: train on digits 0-4, then 5-9. |
| `python src/multi_seed_variability.py` | Same experiment across 5 seeds with scatter + error bars. |
| `python src/fashion_mnist_variability.py` | Same comparison on Fashion-MNIST. |

## Outputs

Generated PNGs in `output/`:
=======
All scripts live in `models/` and are run from the **repository root** (so their
relative `./data` and `output/` paths resolve correctly):

| Script | What it does |
|---|---|
| `python models/backprop.py` | End-to-end backprop MLP, `784→400→10` (~97-98% test acc, 5 epochs). Baseline. |
| `python models/hebbian.py` | Unsupervised competitive Hebbian hidden layer (with a conscience mechanism to prevent template collapse) + supervised linear readout. |
| `python models/feedback_alignment.py` | Backprop variant that removes weight transport: the hidden layer's error signal is computed with a fixed random matrix instead of the output layer's own weights. Prints the W-vs-B alignment angle each epoch. |
| `python models/compare_and_visualize_v2.py` | Trains Backprop, Hebbian, and a random-hidden-layer control; multi-seed accuracy, sample-efficiency sweep, training curves, confusion matrices, misclassified digits, and learned filters. |
| `python models/catastrophic_forgetting.py` | Split-MNIST (train on digits 0-4, then 5-9, re-test on 0-4) for Backprop, Hebbian, and Random, each in dense and sparse (top-k masked) variants, to disentangle whether forgetting comes from the *learning rule* or from representation *sparsity*. |
| `python models/multi_seed_variability.py` | Backprop / Hebbian / Hebbian-tuned / Random across 5 seeds on MNIST, to check how much a single-run number can be trusted. |
| `python models/fashion_mnist_variability.py` | Same 4-condition sweep as above, on Fashion-MNIST, to see if a harder task separates Hebbian from a random projection. |
| `python models/filters_fashion.py` | Same "what does each hidden unit look for" visualization as `compare_and_visualize_v2.py`'s filters panel, but on Fashion-MNIST -- shows Hebbian units converging on shared generic garment silhouettes instead of distinct shapes, a visual explanation for why its edge over Random shrinks on the harder dataset. |
| `python models/accuracy_fashion.py` | Same 3-condition (Backprop / Hebbian / Random) multi-seed bar chart as `compare_and_visualize_v2.py`'s accuracy_comparison.png, but on Fashion-MNIST. |

`models/common.py` is a shared library (data loading + centering, the competitive
Hebbian rule, backprop/readout training, the seed-variability experiment runner) --
not meant to be run directly. Every script above imports from it, so a fix made
once (e.g. to the Hebbian rule) applies everywhere.

## Outputs

Scripts save PNGs to `output/` (already committed from a prior run):
>>>>>>> joshbranch

- `accuracy_comparison.png` — bar chart of test accuracies, mean ± std over seeds
- `sample_efficiency.png` — accuracy vs. fraction of labeled training data used
- `training_curves.png` — accuracy vs. epoch, backprop vs. Hebbian readout
- `confusion_matrices.png` — error matrices (diagonal removed)
- `misclassified.png` — example digits each model gets wrong
- `filters.png` — weight patterns of hidden units
<<<<<<< HEAD
- `sample_efficiency.png` — accuracy vs. fraction of labeled data
- `training_curves.png` — accuracy vs. epoch
- `seed_variability.png` — per-seed scatter across 5 seeds
- `catastrophic_forgetting.png` — before/after task interference
- `fashion_seed_variability.png` — same on Fashion-MNIST
=======
- `catastrophic_forgetting.png` — Task-A accuracy before/after learning Task B, dense vs. sparse
- `catastrophic_forgetting_probe.png` — same experiment, but splits "after" into what the deployed readout can still do vs. what a FRESH readout can recover from the same (drifted) features -- shows most of the apparent forgetting is the readout's decision boundary being overwritten, not the underlying representation being destroyed
- `seed_variability.png` / `fashion_seed_variability.png` — per-seed spread for the 4-condition sweep
- `filters_fashion.png` — the filters visualization repeated on Fashion-MNIST
- `accuracy_fashion.png` — the accuracy bar chart repeated on Fashion-MNIST

`feedback_alignment.py` prints its results (accuracy + alignment angle per epoch)
rather than saving a figure.
>>>>>>> joshbranch

## Requirements

- Python 3
- `torch`, `torchvision`, `matplotlib`, `numpy` (install globally; no package file)
- MNIST / Fashion-MNIST download to `./data/` on first run

## Usage

Run from the repository root:

```sh
<<<<<<< HEAD
python src/backprop.py                    # baseline accuracy
python src/hebbian.py                     # comparison accuracy
python src/compare_and_visualize_v2.py    # generate comparison PNGs
=======
python models/backprop.py               # baseline accuracy
python models/hebbian.py                # comparison accuracy
python models/feedback_alignment.py     # weight-transport-free third model
python models/compare_and_visualize_v2.py   # generate the main comparison figures
python models/catastrophic_forgetting.py    # Split-MNIST forgetting experiment
python models/multi_seed_variability.py     # seed spread + Hebbian tuning check (MNIST)
python models/fashion_mnist_variability.py  # same, on Fashion-MNIST
python models/filters_fashion.py            # filters visualization on Fashion-MNIST
python models/accuracy_fashion.py           # accuracy bar chart on Fashion-MNIST
>>>>>>> joshbranch
```

The multi-seed / sweep scripts are the slowest (minutes per seed on CPU) --
lower `NUM_SEEDS` / epoch counts near the top of each file while iterating.

## Architecture note

<<<<<<< HEAD
All models share a `784 -> 400 -> 10` architecture. The compare script includes a random-hidden-layer control to verify the Hebbian layer learned real structure.
=======
`compare_and_visualize_v2.py` includes a random-hidden-layer control to verify the
Hebbian layer learned real structure. If Hebbian significantly outperforms random,
the local rule discovered meaningful features without labels or backprop. On plain
MNIST this gap has historically been small (see `fashion_mnist_variability.py`'s
docstring) -- `common.hebbian_W1`'s conscience mechanism and the mean-centering in
`common.load_dataset` both target that gap directly.
>>>>>>> joshbranch
