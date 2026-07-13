# CompBioFinalProject

Compares backpropagation vs. Hebbian learning on MNIST using a shared `784 → 400 → 10` MLP architecture.

## Motivation

Backpropagation achieves high accuracy but requires a global error signal — biologically implausible. Hebbian learning is local and label-free ("cells that fire together wire together"), making it more biologically plausible, but typically less accurate. The accuracy gap *is* the result.

## Scripts

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

- `accuracy_comparison.png` — bar chart of test accuracies
- `confusion_matrices.png` — error matrices (diagonal removed)
- `misclassified.png` — example digits each model gets wrong
- `filters.png` — weight patterns of hidden units
- `sample_efficiency.png` — accuracy vs. fraction of labeled data
- `training_curves.png` — accuracy vs. epoch
- `seed_variability.png` — per-seed scatter across 5 seeds
- `catastrophic_forgetting.png` — before/after task interference
- `fashion_seed_variability.png` — same on Fashion-MNIST

## Requirements

- Python 3
- `torch`, `torchvision`, `matplotlib` (install globally; no package file)
- MNIST downloads to `./data/` on first run

## Usage

```sh
python src/backprop.py                    # baseline accuracy
python src/hebbian.py                     # comparison accuracy
python src/compare_and_visualize_v2.py    # generate comparison PNGs
```

## Architecture note

All models share a `784 -> 400 -> 10` architecture. The compare script includes a random-hidden-layer control to verify the Hebbian layer learned real structure.
