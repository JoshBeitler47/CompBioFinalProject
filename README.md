# CompBioFinalProject

Compares backpropagation vs. Hebbian learning on MNIST using a shared `784 → 400 → 10` MLP architecture.

## Motivation

Backpropagation achieves high accuracy but requires a global error signal — biologically implausible. Hebbian learning is local and label-free ("cells that fire together wire together"), making it more biologically plausible, but typically less accurate. The accuracy gap *is* the result.

## Scripts

Scripts have **no `.py` extension** — run them directly:

| Script | What it does |
|---|---|
| `python BACKPROP` | End-to-end backprop MLP (~97% test acc, 5 epochs). Baseline. |
| `python HEBBIAN` | Unsupervised competitive Hebbian hidden layer + supervised linear readout. |
| `python VISUALS` | Trains backprop, Hebbian, and random-hidden-layer control; saves 4 comparison PNGs. |

## Outputs

`VISUALS` produces four figures, already committed:

- `accuracy_comparison.png` — bar chart of test accuracies
- `confusion_matrices.png` — error matrices (diagonal removed)
- `misclassified.png` — example digits each model gets wrong
- `filters.png` — weight patterns of hidden units

## Requirements

- Python 3
- `torch`, `torchvision`, `matplotlib` (install globally; no package file)
- MNIST downloads to `./data/` on first run

## Usage

```sh
python BACKPROP      # baseline accuracy
python HEBBIAN       # comparison accuracy
python VISUALS       # generate all four PNG figures
```

## Architecture note

`VISUALS` includes a random-hidden-layer control to verify the Hebbian layer learned real structure. If Hebbian significantly outperforms random, the local rule discovered meaningful features without labels or backprop.
