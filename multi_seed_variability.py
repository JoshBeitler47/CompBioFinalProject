"""
multi_seed_variability.py
--------------------------
Answers two questions raised while looking at visuals.py:

  1. "Isn't this supposed to be random? Why do I get the exact same numbers
     every time?" -- Because visuals.py pins torch.manual_seed(0), so every
     random draw (weight init, batch order, template init, ...) follows the
     identical sequence on every run. That's deliberate, so the number you
     write in your report is reproducible. This script removes that fixed
     point on purpose: it reruns the SAME A/B/C comparison under several
     different seeds and reports the mean +/- spread, not one number.

  2. "Is there something to tune to make the Hebbian model better?" -- This
     script adds a 4th condition, "Hebbian (tuned)", that only changes two
     knobs versus the baseline Hebbian rule in visuals.py:
       - more training epochs (3 -> 15): competitive learning only updates
         ONE winning unit per example, so with 400 units and just 3 epochs
         many templates barely move past their random initialization.
       - an annealed learning rate (starts at 0.1, decays to 0.01): a
         standard competitive-learning trick (same idea as Kohonen maps) --
         big early moves let templates find the right neighborhood of digit
         space fast, then a shrinking rate lets them settle instead of
         jittering around a moving average forever.
     Everything else (width, normalization, competition rule, readout) is
     identical to visuals.py, so any accuracy difference is attributable to
     just those two knobs.

Outputs:
  seed_variability.png -- one faint dot per seed per condition (the actual
  spread) plus a bold mean +/- 1 std error bar on top.

Runtime: ~1.5-2 min per seed x NUM_SEEDS (see below). Lower NUM_SEEDS or the
epoch counts while iterating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH = 400          # identical hidden width for every condition, every seed
NUM_SEEDS = 5         # how many independent random seeds to sample

# ---------------------------------------------------------------------------
# Data: loaded ONCE, outside the seed loop. Only training randomness (weight
# init, batch order, template init) varies per seed -- not which digits exist.
# ---------------------------------------------------------------------------
train_set = torchvision.datasets.MNIST(root="./data", train=True,  download=True)
test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True)

def prep(ds):
    X = F.normalize(ds.data.float().view(len(ds), -1) / 255.0, dim=1)
    return X.to(device), ds.targets.clone().to(device)

X_train, y_train = prep(train_set)
X_test,  y_test  = prep(test_set)

# ===========================================================================
# The four conditions (same training code as visuals.py, hebbian_W1 gains an
# lr_end argument so it can anneal instead of using one fixed rate).
# ===========================================================================
def train_backprop(epochs=8):
    net = nn.Sequential(nn.Linear(784, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 10)).to(device)
    opt, lossf = torch.optim.Adam(net.parameters(), lr=1e-3), nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(X_train.size(0), device=device)
        for i in range(0, X_train.size(0), 128):
            idx = perm[i:i+128]
            opt.zero_grad()
            lossf(net(X_train[idx]), y_train[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (net(X_test).argmax(1) == y_test).float().mean().item() * 100

def hebbian_W1(epochs=3, lr_start=0.05, lr_end=0.05):
    W = F.normalize(X_train[torch.randint(0, X_train.size(0), (WIDTH,))].clone(), dim=1)
    for e in range(epochs):
        lr = lr_start + (lr_end - lr_start) * (e / max(epochs - 1, 1))
        perm = torch.randperm(X_train.size(0), device=device)
        for i in range(0, X_train.size(0), 256):
            x = X_train[perm[i:i+256]]
            winners = (x @ W.t()).argmax(1)
            sum_x = torch.zeros_like(W)
            counts = torch.zeros(WIDTH, device=device)
            sum_x.index_add_(0, winners, x)
            counts.index_add_(0, winners, torch.ones_like(winners, dtype=torch.float))
            won = counts > 0
            W[won] = F.normalize(W[won] + lr * (sum_x[won] / counts[won].unsqueeze(1) - W[won]), dim=1)
    return W

def random_W1():
    return F.normalize(torch.randn(WIDTH, 784, device=device), dim=1)

def readout_acc(W1, epochs=100):
    Htr, Hte = F.relu(X_train @ W1.t()), F.relu(X_test @ W1.t())
    ro, lossf = nn.Linear(WIDTH, 10).to(device), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(ro.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(ro(Htr), y_train).backward()
        opt.step()
    with torch.no_grad():
        return (ro(Hte).argmax(1) == y_test).float().mean().item() * 100

# ===========================================================================
# Run all four conditions under NUM_SEEDS different seeds.
# ===========================================================================
conditions = ["Backprop\n(end-to-end)", "Hebbian\n+ readout", "Hebbian (tuned)\n+ readout", "Random\n+ readout"]
results = {c: [] for c in conditions}

for seed in range(NUM_SEEDS):
    print(f"\n=== Seed {seed} ({seed+1}/{NUM_SEEDS}) ===")
    torch.manual_seed(seed)

    acc = train_backprop()
    print(f"  Backprop:          {acc:.2f}%")
    results[conditions[0]].append(acc)

    acc = readout_acc(hebbian_W1(epochs=3, lr_start=0.05, lr_end=0.05))
    print(f"  Hebbian (baseline): {acc:.2f}%")
    results[conditions[1]].append(acc)

    acc = readout_acc(hebbian_W1(epochs=15, lr_start=0.1, lr_end=0.01))
    print(f"  Hebbian (tuned):    {acc:.2f}%")
    results[conditions[2]].append(acc)

    acc = readout_acc(random_W1())
    print(f"  Random:             {acc:.2f}%")
    results[conditions[3]].append(acc)

# ===========================================================================
# Summary table
# ===========================================================================
print("\n=== Summary across", NUM_SEEDS, "seeds ===")
stats = {}
for c in conditions:
    t = torch.tensor(results[c])
    stats[c] = (t.mean().item(), t.std().item())
    print(f"  {c.replace(chr(10), ' ')}: {t.mean():.2f}% +/- {t.std():.2f}%  (runs: {[round(v,1) for v in results[c]]})")

# ===========================================================================
# Figure: per-seed dots (the real spread) + mean +/- std error bars
# ===========================================================================
colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, c in enumerate(conditions):
    xs = [i] * NUM_SEEDS
    jitter = torch.linspace(-0.08, 0.08, NUM_SEEDS).tolist()
    ax.scatter([i + j for j in jitter], results[c], color=colors[i], alpha=0.5, zorder=2, label="individual seeds" if i == 0 else None)
    mean, std = stats[c]
    ax.errorbar(i, mean, yerr=std, fmt="D", color=colors[i], markersize=9, capsize=6, zorder=3,
                markeredgecolor="black", markeredgewidth=0.8, label="mean +/- 1 std" if i == 0 else None)

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels([c.replace("\n", " ") for c in conditions])
ax.set_ylabel("Test accuracy (%)")
ax.set_title(f"Same experiment, {NUM_SEEDS} different random seeds -- the spread is the point")
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("seed_variability.png", dpi=150)
plt.close()

print("\nSaved: seed_variability.png")
