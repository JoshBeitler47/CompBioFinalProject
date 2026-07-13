"""
MNIST Hebbian model -- Foldiak-style anti-Hebbian lateral inhibition +
homeostatic feature learning, plus a supervised linear readout.

This is the "biologically-inspired learning" side of your comparison. It is meant to
sit next to mnist_backprop_baseline.py, and the contrast between the two files is the
whole point of the project:

  * Backprop baseline: every weight is updated by loss.backward() -- ONE global error
    signal pushed backward through the whole network.
  * Hebbian model (this file): the hidden layer learns LOCALLY and WITHOUT LABELS.
    Each hidden unit updates using only its own activity, the activity of units it's
    wired to, and the input it sees ("cells that fire together wire together") --
    there is no backward pass, no loss, no labels anywhere in the hidden layer.

Because a purely Hebbian rule never sees the labels, it cannot classify on its own.
So we use the standard bridge: let the Hebbian layer discover features unsupervised,
then train ONE small supervised linear layer (the "readout") on top of those features
with backprop. This is the ONLY place labels or gradients appear in this file.

Expect this to score a bit BELOW the backprop baseline. That gap is the result you're
after: backprop usually wins on accuracy, but its learning rule is far less
biologically plausible than a local Hebbian one.

WHAT'S ACTUALLY HAPPENING (plain-English version): each hidden unit is a little
template. Instead of one unit "winning" a competition outright (a simpler earlier
version of this file did that), several units settle into an activity level together,
suppressing each other through inhibitory connections -- like a group of neurons that
are wired to shush their neighbors when they fire. Two things get learned besides the
templates themselves: how strongly each pair of units inhibits each other (units that
keep firing together learn to inhibit each other MORE, so next time they spread out
instead of piling onto the same pattern), and each unit's own "how excitable am I"
threshold, which drifts up if it's been firing too often and down if it's been too
quiet -- keeping every unit's activity near a small target rate instead of a few units
hogging everything. This is a real named pair of mechanisms: anti-Hebbian lateral
inhibition (Foldiak 1990) and homeostatic synaptic scaling (Turrigiano) -- see
context/project_notes_backprop_vs_hebbian.md for the full citations.

NOTE ON NAMING: none of this is literally Oja's delta rule (Delta w = eta*y*(x - y*w))
from the project notes. That rule, applied to many units at once with no other change,
makes every unit converge to the same top principal component (see project notes
section 4f) -- useless for a multi-unit hidden layer unless you add a deflation step
(Sanger's rule). The lateral inhibition here sidesteps that collapse a different way:
units that start drifting toward the same pattern get pushed apart by their own mutual
inhibition, which is what "Foldiak-style Hebbian learning" means throughout this file.

Needs the same working install as the baseline: torch + torchvision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------------
# 1. Data. Mean-center each image on the TRAINING set's average image, then
#    normalize it to unit length. Centering matters here: without it, every
#    image shares a large "average digit" component, which wastes some of
#    the hidden layer's capacity on a direction that carries no discriminative
#    signal (project notes, section 6). Normalizing matters separately: it
#    makes a unit's response a clean measure of how well its weights MATCH
#    the input pattern (a cosine similarity).
# ---------------------------------------------------------------------------
train_set = torchvision.datasets.MNIST(root="./data", train=True,  download=True)
test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True)

PIXEL_MEAN = (train_set.data.float() / 255.0).view(len(train_set), -1).mean(dim=0)

def to_matrix(dataset):
    X = dataset.data.float().view(len(dataset), -1) / 255.0  # [n, 784], pixels in [0,1]
    X = F.normalize(X - PIXEL_MEAN, dim=1)                   # center, then -> unit length
    y = dataset.targets.clone()                              # [n]
    return X.to(device), y.to(device)

X_train, y_train = to_matrix(train_set)
X_test,  y_test  = to_matrix(test_set)

# ---------------------------------------------------------------------------
# 2. The Hebbian hidden layer (anti-Hebbian lateral inhibition + homeostasis).
#    W1 has one row per hidden unit; each row is a "template" the unit detects.
#    L1 holds inhibitory connections BETWEEN units (L1[i,j] = how hard unit i
#    suppresses unit j). theta holds each unit's own firing threshold. All
#    three are learned with LOCAL, UNSUPERVISED rules -- no labels, no loss,
#    no .backward() anywhere in this section.
#
#    Per input x, in two stages:
#
#    (a) SETTLE: figure out how active each unit gets, given the input AND
#        the other units' inhibition. Start from the raw feedforward match
#        a = W1 @ x, then repeatedly subtract off how much every OTHER unit
#        is currently inhibiting you, a few times, until it stabilizes:
#            y <- relu(a - theta - L1 @ y)
#        Unlike winner-take-all, several units can end up active at once --
#        this is a soft population code, not one single winner.
#
#    (b) LEARN, using only that settled activity y:
#          - W1 (feedforward): each unit nudges its template toward x, SCALED
#            BY ITS OWN ACTIVITY LEVEL -- a unit that barely crossed threshold
#            moves only slightly; a confidently, strongly active unit moves
#            fully. This is the actual "y" term every Hebbian rule has
#            (Delta w ~ y * x). An earlier version of this file dropped that
#            scaling (gave every unit that fired at all the SAME size step,
#            regardless of how strongly) -- that let marginal, low-confidence
#            activations drag many templates toward the same generic pattern,
#            collapsing almost all 400 units down to a handful of effectively
#            distinct templates. Then renormalizes -- the same Oja-style
#            stabilization used before, so weights can't blow up.
#          - L1 (lateral, THE ANTI-HEBBIAN RULE): if two units are BOTH
#            active more often than you'd expect by chance, strengthen the
#            inhibition between them. Fire together -> get pushed apart next
#            time. This is what stops units from collapsing onto the same
#            pattern, replacing an earlier "conscience" bookkeeping hack
#            with the real named mechanism (Foldiak 1990).
#          - theta (homeostatic threshold): nudge each unit's threshold up
#            if it's been firing more than the target rate, down if less --
#            a real biological mechanism (Turrigiano's synaptic scaling) that
#            keeps every unit's activity near a small target instead of a
#            few units hogging everything, and makes the resulting code
#            sparse (few active units per input) as a side effect, not a
#            bolted-on hack. THRESHOLD_LR needs to be fast enough that this
#            catches up within the first few dozen batches -- theta starts at
#            0 (no threshold at all), so too slow a ramp-up lets far too many
#            units fire on every input for a long stretch before any
#            competitive pressure exists to stop them collapsing together.
# ---------------------------------------------------------------------------
HIDDEN = 400          # number of hidden units / templates
HEBB_EPOCHS = 3
HEBB_LR = 0.05
TARGET_RATE = 0.08     # p: target fraction of units active per input
LATERAL_LR = 0.1        # eta_L: how fast inhibitory connections adapt
THRESHOLD_LR = 0.3      # eta_theta: how fast thresholds adapt
RELAX_ITERS = 4         # settling steps per batch
LATERAL_CAP = 2.0       # safety clip so inhibition can't blow up numerically

torch.manual_seed(0)
# Initialize each template from a random training image -> avoids "dead" units
# that never activate and never learn.
init_idx = torch.randint(0, X_train.size(0), (HIDDEN,))
W1 = F.normalize(X_train[init_idx].clone(), dim=1)   # [HIDDEN, 784]
L1 = torch.zeros(HIDDEN, HIDDEN, device=device)      # no inhibition to start
theta = torch.zeros(HIDDEN, device=device)           # no threshold to start
off_diag = ~torch.eye(HIDDEN, dtype=torch.bool, device=device)  # units don't inhibit themselves via L1

print("\nTraining Hebbian hidden layer (unsupervised, no labels)...")
batch = 256
for epoch in range(1, HEBB_EPOCHS + 1):
    perm = torch.randperm(X_train.size(0), device=device)
    for i in range(0, X_train.size(0), batch):
        x = X_train[perm[i:i + batch]]           # [B, 784]
        a = x @ W1.t()                           # [B, HIDDEN] feedforward drive

        # (a) Settle: let lateral inhibition shape the activity.
        y = F.relu(a - theta)
        for _ in range(RELAX_ITERS):
            y = F.relu(a - theta - y @ L1.t())

        b = x.size(0)
        activity_sum = y.sum(dim=0)              # [HIDDEN]
        mean_y = activity_sum / b                # [HIDDEN], each unit's average activity this batch
        has_activity = mean_y > 1e-6

        # (b1) Feedforward Hebbian update: move toward the activity-weighted mean
        # input direction, step size scaled by mean_y (confidence) -- see the
        # comment above for why -- then renormalize (Oja-style stabilization).
        weighted_x = y.t() @ x                   # [HIDDEN, 784]
        target = torch.zeros_like(W1)
        target[has_activity] = weighted_x[has_activity] / activity_sum[has_activity].unsqueeze(1)
        step = HEBB_LR * mean_y.unsqueeze(1) * (target - W1)
        W1[has_activity] = F.normalize(W1[has_activity] + step[has_activity], dim=1)

        # (b2) Anti-Hebbian lateral update: co-activity above chance (TARGET_RATE^2)
        # strengthens mutual inhibition; below chance, it relaxes back down (never negative).
        co_activity = (y.t() @ y) / b            # [HIDDEN, HIDDEN]
        L1 = (L1 + LATERAL_LR * (co_activity - TARGET_RATE**2)).clamp(0.0, LATERAL_CAP)
        L1 = L1 * off_diag

        # (b3) Homeostatic threshold update: pull each unit's firing rate toward TARGET_RATE.
        batch_rate = (y > 0).float().mean(dim=0)  # [HIDDEN]
        theta = (theta + THRESHOLD_LR * (batch_rate - TARGET_RATE)).clamp(min=0.0)

    with torch.no_grad():
        active_frac = (y > 0).float().mean().item()
    print(f"  Hebbian epoch {epoch}/{HEBB_EPOCHS} done "
          f"(mean fraction of units active per input in last batch: {active_frac:.3f}, target {TARGET_RATE})")

# ---------------------------------------------------------------------------
# 3. The supervised readout (the "bridge").
#    Freeze the Hebbian hidden layer (W1, L1, theta) and train ONE linear layer
#    on top with ordinary backprop. This is the ONLY place labels or gradients
#    appear in this file.
# ---------------------------------------------------------------------------
def features(X):
    """Same settling computation as training, minus any learning: run the
    frozen W1/L1/theta forward to get each input's activity."""
    a = X @ W1.t()
    h = F.relu(a - theta)
    for _ in range(RELAX_ITERS):
        h = F.relu(a - theta - h @ L1.t())
    return h                                      # [n, HIDDEN] hidden activations

H_train = features(X_train)                      # precomputed; W1 is frozen from here
H_test  = features(X_test)

readout = nn.Linear(HIDDEN, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(readout.parameters(), lr=0.01)

print("\nTraining supervised linear readout on the frozen Hebbian features...")
READOUT_EPOCHS = 100
for epoch in range(1, READOUT_EPOCHS + 1):
    optimizer.zero_grad()
    loss = criterion(readout(H_train), y_train)
    loss.backward()                              # backprop -- but only for the readout
    optimizer.step()
    if epoch == 1 or epoch % 25 == 0:
        train_acc = (readout(H_train).argmax(1) == y_train).float().mean().item() * 100
        print(f"  readout epoch {epoch}: loss={loss.item():.3f}  train_acc={train_acc:.2f}%")

# ---------------------------------------------------------------------------
# 4. Final test accuracy -- compare this against your backprop baseline.
# ---------------------------------------------------------------------------
with torch.no_grad():
    test_acc = (readout(H_test).argmax(1) == y_test).float().mean().item() * 100

print(f"\nHebbian model test accuracy: {test_acc:.2f}%")
print("Compare this to your backprop baseline (~97%). The gap IS the result:")
print("backprop usually wins on accuracy, but its learning rule is far less")
print("biologically plausible than this local, label-free Hebbian one.")