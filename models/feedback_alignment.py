"""
feedback_alignment.py
----------------------
MNIST feedback alignment (Lillicrap et al. 2016) -- the third model
recommended in context/project_notes_backprop_vs_hebbian.md as "the *code*"
version of the predictive-coding framing: a small, one-line-of-math change
to backprop that directly attacks the WEIGHT TRANSPORT PROBLEM, which the
notes single out as the sharper, more accurate version of "backprop needs
information a real synapse can't have" (section 2).

WHAT WEIGHT TRANSPORT IS: for a 784 -> HIDDEN -> 10 network, standard
backprop's hidden-layer gradient is

    dL/dh = dL/dout @ W2.T

i.e. it reuses W2 -- the SAME matrix the forward pass just used to go
hidden -> output -- transposed, to send error back output -> hidden. For a
physical synapse on the W1 side, computing its own gradient this way would
require knowing the exact current strength of every W2 synapse downstream of
it, synapses it has no physical contact with. No known biological mechanism
does that; it's the concrete, nameable version of "backprop isn't local."

THE FIX: replace W2.T in that one computation with B, a matrix of FIXED
random numbers, generated once at initialization and never updated by the
optimizer -- it is never tied to W2 in any way, so there is no transport
problem left for that step. Remarkably (Lillicrap et al. 2016), the network
still trains to a usable accuracy: W2 gradually rotates DURING training to
become roughly aligned with B, so the "wrong" feedback direction becomes an
increasingly good approximation of the true gradient direction over time.
This script prints that alignment angle every epoch so you can watch it
happen.

HONEST LIMITS: feedback alignment removes weight transport, but a global
scalar loss gradient is still flowing backward through the network end to
end -- it does NOT remove the "no global error signal" objection on its own,
only the specific "and that signal would need to know downstream weights"
piece of it. Frame it that way in the writeup (see project notes section 2).

Expect accuracy close to, but very slightly below, the plain backprop
baseline (backprop.py) -- W2 has to spend part of training rotating toward B
before the backward signal it carries becomes a good gradient approximation,
which ordinary backprop never has to do (its backward matrix is exactly
correct from step one).

Run directly:  python feedback_alignment.py
Needs the same working install as the rest of this folder: torch,
torchvision, matplotlib (used by common.py's shared data loader).
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
import common as C

device = C.device
print("Using device:", device)

HIDDEN = 400  # shared hidden-layer width across every script in this folder
EPOCHS = 5
LR = 0.1
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Data: same mean-centered, unit-normalized MNIST as every other script here
# (see common.py for why centering matters).
# ---------------------------------------------------------------------------
X_train, y_train, X_test, y_test, _raw_test, _mean = C.load_dataset(torchvision.datasets.MNIST)


# ---------------------------------------------------------------------------
# The feedback-alignment linear layer. Forward pass is identical to
# nn.Linear. The backward pass is where it departs from ordinary backprop:
# the gradient flowing back into this layer's INPUT is computed using B (a
# fixed random matrix) instead of this layer's own weight matrix transposed.
# Gradients w.r.t. this layer's own W and b are still the ordinary local
# outer-product terms -- a real synapse can compute those from its own
# pre/post activity and whatever error signal reaches it; it just no longer
# needs to know W to help COMPUTE that error signal for the layer upstream.
# ---------------------------------------------------------------------------
class FeedbackAlignmentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, B):
        ctx.save_for_backward(x, W, B)
        return x @ W + b

    @staticmethod
    def backward(ctx, grad_output):
        x, W, B = ctx.saved_tensors
        grad_x = grad_output @ B.t()          # <-- fixed random B, NOT W.t(): the whole idea
        grad_W = x.t() @ grad_output
        grad_b = grad_output.sum(dim=0)
        return grad_x, grad_W, grad_b, None   # None: B itself is never updated


class FALinear(nn.Module):
    """A drop-in nn.Linear replacement whose backward pass uses feedback
    alignment. B is a registered buffer (not a Parameter), so the optimizer
    never touches it and it stays fixed for the network's whole lifetime."""

    def __init__(self, in_features, out_features):
        super().__init__()
        bound = 1.0 / in_features**0.5
        self.W = nn.Parameter(torch.empty(in_features, out_features).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("B", torch.empty(in_features, out_features).uniform_(-bound, bound))

    def forward(self, x):
        return FeedbackAlignmentFunction.apply(x, self.W, self.b, self.B)


# ---------------------------------------------------------------------------
# Model: 784 -> HIDDEN -> 10, same shape as backprop.py. Only fc2 needs to be
# an FALinear: fc2's backward pass is the one computation whose grad_x output
# becomes fc1's incoming error signal, so that's the only place weight
# transport happens in a 2-layer network. fc1 stays an ordinary nn.Linear --
# nothing needs to backprop further upstream than the input pixels.
# ---------------------------------------------------------------------------
class FAMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.relu = nn.ReLU()
        self.fc2 = FALinear(HIDDEN, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = FAMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


@torch.no_grad()
def evaluate(X, y):
    model.eval()
    preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item() * 100


@torch.no_grad()
def alignment_angle_degrees():
    """How far W2 has rotated toward B, in degrees. 90 degrees = orthogonal
    (feedback carries no useful direction yet, effectively random). Under
    30-40 degrees is the range where Lillicrap et al. report FA starting to
    train comparably to backprop. Watching this number fall over training is
    the single clearest way to see feedback alignment "work.\""""
    W, B = model.fc2.W.flatten(), model.fc2.B.flatten()
    cos = (W @ B) / (W.norm() * B.norm() + 1e-8)
    cos = cos.clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).item()


# ---------------------------------------------------------------------------
# Training loop -- identical shape to backprop.py's, so the only difference
# in the numbers this produces is the backward-pass substitution above.
# ---------------------------------------------------------------------------
print(f"\nInitial W2-vs-B angle: {alignment_angle_degrees():.1f} degrees (starts near 90 = random)\n")

n = X_train.size(0)
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n, device=device)
    running_loss = 0.0
    n_batches = 0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        images, labels = X_train[idx], y_train[idx]

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()          # <-- uses FeedbackAlignmentFunction.backward for fc2
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    train_acc = evaluate(X_train, y_train)
    test_acc = evaluate(X_test, y_test)
    angle = alignment_angle_degrees()
    print(
        f"Epoch {epoch}: loss={running_loss / n_batches:.3f}  "
        f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%  "
        f"W2-vs-B angle={angle:.1f} deg"
    )

print(
    "\nDone. Compare this test accuracy to backprop.py's baseline. The gap (if any) is the"
)
print(
    "cost of removing weight transport: fc2's backward pass never looked at fc2's own"
)
print(
    "weights to compute fc1's error signal, only a fixed random matrix -- and the falling"
)
print("W2-vs-B angle above is training quietly correcting for that on its own.")
