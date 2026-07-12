"""
MNIST backprop baseline -- a small MLP trained with backpropagation.

This is the "standard neural network" side of your backprop-vs-Hebbian
comparison. Run it directly:

    python mnist_backprop_baseline.py

...or paste the numbered sections into separate notebook cells.

Expected result: ~97-98% test accuracy after 5 epochs. That number is the
baseline you'll compare the Hebbian model against.

Hidden width is 400 to match every other script in this folder (hebbian.py,
catastrophic_forgetting.py, compare_and_visualize_v2.py, ...) -- the whole
comparison only means something if every model gets the same-size hidden
layer.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# 0. Device: use a GPU if one is available, otherwise CPU (CPU is fine here)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

HIDDEN = 400  # shared hidden-layer width across every script in this folder

# ---------------------------------------------------------------------------
# 1. Data: download MNIST and wrap it in DataLoaders that hand out batches.
#    Pixels are mean-centered on the TRAINING set's average image before
#    being handed to the model -- otherwise every input shares a large
#    "average digit" component that isn't useful signal (see
#    context/project_notes_backprop_vs_hebbian.md, section 6). This matters
#    less for backprop than for the Hebbian model, but centering identically
#    everywhere keeps the comparison apples-to-apples.
# ---------------------------------------------------------------------------
train_set_raw = torchvision.datasets.MNIST(root="./data", train=True, download=True)
PIXEL_MEAN = (train_set_raw.data.float() / 255.0).view(len(train_set_raw), -1).mean(dim=0)

class Center:
    def __call__(self, img):
        x = transforms.functional.to_tensor(img).view(-1)
        return (x - PIXEL_MEAN).view(1, 28, 28)

transform = Center()

train_set = torchvision.datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,   shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1000, shuffle=False)

# ---------------------------------------------------------------------------
# 2. Model: a small multi-layer perceptron (MLP)
#    784 inputs (28x28 pixels) -> 400 hidden units -> 10 outputs (digits 0-9)
#    Kept deliberately small and non-convolutional so it's a fair, apples-to-
#    apples match for the Hebbian model later.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # [batch, 1, 28, 28] -> [batch, 784]
        self.fc1 = nn.Linear(784, HIDDEN)    # hidden layer
        self.relu = nn.ReLU()                # non-linearity
        self.fc2 = nn.Linear(HIDDEN, 10)     # output layer: one score per digit

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)                  # raw scores ("logits"); softmax handled by the loss
        return x

model = MLP().to(device)

# ---------------------------------------------------------------------------
# 3. Loss + optimizer
#    CrossEntropyLoss expects raw logits and applies softmax internally.
#    SGD = stochastic gradient descent -- the exact rule from your research thread.
# ---------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ---------------------------------------------------------------------------
# 4. Evaluation helper: accuracy (%) on a given data loader
# ---------------------------------------------------------------------------
def evaluate(loader):
    model.eval()                         # evaluation mode
    correct, total = 0, 0
    with torch.no_grad():                # no gradients needed just to measure
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)   # highest score = predicted digit
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# ---------------------------------------------------------------------------
# 5. Training loop: the four steps, repeated over the data for several epochs
# ---------------------------------------------------------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()                        # training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # (1) forward pass: predictions
        outputs = model(images)
        # (2) compute loss: how wrong are we?
        loss = criterion(outputs, labels)

        # (3) backward pass: autograd fills in every weight's gradient
        optimizer.zero_grad()            # clear leftover gradients from last step
        loss.backward()                  # <-- backpropagation happens right here
        # (4) update: nudge each weight down its gradient
        optimizer.step()

        running_loss += loss.item()

    train_acc = evaluate(train_loader)
    test_acc  = evaluate(test_loader)
    print(f"Epoch {epoch}: loss={running_loss/len(train_loader):.3f}  "
          f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")

print("\nDone. This test accuracy is your backprop baseline "
      "to compare the Hebbian model against.")