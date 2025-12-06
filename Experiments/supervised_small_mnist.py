import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class SimpleMLP(nn.Module):
    """Small MLP for MNIST binary classification (even vs odd)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 1)   # single logit for binary output
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().squeeze(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Basic transform for MNIST
    transform = transforms.Compose([transforms.ToTensor()])

    # Load full train/test splits
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Convert labels to binary: even=1, odd=0
    def make_even_odd(dataset):
        labels = np.array(dataset.targets)
        labels = (labels % 2 == 0).astype(np.int64)
        dataset.targets = torch.from_numpy(labels)
        return dataset

    train_ds = make_even_odd(train_ds)
    test_ds = make_even_odd(test_ds)

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Different sizes for small-data experiment
    subset_sizes = [1000, 3000, 5000]

    for n in subset_sizes:
        print(f"\n=== Supervised training with {n} labeled samples ===")
        # pick random subset of train data
        indices = np.random.choice(len(train_ds), size=n, replace=False)
        small_train = Subset(train_ds, indices)
        train_loader = DataLoader(small_train, batch_size=128, shuffle=True)

        model = SimpleMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train for a few epochs (small dataset)
        for epoch in range(10):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1} / 10 - loss: {loss:.4f}")

        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Test accuracy with {n} labels: {acc:.4f}")


if __name__ == "__main__":
    main()
