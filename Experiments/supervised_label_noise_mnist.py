import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


class SimpleMLP(nn.Module):
    """Same MLP as in the small-data experiment."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        return self.net(x)


def add_label_noise(labels: torch.Tensor, noise_rate: float) -> torch.Tensor:
    """Randomly flips a fraction of labels between 0 and 1."""
    labels = labels.clone()
    n = len(labels)
    k = int(noise_rate * n)
    if k == 0:
        return labels

    flip_idx = np.random.choice(n, size=k, replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    return labels


def train_and_eval_with_noise(noise_rate: float) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Convert to binary even/odd
    y_train = (train_ds.targets % 2 == 0).long()
    y_test = (test_ds.targets % 2 == 0).long()

    # Apply label noise to training labels
    y_train_noisy = add_label_noise(y_train, noise_rate)
    train_ds.targets = y_train_noisy
    test_ds.targets = y_test

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # Short training loop
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().squeeze(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


if __name__ == "__main__":
    for noise in [0.0, 0.2, 0.4]:
        acc = train_and_eval_with_noise(noise)
        print(f"Noise {noise:.1f} -> test accuracy: {acc:.4f}")
