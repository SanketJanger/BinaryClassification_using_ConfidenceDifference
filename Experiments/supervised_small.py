import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 300),
            nn.ReLU(),
            nn.Linear(300, 1)  # binary output (logit)
        )

    def forward(self, x):
        return self.net(x)

# eval helpers
def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Main experiment
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Load full train/test
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Convert to binary: even=1, odd=0
    def binarize(dataset):
        ys = np.array(dataset.targets)
        ys = (ys % 2 == 0).astype(np.int64)
        dataset.targets = torch.from_numpy(ys)
        return dataset

    train_ds = binarize(train_ds)
    test_ds  = binarize(test_ds)

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Different small training sizes
    subset_sizes = [1000, 3000, 5000]

    for n in subset_sizes:
        print(f"\n=== Training supervised MLP on {n} samples ===")
        idx = np.random.choice(len(train_ds), size=n, replace=False)
        small_train = Subset(train_ds, idx)
        train_loader = DataLoader(small_train, batch_size=128, shuffle=True)

        model = MLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train for a few epochs (small dataset)
        for epoch in range(10):
            loss = train_epoch(model, train_loader, optimizer, device)
        acc = eval_accuracy(model, test_loader, device)
        print(f"Supervised test accuracy with n={n}: {acc:.4f}")

if __name__ == "__main__":
    main()
