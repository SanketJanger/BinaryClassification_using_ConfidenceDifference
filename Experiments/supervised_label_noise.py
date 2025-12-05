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
            nn.Linear(300, 1)
        )
    def forward(self, x):
        return self.net(x)

def apply_label_noise(targets, noise_rate):
    targets = targets.clone()
    n = len(targets)
    k = int(noise_rate * n)
    idx = np.random.choice(n, size=k, replace=False)
    targets[idx] = 1 - targets[idx] 
    return targets

def train_and_eval(noise_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Binarize
    y_train = (train_ds.targets % 2 == 0).long()
    y_test  = (test_ds.targets % 2 == 0).long()

    # Apply label noise to train labels
    y_train_noisy = apply_label_noise(y_train, noise_rate)
    train_ds.targets = y_train_noisy
    test_ds.targets  = y_test

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
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
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

if __name__ == "__main__":
    for noise in [0.0, 0.2, 0.4]:
        acc = train_and_eval(noise)
        print(f"Noise {noise:.1f} -> supervised test accuracy: {acc:.4f}")
