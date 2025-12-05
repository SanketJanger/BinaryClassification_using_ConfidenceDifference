import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# You need to replace this with loading your trained ConfDiff models:
# model_abs = ...
# model_relu = ...
# model_unb = ...

def get_binary_mnist(n_samples=3000):
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # even=1, odd=0
    ys = (ds.targets % 2 == 0).long()
    ds.targets = ys
    idx = np.random.choice(len(ds), size=n_samples, replace=False)
    xs = torch.stack([ds[i][0] for i in idx])   # [N, 1, 28, 28]
    ys = ys[idx]
    return xs, ys

def embed_pca(xs):
    xs_flat = xs.view(xs.size(0), -1).numpy()
    pca = PCA(n_components=2)
    z = pca.fit_transform(xs_flat)
    return z, pca

def plot_boundary(z, y, model, pca, title, filename):
    # Build grid in 2D PCA space
    x_min, x_max = z[:,0].min()-1, z[:,0].max()+1
    y_min, y_max = z[:,1].min()-1, z[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]  # [M,2]

    # Inverse transform to original input space
    grid_orig_flat = pca.inverse_transform(grid)
    grid_orig = torch.tensor(grid_orig_flat, dtype=torch.float32).view(-1, 1, 28, 28)

    with torch.no_grad():
        logits = model(grid_orig)  # assumes model: x->[N,1] logit
        probs = torch.sigmoid(logits).cpu().numpy().reshape(xx.shape)

    plt.figure()
    # contour: decision boundary where prob = 0.5
    plt.contourf(xx, yy, probs, levels=[0,0.5,1], alpha=0.3)
    # scatter original points
    plt.scatter(z[:,0], z[:,1], c=y, s=5, cmap='bwr', alpha=0.8)
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    xs, ys = get_binary_mnist()
    z, pca = embed_pca(xs)

if __name__ == "__main__":
    main()
