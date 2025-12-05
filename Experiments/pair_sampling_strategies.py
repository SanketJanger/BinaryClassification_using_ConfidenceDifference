import torch
import numpy as np
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, features, strategy="random", n_pairs=15000):
        """
        features: tensor [N, D]
        strategy: "random" or "diverse"
        """
        self.features = features
        self.n_pairs = n_pairs
        self.strategy = strategy

        self.pairs_idx = self._make_pairs()

    def _make_pairs(self):
        N = self.features.size(0)
        if self.strategy == "random":
            i1 = np.random.randint(0, N, size=self.n_pairs)
            i2 = np.random.randint(0, N, size=self.n_pairs)
            return np.stack([i1, i2], axis=1)

        elif self.strategy == "diverse":
            # sample a random anchor, then choose farthest point for each
            idx_pairs = []
            idx_all = np.arange(N)
            feats = self.features.numpy()
            for _ in range(self.n_pairs):
                i1 = np.random.choice(N)
                # compute distances to all points
                dists = np.linalg.norm(feats - feats[i1], axis=1)
                # choose from top 10% farthest points
                cutoff = int(0.9 * N)
                far_idx_candidates = np.argsort(dists)[cutoff:]
                i2 = np.random.choice(far_idx_candidates)
                idx_pairs.append((i1, i2))
            return np.array(idx_pairs)
        else:
            raise ValueError("Unknown strategy")

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        i1, i2 = self.pairs_idx[idx]
        return self.features[i1], self.features[i2]
