"""Fashion-MNIST (official train + test merged, 70k images) with a seeded 70/30 split."""

from pathlib import Path

import torch
from torchvision import datasets

DATA_DIR = Path(__file__).parent / "data"

# Mean/std of the Fashion-MNIST pixels (in [0, 1] scale).
# Standardizing inputs to ~zero mean / unit variance makes optimization easier.
MEAN, STD = 0.2860, 0.3530


def _load_all() -> tuple[torch.Tensor, torch.Tensor]:
    train = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    # .data is uint8 (N, 28, 28): concatenate, scale to [0, 1], add channel dim.
    x = torch.cat([train.data, test.data]).float().div_(255.0).unsqueeze(1)
    y = torch.cat([train.targets, test.targets])
    return (x - MEAN) / STD, y


def load_split(seed: int, device: torch.device, test_fraction: float = 0.3):
    """Return (x_train, y_train, x_test, y_test) as tensors already on `device`."""
    x, y = _load_all()
    # A dedicated generator: the split depends only on `seed`,
    # not on whatever else consumed random numbers before.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    n_test = int(len(x) * test_fraction)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return (
        x[train_idx].to(device), y[train_idx].to(device),
        x[test_idx].to(device), y[test_idx].to(device),
    )
