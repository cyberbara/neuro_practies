"""Multi-layer perceptron trained with backprop on Fashion-MNIST (or MNIST).

Built on PyTorch. Trains two hidden-layer MLPs and varies:
  - hidden size
  - activation (ReLU vs Tanh)
  - optimiser (SGD vs Adam)
  - dropout probability

Records train/test accuracy per epoch and produces learning curves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FitResult:
    history: dict          # keys: train_loss, train_acc, test_loss, test_acc (lists)
    params: int            # trainable parameter count
    wall_time: float       # seconds


def load_fashion_mnist(limit_train: int | None = 30_000,
                       limit_test: int | None = 5_000,
                       seed: int = 0):
    """Load Fashion-MNIST. Tries torchvision first, falls back to a small
    sklearn-style synthetic dataset if torchvision isn't installed."""
    try:
        from torchvision import datasets, transforms
        tx = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(root="~/.cache/torchvision",
                                       train=True, download=True, transform=tx)
        test = datasets.FashionMNIST(root="~/.cache/torchvision",
                                      train=False, download=True, transform=tx)
        Xtr = train.data.float() / 255.0
        ytr = train.targets.long()
        Xte = test.data.float() / 255.0
        yte = test.targets.long()
    except Exception:
        # Synthetic fallback: 8x8 digits-like blobs.
        rng = np.random.default_rng(seed)
        from sklearn.datasets import load_digits
        d = load_digits()
        X = d.data / 16.0
        y = d.target.astype(np.int64)
        n = X.shape[0]
        idx = rng.permutation(n)
        split = int(n * 0.7)
        Xtr = torch.tensor(X[idx[:split]], dtype=torch.float32)
        ytr = torch.tensor(y[idx[:split]], dtype=torch.long)
        Xte = torch.tensor(X[idx[split:]], dtype=torch.float32)
        yte = torch.tensor(y[idx[split:]], dtype=torch.long)

    if limit_train:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    # Flatten to (N, 784) for the MLP.
    return (Xtr.view(Xtr.size(0), -1), ytr,
            Xte.view(Xte.size(0), -1), yte)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int = 10,
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        Act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers = [nn.Linear(n_in, n_hidden), Act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(n_hidden, n_hidden), Act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(n_hidden, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit(model: nn.Module, Xtr, ytr, Xte, yte,
        epochs: int = 5, batch_size: int = 128,
        optimiser: str = "adam", lr: float = 1e-3) -> FitResult:
    import time
    if optimiser == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(Xtr, ytr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        tot_loss = 0.0
        tot_correct = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            tot_correct += (out.argmax(1) == yb).sum().item()
        history["train_loss"].append(tot_loss / len(Xtr))
        history["train_acc"].append(tot_correct / len(Xtr))
        model.eval()
        with torch.no_grad():
            out = model(Xte)
            history["test_loss"].append(loss_fn(out, yte).item())
            history["test_acc"].append((out.argmax(1) == yte).float().mean().item())
    wall = time.perf_counter() - t0
    return FitResult(history, count_params(model), wall)
