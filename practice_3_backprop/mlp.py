"""Multi-layer perceptron trained with backprop.

Built on PyTorch. Trains two hidden-layer MLPs and varies:
  - hidden size
  - activation (ReLU vs Tanh)
  - optimiser (SGD vs Adam)
  - dropout probability

Records train/test accuracy per epoch and produces learning curves.

The data is the shared letters dataset (see `common/letters.py`): 26
uppercase letters, 28x28, train fonts differ from test fonts. The network
classifies the letter; the headline task of the repo — "find the letter X" —
is answered by the softmax probability of the X neuron.
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


def load_letters_tensors(split, device: str = "cpu"):
    """Turn a LetterSplit into flattened float32 tensors."""
    Xtr = torch.tensor(split.X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(split.y_train, dtype=torch.long, device=device)
    Xte = torch.tensor(split.X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(split.y_test, dtype=torch.long, device=device)
    return Xtr, ytr, Xte, yte


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
