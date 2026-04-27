"""PyTorch Dataset and DataLoader builders."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LogicDataset(Dataset):
    """Dataset for logic game puzzles."""

    def __init__(self, X: np.ndarray, y: np.ndarray, puzzles: list | None = None, solutions: list | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.puzzles = puzzles or []
        self.solutions = solutions or []

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self) -> int:
        return self.X.shape[1]

    @property
    def output_dim(self) -> int:
        return self.y.shape[1]


def build_dataloaders(
    splits: dict[str, dict[str, np.ndarray]],
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders = {}
    for split, data in splits.items():
        ds = LogicDataset(data["X"], data["y"], data.get("puzzles"), data.get("solutions"))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders
