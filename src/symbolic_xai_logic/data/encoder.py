"""Encoding utilities for puzzles and solutions."""
from __future__ import annotations
from typing import Any
import numpy as np


def encode_sample(game: Any, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
    """Encode a single puzzle sample."""
    return game.encode(puzzle, encoding=encoding)


def one_hot(value: int, n_classes: int) -> np.ndarray:
    arr = np.zeros(n_classes, dtype=np.float32)
    if 0 <= value < n_classes:
        arr[value] = 1.0
    return arr


def flatten_one_hot(grid: list[list[int]], n_classes: int) -> np.ndarray:
    """Flatten a 2D grid where each cell is one-hot encoded."""
    parts = []
    for row in grid:
        for val in row:
            parts.append(one_hot(val - 1, n_classes) if val > 0 else np.ones(n_classes, dtype=np.float32) / n_classes)
    return np.concatenate(parts)


def normalize(arr: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    rng = vmax - vmin
    if rng == 0:
        return arr - vmin
    return (arr - vmin) / rng
