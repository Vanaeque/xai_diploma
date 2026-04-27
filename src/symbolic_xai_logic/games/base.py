from __future__ import annotations
"""Abstract Game interface."""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Game(ABC):
    """Base class for all logic games."""

    @abstractmethod
    def generate(self, n: int, difficulty: str = "easy", seed: int = 42) -> list[tuple[Any, Any]]:
        """Return list of (puzzle, solution) pairs."""

    @abstractmethod
    def is_valid(self, state: Any) -> bool:
        """Return True if state is a valid/consistent solution."""

    @abstractmethod
    def solve_symbolic(self, puzzle: Any) -> Any | None:
        """Return symbolic solution, or None if unsatisfiable."""

    @abstractmethod
    def encode(self, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
        """Encode puzzle into a flat numpy array."""

    @abstractmethod
    def concepts(self, state: Any) -> dict[str, Any]:
        """Return dict of human-meaningful boolean/numeric features."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimensionality of encoded input vector."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of output (solution) vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Game name identifier."""
