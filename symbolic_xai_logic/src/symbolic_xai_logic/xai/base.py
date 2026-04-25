from __future__ import annotations
"""Abstract Explainer interface."""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch.nn as nn


class Explainer(ABC):
    """Base class for all XAI methods."""

    def __init__(self, model: nn.Module, game: Any, **kwargs):
        self.model = model
        self.game = game

    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Compute explanations for a batch of inputs.
        Returns dict with at minimum:
          - 'attributions': np.ndarray of shape (n_samples, input_dim)
          - 'summary': human-readable description
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Explainer name."""

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """
        Measure fidelity of explanation to the model's predictions.
        Default: return 0.0 (subclasses override).
        """
        return 0.0
