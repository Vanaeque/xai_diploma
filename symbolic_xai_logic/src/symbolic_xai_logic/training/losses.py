"""Loss functions for logic game solving."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SudokuLoss(nn.Module):
    """BCE loss over one-hot cell distributions for Sudoku."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred, target)


class BinaryLoss(nn.Module):
    """BCE loss for binary outputs (N-Queens, Knights-Knaves, SAT3)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred, target)


def get_loss(game_name: str) -> nn.Module:
    if "sudoku" in game_name or "nqueens" in game_name:
        return SudokuLoss()
    return BinaryLoss()
