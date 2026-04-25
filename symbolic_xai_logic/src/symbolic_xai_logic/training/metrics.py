"""Evaluation metrics for logic game solving."""
from __future__ import annotations
import numpy as np
import torch
from typing import Any, Callable


def accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Element-wise binary accuracy."""
    pred_binary = (torch.sigmoid(preds) > threshold).float()
    return (pred_binary == targets).float().mean().item()


def cell_accuracy(preds: torch.Tensor, targets: torch.Tensor, n_classes: int) -> float:
    """
    Per-cell accuracy for one-hot encoded outputs (e.g. Sudoku).
    preds/targets shape: (B, n_cells * n_classes)
    """
    B = preds.shape[0]
    preds_r = preds.view(B, -1, n_classes)
    tgt_r = targets.view(B, -1, n_classes)
    pred_cls = preds_r.argmax(dim=-1)
    tgt_cls = tgt_r.argmax(dim=-1)
    return (pred_cls == tgt_cls).float().mean().item()


def constraint_satisfaction_rate(
    preds: torch.Tensor,
    game: Any,
    n_classes: int | None = None,
) -> float:
    """
    Fraction of predictions that are valid game solutions.
    Decodes predictions and calls game.is_valid().
    """
    from ..games.sudoku import SudokuGame
    from ..games.nqueens import NQueensGame

    valid_count = 0
    B = preds.shape[0]

    for i in range(B):
        pred = preds[i].detach().cpu().numpy()
        if isinstance(game, SudokuGame):
            n = game.size
            decoded = game.decode(pred)
            if game.is_valid(decoded):
                valid_count += 1
        elif isinstance(game, NQueensGame):
            n = game.size
            board = (torch.sigmoid(preds[i]) > 0.5).float().cpu().numpy()
            board = board.reshape(n, n).tolist()
            if game.is_valid(board):
                valid_count += 1
        else:
            pred_binary = (torch.sigmoid(preds[i]) > 0.5).float().cpu().numpy()
            if game.is_valid(pred_binary):
                valid_count += 1

    return valid_count / B if B > 0 else 0.0
