"""Dataset generation for all games."""
from __future__ import annotations
from typing import Any
import numpy as np
from ..games.base import Game


def generate_dataset(
    game: Game,
    n_train: int = 2000,
    n_val: int = 400,
    n_test: int = 400,
    n_explain: int = 0,
    difficulty: str = "easy",
    encoding: str = "one_hot",
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Generate train/val/test (and optionally explain) splits.

    Returns dict with keys 'train', 'val', 'test', each containing:
      - 'X': encoded puzzles, shape (n, input_dim)
      - 'y': encoded solutions, shape (n, output_dim)
      - 'puzzles': raw puzzles list
      - 'solutions': raw solutions list

    When ``n_explain > 0``, an additional 'explain' split is produced.  This is
    used by symbolic XAI methods (rule_extraction, symbolic_regression) which
    need many more samples than the test set provides — see P0-3 in
    copilot_upgrade_instructions.md.  The explain split is drawn from the same
    distribution as test (different puzzles, same difficulty) and is disjoint
    from train/val/test.
    """
    total = n_train + n_val + n_test + n_explain
    all_pairs = game.generate(total, difficulty=difficulty, seed=seed)

    X_list, y_list, puzzles, solutions = [], [], [], []
    for puzzle, sol in all_pairs:
        x = game.encode(puzzle, encoding=encoding)
        y = _encode_solution(game, sol, encoding)
        X_list.append(x)
        y_list.append(y)
        puzzles.append(puzzle)
        solutions.append(sol)

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)

    splits = {}
    idx = 0
    split_sizes = [("train", n_train), ("val", n_val), ("test", n_test)]
    if n_explain > 0:
        split_sizes.append(("explain", n_explain))
    for split, size in split_sizes:
        splits[split] = {
            "X": X[idx:idx + size],
            "y": y[idx:idx + size],
            "puzzles": puzzles[idx:idx + size],
            "solutions": solutions[idx:idx + size],
        }
        idx += size
    return splits


def _encode_solution(game: Game, solution: Any, encoding: str) -> np.ndarray:
    """Encode a solution into a target vector.
    
    Note: Solutions are always encoded in "one_hot" format regardless of the input
    encoding. Spatial encoding is only for CNN inputs, not for training targets.
    """
    from ..games.sudoku import SudokuGame
    from ..games.minesweeper import MinesweeperGame

    if isinstance(game, SudokuGame):
        # Always use one_hot for solutions, never spatial
        return game.encode(solution, encoding="one_hot")
    elif isinstance(game, MinesweeperGame):
        # Solution is a 2D mine grid (0/1); flatten to (size*size,)
        return np.array(solution, dtype=np.float32).reshape(-1)
    else:
        return np.array(solution, dtype=np.float32).reshape(-1)
