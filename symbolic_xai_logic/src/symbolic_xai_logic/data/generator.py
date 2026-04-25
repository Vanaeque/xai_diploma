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
    difficulty: str = "easy",
    encoding: str = "one_hot",
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Generate train/val/test splits.
    Returns dict with keys 'train', 'val', 'test', each containing:
      - 'X': encoded puzzles, shape (n, input_dim)
      - 'y': encoded solutions, shape (n, output_dim)
      - 'puzzles': raw puzzles list
      - 'solutions': raw solutions list
    """
    total = n_train + n_val + n_test
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
    for split, size in [("train", n_train), ("val", n_val), ("test", n_test)]:
        splits[split] = {
            "X": X[idx:idx + size],
            "y": y[idx:idx + size],
            "puzzles": puzzles[idx:idx + size],
            "solutions": solutions[idx:idx + size],
        }
        idx += size
    return splits


def _encode_solution(game: Game, solution: Any, encoding: str) -> np.ndarray:
    """Encode a solution into a target vector."""
    from ..games.sudoku import SudokuGame
    from ..games.nqueens import NQueensGame
    from ..games.knights_knaves import KnightsKnavesGame
    from ..games.sat3 import SAT3Game
    from ..games.minesweeper import MinesweeperGame

    if isinstance(game, SudokuGame):
        return game.encode(solution, encoding=encoding)
    elif isinstance(game, NQueensGame):
        return game.encode(solution, encoding=encoding)
    elif isinstance(game, KnightsKnavesGame):
        return np.array(solution, dtype=np.float32)
    elif isinstance(game, SAT3Game):
        return np.array(solution, dtype=np.float32)
    elif isinstance(game, MinesweeperGame):
        # Solution is a 2D mine grid (0/1); flatten to (size*size,)
        return np.array(solution, dtype=np.float32).reshape(-1)
    else:
        return np.array(solution, dtype=np.float32).reshape(-1)
