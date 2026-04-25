"""N-Queens game: place N queens on NxN board with no conflicts."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
from .base import Game


class NQueensGame(Game):
    """N-Queens: find a placement where no two queens attack each other."""

    def __init__(self, size: int = 6, **kwargs):
        self.size = size

    @property
    def name(self) -> str:
        return f"nqueens{self.size}"

    @property
    def input_dim(self) -> int:
        return self.size * self.size

    @property
    def output_dim(self) -> int:
        return self.size * self.size

    def _all_solutions(self) -> list[list[int]]:
        """Return all solutions as list of column-per-row assignments."""
        solutions = []

        def bt(queens: list[int]) -> None:
            row = len(queens)
            if row == self.size:
                solutions.append(queens[:])
                return
            for col in range(self.size):
                if all(
                    queens[r] != col
                    and abs(queens[r] - col) != row - r
                    for r in range(row)
                ):
                    queens.append(col)
                    bt(queens)
                    queens.pop()

        bt([])
        return solutions

    def generate(self, n: int, difficulty: str = "easy", seed: int = 42) -> list[tuple[Any, Any]]:
        rng = random.Random(seed)
        all_sols = self._all_solutions()
        if not all_sols:
            return []
        pairs = []
        for _ in range(n):
            sol_cols = rng.choice(all_sols)
            board = [[0] * self.size for _ in range(self.size)]
            for r, c in enumerate(sol_cols):
                board[r][c] = 1

            # Puzzle: show partial board with some queens revealed
            n_revealed = max(1, self.size // 3) if difficulty == "easy" else max(1, self.size // 4)
            rows_revealed = rng.sample(range(self.size), min(n_revealed, self.size))
            puzzle = [[0] * self.size for _ in range(self.size)]
            for r in rows_revealed:
                puzzle[r][sol_cols[r]] = 1

            pairs.append((puzzle, board))
        return pairs

    def is_valid(self, state: Any) -> bool:
        if isinstance(state, np.ndarray):
            board = state.reshape(self.size, self.size).tolist()
        else:
            board = state
        queens = [(r, c) for r in range(self.size) for c in range(self.size) if board[r][c] == 1]
        if len(queens) != self.size:
            return False
        row_counts = [sum(board[r]) for r in range(self.size)]
        col_counts = [sum(board[r][c] for r in range(self.size)) for c in range(self.size)]
        if any(rc != 1 for rc in row_counts) or any(cc != 1 for cc in col_counts):
            return False
        for i, (r1, c1) in enumerate(queens):
            for r2, c2 in queens[i + 1:]:
                if abs(r1 - r2) == abs(c1 - c2):
                    return False
        return True

    def solve_symbolic(self, puzzle: Any) -> Any | None:
        try:
            from z3 import Int, And, Or, Distinct, Solver, sat, If, Sum
        except ImportError:
            return self._solve_backtrack(puzzle)

        if isinstance(puzzle, np.ndarray):
            puzzle = puzzle.reshape(self.size, self.size).tolist()

        queens = [Int(f"q_{r}") for r in range(self.size)]
        s = Solver()
        for q in queens:
            s.add(q >= 0, q < self.size)
        s.add(Distinct(queens))
        for i in range(self.size):
            for j in range(i + 1, self.size):
                s.add(queens[i] - queens[j] != i - j)
                s.add(queens[i] - queens[j] != j - i)

        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] == 1:
                    s.add(queens[r] == c)

        if s.check() == sat:
            m = s.model()
            board = [[0] * self.size for _ in range(self.size)]
            for r in range(self.size):
                c = m[queens[r]].as_long()
                board[r][c] = 1
            return board
        return None

    def _solve_backtrack(self, puzzle: Any) -> Any | None:
        if isinstance(puzzle, np.ndarray):
            puzzle = puzzle.reshape(self.size, self.size).tolist()
        fixed = {}
        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] == 1:
                    fixed[r] = c
        queens: list[int] = []

        def bt(row: int) -> bool:
            if row == self.size:
                return True
            cols = [fixed[row]] if row in fixed else range(self.size)
            for col in cols:
                if all(
                    queens[r] != col and abs(queens[r] - col) != row - r
                    for r in range(row)
                ):
                    queens.append(col)
                    if bt(row + 1):
                        return True
                    queens.pop()
            return False

        if bt(0):
            board = [[0] * self.size for _ in range(self.size)]
            for r, c in enumerate(queens):
                board[r][c] = 1
            return board
        return None

    def encode(self, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
        if isinstance(puzzle, np.ndarray):
            return puzzle.reshape(-1).astype(np.float32)
        return np.array([puzzle[r][c] for r in range(self.size) for c in range(self.size)], dtype=np.float32)

    def concepts(self, state: Any) -> dict[str, Any]:
        if isinstance(state, np.ndarray):
            board = state.reshape(self.size, self.size).tolist()
        else:
            board = state
        result = {}
        for r in range(self.size):
            result[f"row_{r}_occupied"] = int(any(board[r][c] for c in range(self.size)))
        for c in range(self.size):
            result[f"col_{c}_occupied"] = int(any(board[r][c] for r in range(self.size)))
        queens = [(r, c) for r in range(self.size) for c in range(self.size) if board[r][c] == 1]
        result["n_queens"] = len(queens)
        result["no_row_conflict"] = int(len(set(r for r, _ in queens)) == len(queens))
        result["no_col_conflict"] = int(len(set(c for _, c in queens)) == len(queens))
        diag_ok = all(
            abs(queens[i][0] - queens[j][0]) != abs(queens[i][1] - queens[j][1])
            for i in range(len(queens))
            for j in range(i + 1, len(queens))
        )
        result["no_diag_conflict"] = int(diag_ok)
        return result
