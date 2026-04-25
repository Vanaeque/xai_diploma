"""Sudoku game: 4x4 and 9x9 variants with symbolic solver."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
from .base import Game


class SudokuGame(Game):
    """4x4 or 9x9 Sudoku generator, validator, and z3-based solver."""

    def __init__(self, size: int = 4, **kwargs):
        assert size in (4, 9), "Only 4x4 and 9x9 supported"
        self.size = size
        self.box_size = int(size ** 0.5)

    @property
    def name(self) -> str:
        return f"sudoku{self.size}"

    @property
    def input_dim(self) -> int:
        return self.size * self.size * self.size  # one-hot over digits per cell

    @property
    def output_dim(self) -> int:
        return self.size * self.size * self.size

    def _empty_grid(self) -> list[list[int]]:
        return [[0] * self.size for _ in range(self.size)]

    def _is_valid_partial(self, grid: list[list[int]], row: int, col: int, num: int) -> bool:
        if num in grid[row]:
            return False
        if num in [grid[r][col] for r in range(self.size)]:
            return False
        br, bc = (row // self.box_size) * self.box_size, (col // self.box_size) * self.box_size
        for r in range(br, br + self.box_size):
            for c in range(bc, bc + self.box_size):
                if grid[r][c] == num:
                    return False
        return True

    def _solve_backtrack(self, grid: list[list[int]], rng: random.Random | None = None) -> bool:
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] == 0:
                    digits = list(range(1, self.size + 1))
                    if rng:
                        rng.shuffle(digits)
                    for d in digits:
                        if self._is_valid_partial(grid, r, c, d):
                            grid[r][c] = d
                            if self._solve_backtrack(grid, rng):
                                return True
                            grid[r][c] = 0
                    return False
        return True

    def _generate_full_solution(self, rng: random.Random) -> list[list[int]]:
        grid = self._empty_grid()
        self._solve_backtrack(grid, rng)
        return grid

    def _make_puzzle(self, solution: list[list[int]], n_given: int, rng: random.Random) -> list[list[int]]:
        puzzle = [row[:] for row in solution]
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        rng.shuffle(cells)
        remove = self.size * self.size - n_given
        for r, c in cells[:remove]:
            puzzle[r][c] = 0
        return puzzle

    def _n_given(self, difficulty: str) -> int:
        if self.size == 4:
            return {"easy": 12, "medium": 8, "hard": 6}.get(difficulty, 10)
        return {"easy": 45, "medium": 35, "hard": 25}.get(difficulty, 40)

    def generate(self, n: int, difficulty: str = "easy", seed: int = 42) -> list[tuple[Any, Any]]:
        rng = random.Random(seed)
        pairs = []
        for i in range(n):
            sol = self._generate_full_solution(random.Random(rng.randint(0, 2**31)))
            n_given = self._n_given(difficulty)
            puzzle = self._make_puzzle(sol, n_given, random.Random(rng.randint(0, 2**31)))
            pairs.append((puzzle, sol))
        return pairs

    def is_valid(self, state: Any) -> bool:
        if isinstance(state, np.ndarray):
            state = state.reshape(self.size, self.size).tolist()
        for r in range(self.size):
            row_vals = [state[r][c] for c in range(self.size) if state[r][c] != 0]
            if len(row_vals) != len(set(row_vals)):
                return False
        for c in range(self.size):
            col_vals = [state[r][c] for r in range(self.size) if state[r][c] != 0]
            if len(col_vals) != len(set(col_vals)):
                return False
        for br in range(0, self.size, self.box_size):
            for bc in range(0, self.size, self.box_size):
                box_vals = [
                    state[br + dr][bc + dc]
                    for dr in range(self.box_size)
                    for dc in range(self.box_size)
                    if state[br + dr][bc + dc] != 0
                ]
                if len(box_vals) != len(set(box_vals)):
                    return False
        return True

    def solve_symbolic(self, puzzle: Any) -> Any | None:
        try:
            from z3 import Int, And, Or, Distinct, Solver, sat
        except ImportError:
            return self._solve_backtrack_copy(puzzle)

        if isinstance(puzzle, np.ndarray):
            puzzle = puzzle.reshape(self.size, self.size).tolist()

        cells = [[Int(f"x_{r}_{c}") for c in range(self.size)] for r in range(self.size)]
        s = Solver()

        for r in range(self.size):
            for c in range(self.size):
                s.add(cells[r][c] >= 1, cells[r][c] <= self.size)
                if puzzle[r][c] != 0:
                    s.add(cells[r][c] == puzzle[r][c])

        for r in range(self.size):
            s.add(Distinct(cells[r]))
        for c in range(self.size):
            s.add(Distinct([cells[r][c] for r in range(self.size)]))
        for br in range(0, self.size, self.box_size):
            for bc in range(0, self.size, self.box_size):
                box = [cells[br + dr][bc + dc] for dr in range(self.box_size) for dc in range(self.box_size)]
                s.add(Distinct(box))

        if s.check() == sat:
            m = s.model()
            return [[m[cells[r][c]].as_long() for c in range(self.size)] for r in range(self.size)]
        return None

    def _solve_backtrack_copy(self, puzzle: Any) -> Any | None:
        if isinstance(puzzle, np.ndarray):
            grid = puzzle.reshape(self.size, self.size).tolist()
        else:
            grid = [row[:] for row in puzzle]
        success = self._solve_backtrack(grid)
        return grid if success else None

    def encode(self, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
        if isinstance(puzzle, np.ndarray):
            puzzle = puzzle.reshape(self.size, self.size).tolist()
        n = self.size
        if encoding == "one_hot":
            arr = np.zeros((n * n * n,), dtype=np.float32)
            for r in range(n):
                for c in range(n):
                    val = puzzle[r][c]
                    base = (r * n + c) * n
                    if val == 0:
                        arr[base:base + n] = 1.0 / n  # uniform for unknown
                    else:
                        arr[base + val - 1] = 1.0
            return arr
        elif encoding == "flat":
            arr = np.array([puzzle[r][c] for r in range(n) for c in range(n)], dtype=np.float32)
            return arr / n
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    def decode(self, output: np.ndarray) -> list[list[int]]:
        n = self.size
        grid = []
        for r in range(n):
            row = []
            for c in range(n):
                base = (r * n + c) * n
                digit = int(np.argmax(output[base:base + n])) + 1
                row.append(digit)
            grid.append(row)
        return grid

    def concepts(self, state: Any) -> dict[str, Any]:
        if isinstance(state, np.ndarray):
            state = state.reshape(self.size, self.size).tolist()
        n = self.size
        result = {}
        for d in range(1, n + 1):
            for r in range(n):
                result[f"row_{r}_has_{d}"] = int(d in [state[r][c] for c in range(n)])
            for c in range(n):
                result[f"col_{c}_has_{d}"] = int(d in [state[r][c] for r in range(n)])
            for bi, br in enumerate(range(0, n, self.box_size)):
                for bj, bc in enumerate(range(0, n, self.box_size)):
                    box_vals = [state[br + dr][bc + dc] for dr in range(self.box_size) for dc in range(self.box_size)]
                    result[f"box_{bi}_{bj}_has_{d}"] = int(d in box_vals)
        return result
