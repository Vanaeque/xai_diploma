"""Minesweeper game: deduce mine positions from partial cell reveals."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
from .base import Game

# Sentinel values for the puzzle grid
UNKNOWN = -1   # unrevealed cell
MINE = 9       # confirmed mine marker (unused in standard puzzles, kept for API compat)

# Spatial encoding channel indices (C, H, W order)
CH_HIDDEN = 0        # 1 if cell is unrevealed
CH_FLAGGED = 1       # reserved for future flagged channel (always 0 in this impl)
CH_NUMBER_BASE = 2   # channels 2–10 carry one-hot for revealed count 0–8
N_SPATIAL_CHANNELS = 11  # hidden + flagged + 9 number channels


class MinesweeperGame(Game):
    """
    Minesweeper on a size×size board with n_mines mines.

    Puzzle format: 2D list where each cell is:
      UNKNOWN (-1) → unrevealed
      0–8          → revealed safe cell (adjacent mine count)

    Solution format: 2D list of 0/1 where 1 = mine.

    Encodings
    ---------
    one_hot  : flat vector (size²×11), channel-last, one-hot per cell state.
    spatial  : flat vector (11×size²) in channel-first order so a CNN can
               reshape it to (B, 11, size, size) without any reordering.
    flat     : normalised values ∈ [−1/9, 8/9], useful for quick baselines.

    Generation
    ----------
    Post-hoc filter: boards where no revealed number uniquely identifies a
    provably-safe or provably-mine cell are discarded ("has_deducible_cell"
    filter).  This gives cleaner supervised targets consistent with the
    canonical Minesweeper rule the XAI is expected to recover.
    """

    _REVEAL_RATE = {"easy": 0.60, "medium": 0.35, "hard": 0.15}
    # Standard mine densities for each board size
    _DEFAULT_MINES = {8: 10, 16: 40, 9: 10}

    def __init__(self, size: int = 8, n_mines: int | None = None, **kwargs):
        assert size >= 4, "Minimum board size is 4"
        if n_mines is None:
            n_mines = self._DEFAULT_MINES.get(size, max(1, int(size * size * 0.16)))
        assert n_mines < size * size, "Too many mines for board size"
        self.size = size
        self.n_mines = n_mines
        self._n_states = N_SPATIAL_CHANNELS

    @property
    def name(self) -> str:
        return f"minesweeper{self.size}"

    @property
    def input_dim(self) -> int:
        # Default (one_hot): channel-last, 11 values per cell
        return self.size * self.size * self._n_states

    @property
    def spatial_input_dim(self) -> int:
        # Spatial CNN encoding: same total, channel-first
        return self._n_states * self.size * self.size

    @property
    def output_dim(self) -> int:
        return self.size * self.size

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _neighbours(self, r: int, c: int) -> list[tuple[int, int]]:
        n = self.size
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0) and 0 <= r + dr < n and 0 <= c + dc < n
        ]

    def _count_adjacent(self, mines: list[list[int]], r: int, c: int) -> int:
        return sum(mines[nr][nc] for nr, nc in self._neighbours(r, c))

    def _compute_numbers(self, mines: list[list[int]]) -> list[list[int]]:
        n = self.size
        return [
            [self._count_adjacent(mines, r, c) if mines[r][c] == 0 else MINE
             for c in range(n)]
            for r in range(n)
        ]

    def _place_mines(self, rng: random.Random) -> list[list[int]]:
        n = self.size
        flat = [0] * (n * n)
        for pos in rng.sample(range(n * n), self.n_mines):
            flat[pos] = 1
        return [flat[r * n:(r + 1) * n] for r in range(n)]

    def _make_puzzle(
        self,
        mines: list[list[int]],
        numbers: list[list[int]],
        reveal_rate: float,
        rng: random.Random,
    ) -> list[list[int]]:
        n = self.size
        safe_cells = [(r, c) for r in range(n) for c in range(n) if mines[r][c] == 0]
        n_reveal = max(1, int(len(safe_cells) * reveal_rate))
        revealed = set(map(tuple, rng.sample(safe_cells, min(n_reveal, len(safe_cells)))))
        puzzle = [[UNKNOWN] * n for _ in range(n)]
        for r, c in revealed:
            puzzle[r][c] = numbers[r][c]
        return puzzle

    def _has_deducible_cell(self, puzzle: list[list[int]], mines: list[list[int]]) -> bool:
        """
        Return True if at least one revealed-number cell makes a neighbour
        provably safe or provably a mine.

        Canonical local rule:
          - If number(r,c) == count(unknown neighbours) → all unknowns are mines.
          - If number(r,c) == 0 → all neighbours are safe.
        These are the exact conditions the XAI should recover.
        """
        n = self.size
        for r in range(n):
            for c in range(n):
                clue = puzzle[r][c]
                if clue == UNKNOWN:
                    continue
                if clue == 0:
                    return True   # all neighbours provably safe
                unknown_nbrs = [
                    (nr, nc) for nr, nc in self._neighbours(r, c)
                    if puzzle[nr][nc] == UNKNOWN
                ]
                if len(unknown_nbrs) == clue:
                    return True   # all unknown neighbours provably mines
        return False

    # ------------------------------------------------------------------
    # Game interface
    # ------------------------------------------------------------------

    def generate(self, n: int, difficulty: str = "easy", seed: int = 42) -> list[tuple[Any, Any]]:
        rng = random.Random(seed)
        reveal_rate = self._REVEAL_RATE.get(difficulty, 0.40)
        pairs: list[tuple[Any, Any]] = []
        attempts = 0
        max_attempts = n * 20

        while len(pairs) < n and attempts < max_attempts:
            attempts += 1
            sub_rng = random.Random(rng.randint(0, 2 ** 31))
            mines = self._place_mines(sub_rng)
            numbers = self._compute_numbers(mines)
            puzzle = self._make_puzzle(mines, numbers, reveal_rate, sub_rng)
            # Post-hoc filter: keep only boards where the canonical local rule applies
            if self._has_deducible_cell(puzzle, mines):
                pairs.append((puzzle, mines))

        # If filter was too strict, fall back without filtering
        while len(pairs) < n:
            sub_rng = random.Random(rng.randint(0, 2 ** 31))
            mines = self._place_mines(sub_rng)
            numbers = self._compute_numbers(mines)
            puzzle = self._make_puzzle(mines, numbers, reveal_rate, sub_rng)
            pairs.append((puzzle, mines))

        return pairs

    def is_valid(self, state: Any) -> bool:
        if isinstance(state, np.ndarray):
            grid = state.reshape(self.size, self.size).tolist()
        else:
            grid = state
        n = self.size
        mine_count = sum(grid[r][c] for r in range(n) for c in range(n))
        return mine_count == self.n_mines

    def is_consistent_with_puzzle(self, puzzle: Any, mine_grid: Any) -> bool:
        n = self.size
        if isinstance(mine_grid, np.ndarray):
            mg = mine_grid.reshape(n, n).tolist()
        else:
            mg = mine_grid
        if isinstance(puzzle, np.ndarray):
            pz = puzzle.reshape(n, n).tolist()
        else:
            pz = puzzle
        for r in range(n):
            for c in range(n):
                clue = pz[r][c]
                if clue != UNKNOWN and clue != MINE and isinstance(clue, int) and 0 <= clue <= 8:
                    if self._count_adjacent(mg, r, c) != clue:
                        return False
        return True

    def solve_symbolic(self, puzzle: Any) -> Any | None:
        try:
            from z3 import Bool, Not, Sum, If, Solver, sat, is_true
        except ImportError:
            return self._solve_constraint_prop(puzzle)

        if isinstance(puzzle, np.ndarray):
            puzzle = puzzle.reshape(self.size, self.size).tolist()

        n = self.size
        cells = [[Bool(f"m_{r}_{c}") for c in range(n)] for r in range(n)]
        s = Solver()

        all_cells = [cells[r][c] for r in range(n) for c in range(n)]
        s.add(Sum([If(v, 1, 0) for v in all_cells]) == self.n_mines)

        for r in range(n):
            for c in range(n):
                clue = puzzle[r][c]
                if clue == UNKNOWN:
                    continue
                s.add(Not(cells[r][c]))
                if isinstance(clue, int) and 0 <= clue <= 8:
                    nbrs = [cells[nr][nc] for nr, nc in self._neighbours(r, c)]
                    s.add(Sum([If(v, 1, 0) for v in nbrs]) == clue)

        if s.check() == sat:
            m = s.model()
            return [[1 if is_true(m[cells[r][c]]) else 0 for c in range(n)] for r in range(n)]
        return None

    def _solve_constraint_prop(self, puzzle: Any) -> Any | None:
        if isinstance(puzzle, np.ndarray):
            grid = puzzle.reshape(self.size, self.size).tolist()
        else:
            grid = [row[:] for row in puzzle]

        n = self.size
        mines = [[-1] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if grid[r][c] != UNKNOWN:
                    mines[r][c] = 0

        unknown = [(r, c) for r in range(n) for c in range(n) if mines[r][c] == -1]

        def bt(idx: int, placed: int) -> bool:
            if idx == len(unknown):
                return placed == self.n_mines and self._check_clues(grid, mines)
            r, c = unknown[idx]
            remaining = len(unknown) - idx
            for val in (1, 0):
                if val == 1 and placed + 1 + (remaining - 1) < self.n_mines:
                    continue
                if val == 0 and placed + remaining < self.n_mines:
                    continue
                mines[r][c] = val
                if bt(idx + 1, placed + val):
                    return True
            mines[r][c] = -1
            return False

        if bt(0, 0):
            return [[max(0, mines[r][c]) for c in range(n)] for r in range(n)]
        return None

    def _check_clues(self, grid: list, mines: list) -> bool:
        n = self.size
        for r in range(n):
            for c in range(n):
                clue = grid[r][c]
                if clue != UNKNOWN and isinstance(clue, int) and 0 <= clue <= 8:
                    if self._count_adjacent(mines, r, c) != clue:
                        return False
        return True

    def encode(self, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
        """
        Encode the puzzle as a flat array.

        one_hot  : (size² × 11,) channel-last one-hot – compatible with MLP.
        spatial  : (11 × size²,) channel-first – reshape to (11, H, W) for CNN.
        flat     : (size²,) normalised values.
        """
        if isinstance(puzzle, np.ndarray):
            grid = puzzle.reshape(self.size, self.size).tolist()
        else:
            grid = puzzle
        n = self.size
        k = self._n_states   # 11

        if encoding == "one_hot":
            arr = np.zeros(n * n * k, dtype=np.float32)
            for r in range(n):
                for c in range(n):
                    base = (r * n + c) * k
                    val = grid[r][c]
                    if val == UNKNOWN:
                        arr[base + CH_HIDDEN] = 1.0
                    else:
                        arr[base + CH_NUMBER_BASE + int(val)] = 1.0
            return arr

        elif encoding == "spatial":
            # Channel-first: shape (k, n, n) → flatten as (k*n*n,)
            arr = np.zeros((k, n, n), dtype=np.float32)
            for r in range(n):
                for c in range(n):
                    val = grid[r][c]
                    if val == UNKNOWN:
                        arr[CH_HIDDEN, r, c] = 1.0
                    else:
                        arr[CH_NUMBER_BASE + int(val), r, c] = 1.0
            return arr.reshape(-1)

        elif encoding == "flat":
            arr = np.array(
                [grid[r][c] / 9.0 for r in range(n) for c in range(n)],
                dtype=np.float32,
            )
            return arr

        else:
            raise ValueError(f"Unknown encoding: {encoding!r}. Choose one_hot | spatial | flat.")

    def decode(self, output: np.ndarray) -> list[list[int]]:
        n = self.size
        flat = (output.reshape(-1) > 0.5).astype(int).tolist()
        return [flat[r * n:(r + 1) * n] for r in range(n)]

    def concepts(self, state: Any) -> dict[str, Any]:
        """
        Concepts for probing.  `state` is the mine-placement solution (0/1 grid).

        Includes the canonical local-rule concepts:
          - provably_safe_exists  : ∃ revealed-0 cell (all neighbours safe)
          - all_unknowns_mine_exists : ∃ cell where #unknown_nbrs == clue
            (these are the two cases the XAI is expected to recover)
        """
        if isinstance(state, np.ndarray):
            mines = state.reshape(self.size, self.size).tolist()
        else:
            mines = state
        n = self.size
        numbers = self._compute_numbers(mines)
        result: dict[str, Any] = {}

        # Per-row / per-column mine presence
        for r in range(n):
            row_mines = sum(mines[r])
            result[f"row_{r}_mine_count"] = row_mines
            result[f"row_{r}_has_mine"] = int(row_mines > 0)
        for c in range(n):
            col_mines = sum(mines[r][c] for r in range(n))
            result[f"col_{c}_mine_count"] = col_mines
            result[f"col_{c}_has_mine"] = int(col_mines > 0)

        # Global counts
        total = sum(mines[r][c] for r in range(n) for c in range(n))
        result["total_mines"] = total
        result["correct_mine_count"] = int(total == self.n_mines)
        result["mine_density"] = total / (n * n)

        # Corner mines
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        result["mines_in_corners"] = sum(mines[r][c] for r, c in corners)
        result["mine_in_top_half"] = int(
            sum(mines[r][c] for r in range(n // 2) for c in range(n)) > 0
        )
        result["mine_in_bottom_half"] = int(
            sum(mines[r][c] for r in range(n // 2, n) for c in range(n)) > 0
        )

        # Canonical local-rule concepts (used to evaluate XAI recovery)
        has_zero_clue = any(
            numbers[r][c] == 0 and mines[r][c] == 0
            for r in range(n) for c in range(n)
        )
        result["has_zero_clue"] = int(has_zero_clue)  # provably_safe_exists proxy

        # Cell where all unknown neighbours equal the clue → all are mines
        saturated = any(
            numbers[r][c] == mines[r][c] == 0 or (
                mines[r][c] == 0
                and isinstance(numbers[r][c], int)
                and numbers[r][c] > 0
                and sum(1 for nr, nc in self._neighbours(r, c) if mines[nr][nc] == 1)
                   == numbers[r][c]
            )
            for r in range(n) for c in range(n)
        )
        result["has_saturated_clue"] = int(saturated)

        # Max clue on the board (proxy for board complexity)
        clues = [numbers[r][c] for r in range(n) for c in range(n)
                 if mines[r][c] == 0 and isinstance(numbers[r][c], int)]
        result["max_clue"] = int(max(clues)) if clues else 0
        result["mean_clue"] = float(np.mean(clues)) if clues else 0.0

        return result
