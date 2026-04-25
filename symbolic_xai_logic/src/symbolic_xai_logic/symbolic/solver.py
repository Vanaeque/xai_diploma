"""Z3/sympy ground-truth solvers."""
from __future__ import annotations
from typing import Any


class SymbolicSolver:
    """Wrapper that delegates to each game's symbolic solver."""

    def __init__(self, game: Any):
        self.game = game

    def solve(self, puzzle: Any) -> Any | None:
        return self.game.solve_symbolic(puzzle)

    def verify(self, puzzle: Any, solution: Any) -> bool:
        """Verify that a solution satisfies all constraints."""
        return self.game.is_valid(solution)

    def solve_batch(self, puzzles: list) -> list:
        """Solve a batch of puzzles, returning None for unsatisfiable."""
        return [self.solve(p) for p in puzzles]

    def symbolic_rules(self) -> list[str]:
        """Return human-readable symbolic rules for the game."""
        from .rules import get_canonical_rules
        return get_canonical_rules(self.game.name)
