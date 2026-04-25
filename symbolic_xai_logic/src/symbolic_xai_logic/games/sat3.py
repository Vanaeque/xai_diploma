"""Random 3-SAT instances."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
from .base import Game


class SAT3Game(Game):
    """
    Random 3-SAT: n_vars boolean variables, n_clauses 3-literal clauses.
    Goal: find a satisfying assignment.
    """

    def __init__(self, n_vars: int = 10, n_clauses: int = 40, **kwargs):
        self.n_vars = n_vars
        self.n_clauses = n_clauses

    @property
    def name(self) -> str:
        return "sat3"

    @property
    def input_dim(self) -> int:
        return self.n_vars + self.n_clauses * 3 * 2  # vars + clauses encoded as (var, sign)

    @property
    def output_dim(self) -> int:
        return self.n_vars  # binary assignment

    def _generate_formula(self, rng: random.Random) -> list[tuple[int, int, int]]:
        """Return list of (lit0, lit1, lit2) where lit > 0 is positive, lit < 0 is negated."""
        clauses = []
        for _ in range(self.n_clauses):
            vars_ = rng.sample(range(1, self.n_vars + 1), 3)
            clause = tuple(v if rng.random() > 0.5 else -v for v in vars_)
            clauses.append(clause)
        return clauses

    def _evaluate(self, clauses: list, assignment: list[int]) -> bool:
        for clause in clauses:
            satisfied = any(
                (assignment[abs(lit) - 1] == 1 if lit > 0 else assignment[abs(lit) - 1] == 0)
                for lit in clause
            )
            if not satisfied:
                return False
        return True

    def generate(self, n: int, difficulty: str = "easy", seed: int = 42) -> list[tuple[Any, Any]]:
        rng = random.Random(seed)
        pairs = []
        attempts = 0
        while len(pairs) < n and attempts < n * 50:
            attempts += 1
            clauses = self._generate_formula(rng)
            solution = self.solve_symbolic(clauses)
            if solution is not None:
                pairs.append((clauses, solution))
        return pairs

    def is_valid(self, state: Any) -> bool:
        if isinstance(state, (list, np.ndarray)):
            return all(v in (0, 1) for v in state)
        return False

    def solve_symbolic(self, puzzle: Any) -> Any | None:
        try:
            from z3 import Bool, Or, Not, Solver, sat, is_true
        except ImportError:
            return self._solve_dpll(puzzle)

        clauses = puzzle
        s = Solver()
        vars_ = [Bool(f"x_{i}") for i in range(self.n_vars)]

        for clause in clauses:
            lits = []
            for lit in clause:
                idx = abs(lit) - 1
                lits.append(vars_[idx] if lit > 0 else Not(vars_[idx]))
            s.add(Or(*lits))

        if s.check() == sat:
            m = s.model()
            return [1 if is_true(m[vars_[i]]) else 0 for i in range(self.n_vars)]
        return None

    def _solve_dpll(self, clauses: list) -> list[int] | None:
        """Simple DPLL-like solver as fallback."""
        assignment = [-1] * self.n_vars

        def bt(idx: int) -> bool:
            if idx == self.n_vars:
                return self._evaluate(clauses, assignment)
            for val in (0, 1):
                assignment[idx] = val
                if bt(idx + 1):
                    return True
            assignment[idx] = -1
            return False

        if bt(0):
            return [max(0, v) for v in assignment]
        return None

    def encode(self, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
        clauses = puzzle
        # First part: variable existence indicators (all ones, n_vars vars)
        var_part = np.ones(self.n_vars, dtype=np.float32)
        # Second part: clause encoding
        clause_part = np.zeros(self.n_clauses * 3 * 2, dtype=np.float32)
        for i, clause in enumerate(clauses[:self.n_clauses]):
            for j, lit in enumerate(clause[:3]):
                var_idx = abs(lit) - 1
                sign = 0 if lit > 0 else 1
                clause_part[(i * 3 + j) * 2 + 0] = var_idx / self.n_vars
                clause_part[(i * 3 + j) * 2 + 1] = float(sign)
        return np.concatenate([var_part, clause_part])

    def concepts(self, state: Any) -> dict[str, Any]:
        if isinstance(state, np.ndarray):
            assignment = state.tolist()
        else:
            assignment = list(state)
        result = {}
        for i, v in enumerate(assignment):
            result[f"var_{i}_true"] = int(v == 1)
        result["n_true"] = sum(assignment)
        result["n_false"] = len(assignment) - sum(assignment)
        result["majority_true"] = int(sum(assignment) > len(assignment) / 2)
        return result
