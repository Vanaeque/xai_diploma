"""Knights and Knaves logic puzzles."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
from .base import Game


class KnightsKnavesGame(Game):
    """
    Knights always tell the truth; Knaves always lie.
    Each puzzle: n agents, each makes a statement about others' types.
    Goal: infer each agent's type (1=knight, 0=knave).
    """

    def __init__(self, n_agents: int = 4, **kwargs):
        self.n_agents = n_agents

    @property
    def name(self) -> str:
        return "knights_knaves"

    @property
    def input_dim(self) -> int:
        # For each agent pair (i, j): statement type (claim i says j is knight/knave)
        return self.n_agents * self.n_agents * 2

    @property
    def output_dim(self) -> int:
        return self.n_agents  # binary type per agent

    def _generate_statements(self, types: list[int], rng: random.Random) -> list[list[tuple[int, int]]]:
        """
        Returns statements[i] = list of (j, claim) where claim=1 means "j is a knight".
        A knight's claim is truthful; a knave's claim is false.
        """
        n = self.n_agents
        statements = [[] for _ in range(n)]
        for i in range(n):
            others = [j for j in range(n) if j != i]
            n_stmts = rng.randint(1, min(3, len(others)))
            targets = rng.sample(others, n_stmts)
            for j in targets:
                truth = types[j]  # actual type of j
                if types[i] == 1:  # knight tells truth
                    statements[i].append((j, truth))
                else:  # knave lies
                    statements[i].append((j, 1 - truth))
        return statements

    def generate(self, n: int, difficulty: str = "easy", seed: int = 42) -> list[tuple[Any, Any]]:
        rng = random.Random(seed)
        pairs = []
        for _ in range(n):
            types = [rng.randint(0, 1) for _ in range(self.n_agents)]
            statements = self._generate_statements(types, rng)
            pairs.append((statements, types))
        return pairs

    def is_valid(self, state: Any) -> bool:
        if isinstance(state, np.ndarray):
            types = state.tolist()
        else:
            types = state
        return all(t in (0, 1) for t in types)

    def solve_symbolic(self, puzzle: Any) -> Any | None:
        try:
            from z3 import Bool, And, Or, Not, Implies, Solver, sat, is_true
        except ImportError:
            return self._solve_brute(puzzle)

        statements = puzzle
        n = self.n_agents
        s = Solver()
        agent_vars = [Bool(f"k_{i}") for i in range(n)]

        for i in range(n):
            for j, claim in statements[i]:
                # If agent i is knight (True), then claim must match j's type
                # If agent i is knave (False), then claim must not match j's type
                claimed_val = agent_vars[j] if claim == 1 else Not(agent_vars[j])
                s.add(Implies(agent_vars[i], claimed_val))
                s.add(Implies(Not(agent_vars[i]), Not(claimed_val)))

        if s.check() == sat:
            m = s.model()
            return [1 if is_true(m[agent_vars[i]]) else 0 for i in range(n)]
        return None

    def _solve_brute(self, puzzle: Any) -> Any | None:
        from itertools import product
        statements = puzzle
        for assignment in product([0, 1], repeat=self.n_agents):
            valid = True
            for i in range(self.n_agents):
                for j, claim in statements[i]:
                    actual = assignment[j]
                    if assignment[i] == 1 and claim != actual:
                        valid = False; break
                    if assignment[i] == 0 and claim == actual:
                        valid = False; break
                if not valid:
                    break
            if valid:
                return list(assignment)
        return None

    def encode(self, puzzle: Any, encoding: str = "one_hot") -> np.ndarray:
        statements = puzzle
        n = self.n_agents
        # Encode as n x n x 2 matrix: [no_statement, says_knight, says_knave]
        arr = np.zeros((n, n, 2), dtype=np.float32)
        for i in range(n):
            for j, claim in statements[i]:
                arr[i, j, claim] = 1.0
        return arr.reshape(-1)

    def concepts(self, state: Any) -> dict[str, Any]:
        if isinstance(state, np.ndarray):
            types = state.tolist()
        else:
            types = list(state)
        result = {}
        for i, t in enumerate(types):
            result[f"agent_{i}_is_knight"] = int(t == 1)
        result["n_knights"] = sum(types)
        result["n_knaves"] = len(types) - sum(types)
        result["majority_knights"] = int(sum(types) > len(types) / 2)
        return result
