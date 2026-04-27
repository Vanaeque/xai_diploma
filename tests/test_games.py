"""Tests: game validators agree with z3 ground truth."""
import pytest
import numpy as np

from symbolic_xai_logic.games.sudoku import SudokuGame
from symbolic_xai_logic.games.nqueens import NQueensGame
from symbolic_xai_logic.games.knights_knaves import KnightsKnavesGame
from symbolic_xai_logic.games.sat3 import SAT3Game


class TestSudokuGame:
    def test_generate_returns_pairs(self):
        game = SudokuGame(size=4)
        pairs = game.generate(5, seed=42)
        assert len(pairs) == 5
        for puzzle, sol in pairs:
            assert len(puzzle) == 4
            assert len(sol) == 4

    def test_valid_solution_passes(self):
        game = SudokuGame(size=4)
        valid = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
        assert game.is_valid(valid)

    def test_invalid_row_fails(self):
        game = SudokuGame(size=4)
        invalid = [[1, 1, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
        assert not game.is_valid(invalid)

    def test_solve_symbolic_gives_valid(self):
        game = SudokuGame(size=4)
        pairs = game.generate(3, seed=0)
        for puzzle, _ in pairs:
            sol = game.solve_symbolic(puzzle)
            if sol is not None:
                assert game.is_valid(sol)

    def test_encode_shape(self):
        game = SudokuGame(size=4)
        puzzle = [[1, 0, 3, 0], [0, 4, 0, 2], [0, 0, 4, 0], [0, 0, 0, 0]]
        enc = game.encode(puzzle)
        assert enc.shape == (game.input_dim,)

    def test_concepts_keys(self):
        game = SudokuGame(size=4)
        sol = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
        concepts = game.concepts(sol)
        assert "row_0_has_1" in concepts
        assert "col_0_has_1" in concepts

    def test_generate_solutions_are_valid(self):
        game = SudokuGame(size=4)
        pairs = game.generate(10, seed=7)
        for _, sol in pairs:
            assert game.is_valid(sol), f"Invalid solution: {sol}"


class TestNQueensGame:
    def test_generate(self):
        game = NQueensGame(size=6)
        pairs = game.generate(5, seed=42)
        assert len(pairs) == 5

    def test_valid_solution(self):
        game = NQueensGame(size=4)
        board = [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]
        assert game.is_valid(board)

    def test_invalid_solution_row(self):
        game = NQueensGame(size=4)
        board = [[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]
        assert not game.is_valid(board)

    def test_solve_symbolic(self):
        game = NQueensGame(size=4)
        puzzle = [[0] * 4 for _ in range(4)]
        sol = game.solve_symbolic(puzzle)
        assert sol is not None
        assert game.is_valid(sol)

    def test_encode_shape(self):
        game = NQueensGame(size=6)
        puzzle = [[0] * 6 for _ in range(6)]
        enc = game.encode(puzzle)
        assert enc.shape == (game.input_dim,)


class TestKnightsKnavesGame:
    def test_generate(self):
        game = KnightsKnavesGame(n_agents=3)
        pairs = game.generate(5, seed=42)
        assert len(pairs) == 5

    def test_solve_symbolic(self):
        game = KnightsKnavesGame(n_agents=3)
        pairs = game.generate(5, seed=0)
        for puzzle, expected_types in pairs:
            sol = game.solve_symbolic(puzzle)
            if sol is not None:
                assert len(sol) == 3
                assert all(t in (0, 1) for t in sol)

    def test_encode_shape(self):
        game = KnightsKnavesGame(n_agents=4)
        puzzle = [[(1, 1)], [(0, 0)], [], [(2, 1)]]
        enc = game.encode(puzzle)
        assert enc.shape == (game.input_dim,)

    def test_concepts(self):
        game = KnightsKnavesGame(n_agents=3)
        concepts = game.concepts([1, 0, 1])
        assert concepts["n_knights"] == 2
        assert concepts["n_knaves"] == 1


class TestSAT3Game:
    def test_generate(self):
        game = SAT3Game(n_vars=8, n_clauses=24)
        pairs = game.generate(5, seed=42)
        assert len(pairs) > 0

    def test_solve_symbolic(self):
        game = SAT3Game(n_vars=5, n_clauses=10)
        pairs = game.generate(3, seed=1)
        for clauses, sol in pairs:
            assert game.is_valid(sol)
            # Verify the solution actually satisfies the formula
            assert game._evaluate(clauses, sol)

    def test_encode_shape(self):
        game = SAT3Game(n_vars=10, n_clauses=40)
        clauses = [(1, -2, 3), (-4, 5, -6)]
        enc = game.encode(clauses)
        assert enc.shape == (game.input_dim,)
