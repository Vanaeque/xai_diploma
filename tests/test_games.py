"""Tests: game validators agree with z3 ground truth."""
import pytest
import numpy as np

from symbolic_xai_logic.games.sudoku import SudokuGame


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



