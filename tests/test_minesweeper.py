"""Tests for the Minesweeper game module."""
import pytest
import numpy as np

from symbolic_xai_logic.games.minesweeper import MinesweeperGame, UNKNOWN, MINE
from symbolic_xai_logic.data.generator import generate_dataset


class TestMinesweeperGenerate:
    def test_returns_n_pairs(self):
        game = MinesweeperGame(size=8, n_mines=10)
        pairs = game.generate(10, seed=42)
        assert len(pairs) == 10

    def test_solution_has_correct_mine_count(self):
        game = MinesweeperGame(size=8, n_mines=10)
        for puzzle, mines in game.generate(20, seed=0):
            total = sum(mines[r][c] for r in range(8) for c in range(8))
            assert total == 10, f"Expected 10 mines, got {total}"

    def test_solution_is_binary(self):
        game = MinesweeperGame(size=6, n_mines=5)
        for _, mines in game.generate(10, seed=1):
            for row in mines:
                assert all(v in (0, 1) for v in row)

    def test_puzzle_revealed_cells_are_correct_counts(self):
        game = MinesweeperGame(size=8, n_mines=10)
        for puzzle, mines in game.generate(10, seed=3):
            n = game.size
            for r in range(n):
                for c in range(n):
                    val = puzzle[r][c]
                    if val != UNKNOWN:
                        assert mines[r][c] == 0, "Revealed cell should not be a mine"
                        adj = game._count_adjacent(mines, r, c)
                        assert val == adj, f"Cell ({r},{c}) clue={val} but adj mines={adj}"

    def test_difficulty_affects_reveal_rate(self):
        game = MinesweeperGame(size=8, n_mines=10)
        n_unknown_easy = sum(
            1 for p, _ in game.generate(20, difficulty="easy", seed=7)
            for row in p for v in row if v == UNKNOWN
        )
        n_unknown_hard = sum(
            1 for p, _ in game.generate(20, difficulty="hard", seed=7)
            for row in p for v in row if v == UNKNOWN
        )
        assert n_unknown_hard > n_unknown_easy


class TestMinesweeperIsValid:
    def test_valid_solution(self):
        game = MinesweeperGame(size=4, n_mines=3)
        mines = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        assert game.is_valid(mines)

    def test_wrong_mine_count_fails(self):
        game = MinesweeperGame(size=4, n_mines=3)
        mines = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        assert not game.is_valid(mines)

    def test_generated_solutions_are_valid(self):
        game = MinesweeperGame(size=8, n_mines=10)
        for _, mines in game.generate(10, seed=9):
            assert game.is_valid(mines)

    def test_numpy_array_input(self):
        game = MinesweeperGame(size=4, n_mines=2)
        arr = np.zeros((4, 4), dtype=int)
        arr[0, 0] = 1
        arr[1, 1] = 1
        assert game.is_valid(arr)


class TestMinesweeperSolveSymbolic:
    def test_solve_returns_valid_solution(self):
        game = MinesweeperGame(size=6, n_mines=5)
        for puzzle, true_mines in game.generate(5, seed=0):
            sol = game.solve_symbolic(puzzle)
            if sol is not None:
                assert game.is_valid(sol)
                assert game.is_consistent_with_puzzle(puzzle, sol)

    def test_easy_puzzle_solvable(self):
        game = MinesweeperGame(size=4, n_mines=2)
        puzzle, mines = game.generate(1, difficulty="easy", seed=42)[0]
        sol = game.solve_symbolic(puzzle)
        assert sol is not None
        assert game.is_valid(sol)


class TestMinesweeperEncode:
    def test_one_hot_shape(self):
        game = MinesweeperGame(size=8, n_mines=10)
        puzzle, _ = game.generate(1, seed=0)[0]
        enc = game.encode(puzzle, encoding="one_hot")
        assert enc.shape == (game.input_dim,)

    def test_flat_shape(self):
        game = MinesweeperGame(size=8, n_mines=10)
        puzzle, _ = game.generate(1, seed=0)[0]
        enc = game.encode(puzzle, encoding="flat")
        assert enc.shape == (game.size * game.size,)

    def test_one_hot_unknown_cell(self):
        game = MinesweeperGame(size=4, n_mines=2)
        puzzle = [[UNKNOWN] * 4 for _ in range(4)]
        enc = game.encode(puzzle, encoding="one_hot")
        # Every first index of each 11-slot group should be 1.0 (unknown)
        k = game._n_states
        for i in range(game.size * game.size):
            assert enc[i * k] == 1.0

    def test_one_hot_revealed_cell(self):
        game = MinesweeperGame(size=4, n_mines=2)
        puzzle = [[UNKNOWN] * 4 for _ in range(4)]
        puzzle[0][0] = 3  # clue = 3
        enc = game.encode(puzzle, encoding="one_hot")
        k = game._n_states  # 11: CH_HIDDEN=0, CH_FLAGGED=1, CH_NUMBER_BASE=2..10
        # Cell (0,0) base = 0*k = 0
        # Hidden channel (0) → 0 because cell is revealed
        assert enc[0] == 0.0
        # Clue-3 channel: base + CH_NUMBER_BASE + 3 = 0 + 2 + 3 = 5
        from symbolic_xai_logic.games.minesweeper import CH_NUMBER_BASE
        assert enc[0 + CH_NUMBER_BASE + 3] == 1.0

    def test_decode_roundtrip(self):
        game = MinesweeperGame(size=4, n_mines=2)
        _, mines = game.generate(1, seed=5)[0]
        flat = np.array(mines, dtype=np.float32).reshape(-1)
        decoded = game.decode(flat)
        assert decoded == mines

    def test_unknown_encoding_raises(self):
        game = MinesweeperGame(size=4, n_mines=2)
        puzzle = [[UNKNOWN] * 4 for _ in range(4)]
        with pytest.raises(ValueError):
            game.encode(puzzle, encoding="invalid_encoding")


class TestMinesweeperConcepts:
    def test_concepts_keys_present(self):
        game = MinesweeperGame(size=8, n_mines=10)
        _, mines = game.generate(1, seed=0)[0]
        c = game.concepts(mines)
        assert "total_mines" in c
        assert "correct_mine_count" in c
        assert "mine_density" in c
        assert "row_0_mine_count" in c
        assert "col_0_has_mine" in c

    def test_total_mines_concept(self):
        game = MinesweeperGame(size=8, n_mines=10)
        for _, mines in game.generate(5, seed=42):
            c = game.concepts(mines)
            assert c["total_mines"] == 10
            assert c["correct_mine_count"] == 1

    def test_mine_density(self):
        game = MinesweeperGame(size=8, n_mines=10)
        _, mines = game.generate(1, seed=0)[0]
        c = game.concepts(mines)
        assert abs(c["mine_density"] - 10 / 64) < 1e-6


class TestMinesweeperDataset:
    def test_generate_dataset(self):
        game = MinesweeperGame(size=8, n_mines=10)
        splits = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=42)
        assert splits["train"]["X"].shape == (20, game.input_dim)
        assert splits["train"]["y"].shape == (20, game.output_dim)

    def test_output_is_binary(self):
        game = MinesweeperGame(size=8, n_mines=10)
        splits = generate_dataset(game, n_train=10, n_val=2, n_test=2, seed=7)
        y = splits["train"]["y"]
        assert np.all((y == 0) | (y == 1))

    def test_output_mine_count(self):
        game = MinesweeperGame(size=8, n_mines=10)
        splits = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=3)
        y = splits["train"]["y"]
        for row in y:
            assert int(row.sum()) == game.n_mines


class TestMinesweeperRegistered:
    def test_get_game(self):
        from symbolic_xai_logic.games import get_game
        game = get_game("minesweeper", size=8, n_mines=10)
        assert isinstance(game, MinesweeperGame)
        assert game.name == "minesweeper8"
