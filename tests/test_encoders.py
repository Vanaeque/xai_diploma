"""Tests for data encoding."""
import numpy as np
import pytest

from symbolic_xai_logic.games.sudoku import SudokuGame
from symbolic_xai_logic.data.encoder import one_hot, flatten_one_hot, normalize
from symbolic_xai_logic.data.generator import generate_dataset
from symbolic_xai_logic.data.dataset import LogicDataset, build_dataloaders


class TestOneHot:
    def test_basic(self):
        arr = one_hot(2, 4)
        assert arr.shape == (4,)
        assert arr[2] == 1.0
        assert arr.sum() == 1.0

    def test_out_of_range(self):
        arr = one_hot(5, 4)
        assert arr.sum() == 0.0

    def test_flatten_one_hot(self):
        grid = [[1, 2], [3, 4]]
        arr = flatten_one_hot(grid, n_classes=4)
        assert arr.shape == (2 * 2 * 4,)


class TestNormalize:
    def test_normalize(self):
        arr = np.array([0.0, 5.0, 10.0])
        normed = normalize(arr, 0.0, 10.0)
        assert abs(normed[0]) < 1e-6
        assert abs(normed[-1] - 1.0) < 1e-6


class TestDatasetGeneration:
    def test_sudoku_dataset(self):
        game = SudokuGame(size=4)
        splits = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=42)
        assert "train" in splits and "val" in splits and "test" in splits
        assert splits["train"]["X"].shape == (20, game.input_dim)
        assert splits["train"]["y"].shape == (20, game.output_dim)

    def test_loaders(self):
        game = SudokuGame(size=4)
        splits = generate_dataset(game, n_train=32, n_val=8, n_test=8, seed=0)
        loaders = build_dataloaders(splits, batch_size=8)
        assert "train" in loaders
        for X, y in loaders["train"]:
            assert X.shape[1] == game.input_dim
            assert y.shape[1] == game.output_dim
            break

    def test_logic_dataset(self):
        X = np.random.rand(10, 16).astype(np.float32)
        y = np.random.rand(10, 16).astype(np.float32)
        ds = LogicDataset(X, y)
        assert len(ds) == 10
        xi, yi = ds[0]
        assert xi.shape == (16,)
