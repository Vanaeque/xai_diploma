"""Tests: same seed → same metrics."""
import pytest
import numpy as np
import torch

from symbolic_xai_logic.utils.seeding import set_global_seed
from symbolic_xai_logic.games.sudoku import SudokuGame
from symbolic_xai_logic.models.mlp import MLP
from symbolic_xai_logic.data.generator import generate_dataset


class TestReproducibility:
    def test_same_seed_same_data(self):
        game = SudokuGame(size=4)
        splits_a = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=42)
        splits_b = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=42)
        np.testing.assert_array_equal(splits_a["train"]["X"], splits_b["train"]["X"])

    def test_different_seed_different_data(self):
        game = SudokuGame(size=4)
        splits_a = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=42)
        splits_b = generate_dataset(game, n_train=20, n_val=5, n_test=5, seed=99)
        assert not np.array_equal(splits_a["train"]["X"], splits_b["train"]["X"])

    def test_same_seed_same_model_init(self):
        set_global_seed(42)
        model_a = MLP(input_dim=64, output_dim=16, hidden_dims=[32])
        w_a = model_a.net[0].weight.detach().clone()

        set_global_seed(42)
        model_b = MLP(input_dim=64, output_dim=16, hidden_dims=[32])
        w_b = model_b.net[0].weight.detach().clone()

        torch.testing.assert_close(w_a, w_b)

    def test_same_seed_same_forward(self):
        set_global_seed(42)
        model = MLP(input_dim=64, output_dim=16, hidden_dims=[32])
        model.eval()  # disable dropout for deterministic output
        x = torch.randn(4, 64)

        out_a = model(x).detach()
        out_b = model(x).detach()
        torch.testing.assert_close(out_a, out_b)

    def test_global_seed_utility(self):
        set_global_seed(0)
        a = np.random.randint(0, 1000, 10)
        set_global_seed(0)
        b = np.random.randint(0, 1000, 10)
        np.testing.assert_array_equal(a, b)
