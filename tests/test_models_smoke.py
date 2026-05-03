"""Smoke tests for model forward passes."""
import pytest
import torch
import numpy as np

from symbolic_xai_logic.models.mlp import MLP
from symbolic_xai_logic.models.gnn import GNN
from symbolic_xai_logic.models.transformer import TransformerModel
from symbolic_xai_logic.models.cnn import CNN
from symbolic_xai_logic.models.registry import get_model


class TestMLP:
    def test_forward(self):
        model = MLP(input_dim=64, output_dim=16, hidden_dims=[32, 32])
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 16)

    def test_get_activations(self):
        model = MLP(input_dim=64, output_dim=16, hidden_dims=[32, 32])
        x = torch.randn(4, 64)
        acts = model.get_activations(x, layer=-1)
        assert acts.shape[0] == 4


class TestGNN:
    def test_forward(self):
        model = GNN(input_dim=64, output_dim=16, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 16)

    def test_get_activations(self):
        model = GNN(input_dim=64, output_dim=16, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 64)
        acts = model.get_activations(x)
        assert acts.shape[0] == 4


class TestTransformer:
    def test_forward(self):
        model = TransformerModel(input_dim=64, output_dim=16, d_model=32, nhead=2, num_layers=2)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 16)

    def test_get_activations(self):
        model = TransformerModel(input_dim=64, output_dim=16, d_model=32, nhead=2, num_layers=2)
        x = torch.randn(4, 64)
        acts = model.get_activations(x)
        assert acts.shape[0] == 4


class TestRegistry:
    def test_get_mlp(self):
        model = get_model("mlp", input_dim=32, output_dim=8)
        assert model is not None

    def test_get_gnn(self):
        model = get_model("gnn", input_dim=32, output_dim=8, hidden_dim=16, num_layers=2)
        assert model is not None

    def test_get_transformer(self):
        model = get_model("transformer", input_dim=32, output_dim=8, d_model=16, nhead=2, num_layers=2)
        assert model is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_model("unknown_model", input_dim=32, output_dim=8)


class TestModelWithSudokuDims:
    """Test models with actual Sudoku 4x4 dimensions."""

    def test_mlp_sudoku(self):
        input_dim = 4 * 4 * 4  # 64
        output_dim = 4 * 4 * 4
        model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[128, 64])
        x = torch.randn(8, input_dim)
        out = model(x)
        assert out.shape == (8, output_dim)


class TestCNNSudokuSpatialShapes:
    """Regression tests for the historical CNN sudoku shape mismatch.

    Errors observed in results/extended/sudoku{4,9}_medium_cnn_seed*/errors.txt:
        Target size (128, 80) must be the same as input size (128, 64)
        Target size (128, 810) must be the same as input size (128, 729)

    The cause was the runner picking input_dim from game.input_dim (one_hot=64)
    even when the data pipeline produced spatial encoding (80). The fix routes
    sudoku+spatial through game.spatial_input_dim and computes n_channels=size+1.
    """

    def test_cnn_sudoku4_spatial_forward(self):
        # Sudoku 4x4 spatial: (n+1, n, n) flattened = 5*4*4 = 80 input
        # Target one_hot solution: n*n*n = 64
        input_dim = 80
        output_dim = 64
        grid_size, n_channels = 4, 5
        model = CNN(
            input_dim=input_dim,
            output_dim=output_dim,
            grid_size=grid_size,
            n_channels=n_channels,
        )
        x = torch.randn(8, input_dim)
        out = model(x)
        assert out.shape == (8, output_dim), (
            f"CNN output {out.shape} must match BCE target shape (8, {output_dim})"
        )
        # Sanity: BCE on this pair must not raise
        target = torch.rand(8, output_dim)
        torch.nn.functional.binary_cross_entropy_with_logits(out, target)

    def test_cnn_sudoku9_spatial_forward(self):
        # Sudoku 9x9 spatial: 10*9*9 = 810 input; target = 9*9*9 = 729
        input_dim = 810
        output_dim = 729
        grid_size, n_channels = 9, 10
        model = CNN(
            input_dim=input_dim,
            output_dim=output_dim,
            grid_size=grid_size,
            n_channels=n_channels,
        )
        x = torch.randn(4, input_dim)
        out = model(x)
        assert out.shape == (4, output_dim)
        target = torch.rand(4, output_dim)
        torch.nn.functional.binary_cross_entropy_with_logits(out, target)

    def test_runner_sudoku4_cnn_spatial_dims(self):
        """End-to-end: runner.train_only must pick the spatial input_dim path."""
        from symbolic_xai_logic.experiments.runner import ExperimentRunner
        cfg = {
            "seed": 0,
            "device": "cpu",
            "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
            "model": {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
            "data":  {"n_train": 32, "n_val": 16, "n_test": 16, "encoding": "spatial"},
            "training": {
                "epochs": 1, "lr": 1e-3, "batch_size": 8, "eval_interval": 1,
                "checkpoint_dir": "/tmp/test_cnn_sudoku4_ckpt",
            },
            "xai": {"name": "rule_extraction"},
        }
        r = ExperimentRunner(cfg, results_dir="/tmp/test_cnn_sudoku4_results")
        # Must complete without the historical BCE shape error
        ckpt = r.train_only()
        assert ckpt.endswith("sudoku4_best.pt")
