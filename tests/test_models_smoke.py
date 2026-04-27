"""Smoke tests for model forward passes."""
import pytest
import torch
import numpy as np

from symbolic_xai_logic.models.mlp import MLP
from symbolic_xai_logic.models.gnn import GNN
from symbolic_xai_logic.models.transformer import TransformerModel
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
