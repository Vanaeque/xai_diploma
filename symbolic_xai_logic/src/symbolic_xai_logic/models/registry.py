"""Model registry: name -> class mapping."""
from __future__ import annotations
import torch.nn as nn
from .mlp import MLP
from .gnn import GNN
from .transformer import TransformerModel
from .cnn import CNN

MODEL_REGISTRY: dict[str, type] = {
    "mlp": MLP,
    "gnn": GNN,
    "transformer": TransformerModel,
    "cnn": CNN,
}


def get_model(name: str, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_dim=input_dim, output_dim=output_dim, **kwargs)
