"""Multi-Layer Perceptron for logic game solving."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    """Fully connected MLP with configurable depth and width."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(activation, nn.ReLU)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers += [nn.Linear(prev_dim, hdim), act_fn(), nn.Dropout(dropout)]
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Store intermediate activations for XAI
        self._activations: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.net.named_modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.GELU):
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = output.detach()
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """Forward pass and return activations from a given linear layer index."""
        self._activations.clear()
        activations = []
        inp = x
        linear_count = 0
        for module in self.net:
            inp = module(inp)
            if isinstance(module, nn.Linear):
                activations.append(inp.detach())
                linear_count += 1
        if layer < 0:
            layer = len(activations) + layer
        return activations[max(0, min(layer, len(activations) - 1))]
