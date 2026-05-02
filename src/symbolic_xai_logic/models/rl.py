"""Reinforcement Learning model for logic game solving.

Policy network with value head for game-solving via RL.
Supports actor-critic architecture for training with policy gradients.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence


class RL(nn.Module):
    """
    Reinforcement Learning policy network with optional value head.
    
    Architecture:
        shared feature layers → policy head (action logits)
                             → value head (state value estimate)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.1,
        activation: str = "relu",
        use_value_head: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_value_head = use_value_head

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(activation, nn.ReLU)

        # Shared feature extraction layers
        feature_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            feature_layers += [nn.Linear(prev_dim, hdim), act_fn(), nn.Dropout(dropout)]
            prev_dim = hdim
        self.feature_net = nn.Sequential(*feature_layers)

        # Policy head: outputs logits for each action
        self.policy_head = nn.Linear(prev_dim, output_dim)

        # Value head: estimates state value (optional)
        if use_value_head:
            self.value_head = nn.Linear(prev_dim, 1)
        else:
            self.value_head = None

        # Store intermediate activations for XAI
        self._activations: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.feature_net.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh)):
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = output.detach()
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: outputs policy logits (action probabilities).
        
        Args:
            x: State/observation tensor of shape (batch_size, input_dim)
            
        Returns:
            Policy logits of shape (batch_size, output_dim)
        """
        features = self.feature_net(x)
        policy_logits = self.policy_head(features)
        return policy_logits

    def get_policy_and_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get both policy logits and value estimate.
        
        Args:
            x: State/observation tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (policy_logits, value_estimate)
            - policy_logits: shape (batch_size, output_dim)
            - value_estimate: shape (batch_size, 1) if use_value_head, else None
        """
        features = self.feature_net(x)
        policy_logits = self.policy_head(features)
        
        if self.use_value_head:
            value = self.value_head(features)
            return policy_logits, value
        return policy_logits, None

    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """
        Forward pass and return activations from a given layer.
        
        Args:
            x: Input tensor
            layer: Layer index (negative indices count from end)
            
        Returns:
            Activations from the specified layer
        """
        self._activations.clear()
        activations = []
        inp = x
        for module in self.feature_net:
            inp = module(inp)
            if isinstance(module, nn.Linear):
                activations.append(inp.detach())
        if layer < 0:
            layer = len(activations) + layer
        return activations[max(0, min(layer, len(activations) - 1))]
