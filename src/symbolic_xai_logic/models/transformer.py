"""Transformer model for logic game solving."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Transformer encoder that treats the input as a sequence of tokens.
    Input is split into n_tokens tokens of d_model dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Determine token size: find a divisor of input_dim close to d_model
        token_size = None
        for ts in [d_model, d_model // 2, 1]:
            if input_dim % ts == 0:
                token_size = ts
                break
        if token_size is None:
            token_size = 1
        self.token_size = token_size
        self.n_tokens = input_dim // token_size

        self.input_proj = nn.Linear(token_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model * self.n_tokens, output_dim)

        self._activations: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = x.view(B, self.n_tokens, self.token_size)
        h = self.input_proj(tokens) + self.pos_encoding
        h = self.encoder(h)  # (B, n_tokens, d_model)
        self._activations["encoder_out"] = h.detach()
        h_flat = h.reshape(B, -1)
        return self.output_proj(h_flat)

    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        self.forward(x)
        return self._activations.get("encoder_out", x).reshape(x.shape[0], -1)
