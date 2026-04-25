"""Convolutional Neural Network for spatial logic games (Minesweeper, grid puzzles).

Input convention: flat vector of length n_channels * height * width, stored in
channel-first order (C, H, W) so the CNN can reshape internally without any
extra bookkeeping in the data pipeline.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class CNN(nn.Module):
    """
    Small CNN designed for translation-equivariant games (Minesweeper).

    Architecture:
        conv block × num_layers  →  global-avg-pool  →  linear head

    Input:  flat vector (n_channels * H * W,) – channel-first order.
    Output: flat vector (output_dim,) – per-cell logits.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int,
        n_channels: int,
        channels: Sequence[int] = (32, 64, 64),
        kernel_size: int = 3,
        num_layers: int | None = None,   # if set, overrides len(channels)
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.n_channels = n_channels

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}.get(activation, nn.ReLU)

        # Optionally override the channel list with a uniform depth
        if num_layers is not None:
            ch = channels[0] if len(channels) else 32
            channels = [ch] * num_layers

        conv_layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in channels:
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                act_fn(),
                nn.Dropout2d(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.final_channels = in_ch

        # Per-cell prediction head: 1×1 conv → reshape to (B, output_dim)
        # output_dim == grid_size * grid_size for per-cell binary tasks
        self.head = nn.Conv2d(in_ch, output_dim // (grid_size * grid_size) or 1, kernel_size=1)
        self._head_out_ch = output_dim // (grid_size * grid_size) if output_dim % (grid_size * grid_size) == 0 else 1

        # Fallback linear head when output_dim doesn't factor evenly into the grid
        if output_dim != self._head_out_ch * grid_size * grid_size:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch * grid_size * grid_size, output_dim),
            )
            self._use_flat_head = True
        else:
            self._use_flat_head = False

        self._activations: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Reshape flat input to (B, C, H, W)
        spatial = x.view(B, self.n_channels, self.grid_size, self.grid_size)
        feat = self.conv(spatial)                     # (B, final_channels, H, W)
        self._activations["conv_out"] = feat.detach()

        if self._use_flat_head:
            return self.head(feat)                    # linear head handles flatten
        else:
            out = self.head(feat)                     # (B, out_ch, H, W)
            return out.view(B, -1)                    # (B, output_dim)

    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        self.forward(x)
        feat = self._activations.get("conv_out", x)
        return feat.reshape(x.shape[0], -1)
