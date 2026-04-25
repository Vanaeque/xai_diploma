"""Graph Neural Network for constraint-graph logic games."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Simple Graph Convolutional layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: (N, N) or (B, N, N) normalized adjacency
        if adj.dim() == 2:
            out = torch.mm(adj, x)
        else:
            out = torch.bmm(adj, x)
        return F.relu(self.linear(out))


class GNN(nn.Module):
    """
    GCN/GAT-style GNN for logic games encoded as flat vectors.
    For games without explicit graph structure, builds a fully-connected constraint graph.
    Accepts flat input, reshapes to node features, processes, flattens back.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        node_feat_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Treat input as n_nodes nodes with node_feat_dim features each
        if node_feat_dim is None:
            # Choose a factoring: node_feat_dim such that input_dim is divisible
            for nfd in [4, 8, 16, 32, 1]:
                if input_dim % nfd == 0:
                    node_feat_dim = nfd
                    break
            else:
                node_feat_dim = 1
        self.node_feat_dim = node_feat_dim
        self.n_nodes = input_dim // node_feat_dim

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * self.n_nodes, output_dim)

        # Cache activations for XAI
        self._activations: dict[str, torch.Tensor] = {}

    def _build_adj(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        """Build normalized fully-connected adjacency matrix."""
        adj = torch.ones(n_nodes, n_nodes, device=device) / n_nodes
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Reshape to (B, n_nodes, node_feat_dim)
        x_nodes = x.view(B, self.n_nodes, self.node_feat_dim)
        h = F.relu(self.input_proj(x_nodes))  # (B, n_nodes, hidden_dim)

        adj = self._build_adj(self.n_nodes, x.device)  # (n_nodes, n_nodes)
        adj_batch = adj.unsqueeze(0).expand(B, -1, -1)  # (B, n_nodes, n_nodes)

        for conv in self.conv_layers:
            h = conv(h, adj_batch)
            h = self.dropout_layer(h)

        self._activations["final_node"] = h.detach()

        # Flatten and project
        h_flat = h.reshape(B, -1)
        return self.output_proj(h_flat)

    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        self.forward(x)
        return self._activations.get("final_node", x).reshape(x.shape[0], -1)
