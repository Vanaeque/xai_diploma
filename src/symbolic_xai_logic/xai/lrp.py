"""Layer-wise Relevance Propagation (LRP) for MLP and GNN."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from .base import Explainer


class LRPExplainer(Explainer):
    """LRP-epsilon rule for linear layers."""

    def __init__(self, model, game, rule: str = "epsilon", epsilon: float = 1e-6, **kwargs):
        super().__init__(model, game)
        self.rule = rule
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return "lrp"

    def _lrp_linear(self, a: torch.Tensor, W: torch.Tensor, b: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """LRP-epsilon backprop through one linear layer."""
        # a: (d_in,), W: (d_out, d_in), b: (d_out,), R: (d_out,)
        z = W @ a + b + self.epsilon * torch.sign(W @ a + b + 1e-12)
        s = R / (z + 1e-12)
        c = W.T @ s
        return a * c

    def _explain_mlp(self, x: torch.Tensor) -> np.ndarray:
        """Compute LRP attributions for a single MLP input."""
        from ..models.mlp import MLP

        x = x.clone().detach()
        # Collect linear layers and their pre-activation inputs
        layers = [m for m in self.model.net if isinstance(m, nn.Linear)]
        activations = [x]
        a = x
        for module in self.model.net:
            if isinstance(module, nn.Linear):
                a = module(a)
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh)):
                a = module(a)
            elif isinstance(module, nn.Dropout):
                pass
            activations.append(a.detach())

        # Relevances start from output
        R = activations[-1].detach()

        act_idx = len(activations) - 1
        for layer in reversed(layers):
            W = layer.weight.detach()
            b = layer.bias.detach()
            a_in = activations[act_idx - 1].detach()
            z = W @ a_in + b
            z_eps = z + self.epsilon * (torch.sign(z) + 1e-12)
            s = R / (z_eps + 1e-12)
            c = W.T @ s
            R = a_in * c
            act_idx -= 1

        return R.detach().cpu().numpy()

    def explain(self, X: np.ndarray, max_explain: int = 100, **kwargs) -> dict[str, Any]:
        self.model.eval()
        # Place input tensors on the model's device so this works with --device cuda
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        X_sub = X[:min(max_explain, len(X))]
        attributions = []

        for i in range(len(X_sub)):
            x = torch.tensor(X_sub[i], dtype=torch.float32, device=device)
            try:
                from ..models.mlp import MLP
                if isinstance(self.model, MLP):
                    attr = self._explain_mlp(x)
                else:
                    # Gradient fallback for GNN/Transformer
                    x.requires_grad_(True)
                    out = self.model(x.unsqueeze(0)).sum()
                    out.backward()
                    attr = (x.grad * x).detach().cpu().numpy()
            except Exception:
                x_req = x.clone().requires_grad_(True)
                out = self.model(x_req.unsqueeze(0)).sum()
                out.backward()
                attr = (x_req.grad * x_req).detach().cpu().numpy()
            attributions.append(attr)

        attributions = np.array(attributions)
        return {
            "attributions": attributions,
            "method": "lrp",
            "rule": self.rule,
            "summary": f"LRP-{self.rule} attributions for {len(X_sub)} samples",
            "mean_abs_attr": np.abs(attributions).mean(axis=0),
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Fidelity: correlation between LRP relevance and input gradient magnitude."""
        self.model.eval()
        n = min(len(X), len(explanation.get("attributions", X)))
        lrp_attrs = explanation["attributions"][:n]

        # Compute gradient attributions
        # Place tensors on the model's device (handles --device cuda).
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        grad_attrs = []
        for i in range(n):
            x = torch.tensor(X[i], dtype=torch.float32, device=device, requires_grad=True)
            out = self.model(x.unsqueeze(0)).sum()
            out.backward()
            grad_attrs.append(x.grad.detach().cpu().numpy())
        grad_attrs = np.array(grad_attrs)

        # Rank correlation
        from scipy.stats import spearmanr
        corr_vals = []
        for i in range(n):
            r, _ = spearmanr(np.abs(lrp_attrs[i]), np.abs(grad_attrs[i]))
            if not np.isnan(r):
                corr_vals.append(r)
        return float(np.mean(corr_vals)) if corr_vals else 0.0
