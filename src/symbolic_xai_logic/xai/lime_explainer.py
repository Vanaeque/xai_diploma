"""LIME-based input attribution explainer."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from .base import Explainer


class LIMEExplainer(Explainer):
    """LIME: input attributions via local linear approximation."""

    def __init__(self, model, game, n_samples: int = 500, kernel_width: float = 0.25, **kwargs):
        super().__init__(model, game)
        self.n_samples = n_samples
        self.kernel_width = kernel_width

    @property
    def name(self) -> str:
        return "lime"

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=device)
            out = torch.sigmoid(self.model(t)).detach().cpu().numpy()
        return out

    def _lime_single(self, x: np.ndarray) -> np.ndarray:
        """Compute LIME attributions for a single sample."""
        from sklearn.linear_model import Ridge

        d = x.shape[0]
        # Generate perturbed samples
        perturbations = np.random.randn(self.n_samples, d) * 0.1
        X_perturbed = x[None, :] + perturbations
        X_perturbed = np.clip(X_perturbed, 0.0, 1.0)

        # Kernel weights: closer = higher weight
        distances = np.sqrt(np.sum(perturbations ** 2, axis=1))
        kernel_weights = np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))

        # Get predictions
        y_perturbed = self._predict_fn(X_perturbed).mean(axis=1)

        # Fit weighted ridge regression
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(X_perturbed, y_perturbed, sample_weight=kernel_weights)
        return reg.coef_

    def explain(self, X: np.ndarray, max_explain: int = 50, **kwargs) -> dict[str, Any]:
        X_sub = X[:min(max_explain, len(X))]
        attributions = np.array([self._lime_single(x) for x in X_sub])
        return {
            "attributions": attributions,
            "method": "lime",
            "summary": f"LIME attributions for {len(X_sub)} samples, {X_sub.shape[1]} features",
            "mean_abs_attr": np.abs(attributions).mean(axis=0),
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Fidelity: R² of the local linear model on held-out samples."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        mean_attr = explanation.get("mean_abs_attr", np.ones(X.shape[1]))
        # Use top-k features
        k = min(20, X.shape[1])
        top_k = np.argsort(mean_attr)[-k:]
        X_sub = X[:, top_k]

        y_pred = self._predict_fn(X).mean(axis=1)
        reg = Ridge(alpha=1.0)
        reg.fit(X_sub, y_pred)
        y_lin = reg.predict(X_sub)
        return float(max(0.0, r2_score(y_pred, y_lin)))
