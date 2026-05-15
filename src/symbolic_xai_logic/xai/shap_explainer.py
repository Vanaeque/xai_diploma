"""SHAP-based input attribution explainer."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from .base import Explainer


class SHAPExplainer(Explainer):
    """SHAP: Shapley value-based attribution using kernel SHAP."""

    def __init__(self, model, game, n_background: int = 100, max_evals: int = 500, **kwargs):
        super().__init__(model, game)
        self.n_background = n_background
        self.max_evals = max_evals

    @property
    def name(self) -> str:
        return "shap"

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=device)
            out = torch.sigmoid(self.model(t)).detach().cpu().numpy()
        return out.mean(axis=1)

    def explain(self, X: np.ndarray, background: np.ndarray | None = None, max_explain: int = 50, **kwargs) -> dict[str, Any]:
        try:
            import shap
            bg = background[:self.n_background] if background is not None else X[:self.n_background]
            X_sub = X[:min(max_explain, len(X))]

            explainer = shap.KernelExplainer(self._predict_fn, bg)
            shap_values = explainer.shap_values(X_sub, nsamples=self.max_evals, silent=True)
            if isinstance(shap_values, list):
                attributions = np.array(shap_values[0])
            else:
                attributions = np.array(shap_values)
            return {
                "attributions": attributions,
                "method": "shap",
                "summary": f"SHAP attributions for {len(X_sub)} samples",
                "mean_abs_attr": np.abs(attributions).mean(axis=0),
            }
        except Exception as e:
            # Fallback: gradient-based approximation
            return self._gradient_fallback(X, str(e))

    def _gradient_fallback(self, X: np.ndarray, reason: str) -> dict[str, Any]:
        """Gradient × input as fallback when SHAP unavailable."""
        self.model.eval()
        X_t = torch.tensor(X[:50], dtype=torch.float32, requires_grad=True)
        out = self.model(X_t).sum()
        out.backward()
        attributions = (X_t.grad * X_t).detach().numpy()
        return {
            "attributions": attributions,
            "method": "shap_fallback",
            "summary": f"Gradient×input fallback (SHAP failed: {reason})",
            "mean_abs_attr": np.abs(attributions).mean(axis=0),
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        attributions = explanation.get("attributions", np.zeros((len(X), X.shape[1])))
        mean_attr = np.abs(attributions).mean(axis=0)
        k = min(20, X.shape[1])
        top_k = np.argsort(mean_attr)[-k:]
        X_sub = X[:len(attributions), top_k]
        y_pred = self._predict_fn(X[:len(attributions)])
        reg = Ridge(alpha=1.0)
        reg.fit(X_sub, y_pred)
        y_lin = reg.predict(X_sub)
        return float(max(0.0, r2_score(y_pred, y_lin)))
