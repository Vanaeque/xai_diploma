"""SHAP-based input-attribution explainer.

We compute per-input-feature Shapley values that say how much each input
contributes to a single scalar summary of the model's output (we use the
highest-variance output dimension across the explain split, which is the
most informative target — same logic as ``RuleExtractor._pick_target_dim``).

Three backends, tried in order:

1. **GradientExplainer** (preferred) — PyTorch-native, fast on GPU.
   Computes expected gradients over a background distribution.
2. **KernelExplainer** (fallback) — model-agnostic, slow but robust.
3. **Gradient × input** (last resort) — when ``shap`` isn't installed or
   both prior backends fail. Returns the same shape as SHAP would, so
   downstream metrics keep working unchanged.

All backends return ``attributions`` of shape ``(n_samples, input_dim)``,
which means feature-ablation metrics (comprehensiveness, sufficiency,
MoRF, LeRF) work out of the box — unlike concept_probe, whose attributions
are hidden activations and have to be skipped.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch

from .base import Explainer


class SHAPExplainer(Explainer):
    """SHAP feature attributions via PyTorch GradientExplainer when available."""

    def __init__(
        self,
        model,
        game,
        n_background: int = 20,
        max_explain: int = 20,
        target_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(model, game)
        # Conservative defaults so multiple parallel subprocesses sharing
        # one GPU don't compound to OOM.  SHAP's GradientExplainer allocates
        # ~O(n_background × n_explain × n_features × n_layers) for its
        # expected-gradient pass — on sudoku9 transformer that easily hits
        # 15+ GiB.  20/20 keeps each subprocess under ~1 GiB while still
        # giving Shapley-quality attribution estimates for thesis tables.
        self.n_background = n_background
        self.max_explain = max_explain
        # When None, ``explain`` picks the highest-variance output dim.
        # When set explicitly, SHAP attributions are computed for that dim.
        self.target_dim = target_dim
        self._chosen_target_dim: int | None = None
        # Save which backend ran for the report's summary field
        self._backend: str = "uninitialised"

    @property
    def name(self) -> str:
        return "shap"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forward pass; returns (n, output_dim) sigmoid probabilities on CPU/numpy."""
        device = self._device()
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=device)
            return torch.sigmoid(self.model(t)).detach().cpu().numpy()

    def _pick_target_dim(self, y_nn: np.ndarray) -> int:
        """Select the highest-variance output dim as the scalar SHAP target."""
        if self.target_dim is not None:
            return int(self.target_dim)
        if y_nn.ndim < 2:
            return 0
        var = y_nn.var(axis=0)
        return int(var.argmax()) if var.sum() > 0 else 0

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _gradient_explainer(
        self,
        X_explain: np.ndarray,
        background: np.ndarray,
        target_dim: int,
    ) -> np.ndarray | None:
        """SHAP GradientExplainer — fast, PyTorch-native, GPU-aware."""
        try:
            import shap  # type: ignore[import-not-found]
        except ImportError:
            return None
        device = self._device()

        # Wrap the model so GradientExplainer sees a scalar-output module.
        class _TargetSlice(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module, dim: int):
                super().__init__()
                self.inner = inner
                self.dim = dim

            def forward(self, x):
                return torch.sigmoid(self.inner(x))[:, self.dim:self.dim + 1]

        sliced = _TargetSlice(self.model, target_dim).to(device).eval()
        bg_t = torch.tensor(background, dtype=torch.float32, device=device)
        x_t = torch.tensor(X_explain, dtype=torch.float32, device=device)

        try:
            explainer = shap.GradientExplainer(sliced, bg_t)
            shap_values = explainer.shap_values(x_t)
        except Exception:
            return None

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        arr = np.asarray(shap_values, dtype=np.float32)
        # GradientExplainer may return (n, D, 1) — squeeze trailing 1.
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr if arr.ndim == 2 else None

    def _kernel_explainer(
        self,
        X_explain: np.ndarray,
        background: np.ndarray,
        target_dim: int,
    ) -> np.ndarray | None:
        """SHAP KernelExplainer — model-agnostic, slow.  Used as fallback."""
        try:
            import shap  # type: ignore[import-not-found]
        except ImportError:
            return None

        # Predict a single scalar per sample so KernelExplainer is well-defined.
        def predict_fn(x_np: np.ndarray) -> np.ndarray:
            return self._predict_proba(x_np)[:, target_dim]

        try:
            explainer = shap.KernelExplainer(predict_fn, background)
            # nsamples="auto" lets shap pick a reasonable per-sample budget;
            # we cap explain count externally so total cost stays bounded.
            shap_values = explainer.shap_values(X_explain, silent=True)
        except Exception:
            return None
        arr = np.asarray(shap_values, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr if arr.ndim == 2 else None

    def _gradient_input_fallback(
        self,
        X_explain: np.ndarray,
        target_dim: int,
    ) -> np.ndarray:
        """Gradient × input — last resort if shap is unavailable / fails."""
        device = self._device()
        self.model.eval()
        attributions: list[np.ndarray] = []
        for x_row in X_explain:
            x_t = torch.tensor(
                x_row, dtype=torch.float32, device=device, requires_grad=True,
            )
            out = torch.sigmoid(self.model(x_t.unsqueeze(0)))[0, target_dim]
            grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
            attributions.append((grad * x_t).detach().cpu().numpy())
        return np.asarray(attributions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        background: np.ndarray | None = None,
        max_explain: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        n_explain = int(max_explain) if max_explain is not None else self.max_explain
        X_explain = X[: min(n_explain, len(X))]
        bg = X if background is None else background
        bg = bg[: self.n_background]

        # Pick the scalar target dim once based on the explain split's stats.
        y_nn = self._predict_proba(X_explain)
        target_dim = self._pick_target_dim(y_nn)
        self._chosen_target_dim = target_dim

        # Try the three backends in priority order; the first one that returns
        # a valid (n, D) attribution matrix wins.
        for backend_name, backend in (
            ("gradient_explainer", self._gradient_explainer),
            ("kernel_explainer", self._kernel_explainer),
        ):
            if backend.__name__ == "_kernel_explainer":
                arr = backend(X_explain, bg, target_dim)
            else:
                arr = backend(X_explain, bg, target_dim)
            if arr is not None and arr.shape[0] > 0:
                self._backend = backend_name
                attributions = arr
                break
        else:
            self._backend = "gradient_input_fallback"
            attributions = self._gradient_input_fallback(X_explain, target_dim)

        return {
            "attributions": attributions,
            "method": "shap",
            "backend": self._backend,
            "target_dim": int(target_dim),
            "summary": (
                f"SHAP({self._backend}) attributions for {len(X_explain)} samples "
                f"on output dim {target_dim} ({len(bg)} background samples)"
            ),
            "mean_abs_attr": np.abs(attributions).mean(axis=0),
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Local-fidelity: how well a Ridge on the top-k SHAP-attributed
        features tracks the NN's scalar output (target_dim) on the same
        samples.  Returns 0..1 (clipped at 0)."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        attributions = explanation.get("attributions")
        if attributions is None or len(attributions) == 0:
            return 0.0
        target_dim = int(explanation.get("target_dim", self._chosen_target_dim or 0))

        mean_attr = np.abs(attributions).mean(axis=0)
        k = min(20, X.shape[1])
        top_k = np.argsort(mean_attr)[-k:]
        n = min(len(attributions), len(X))
        X_sub = X[:n, top_k]
        y_pred = self._predict_proba(X[:n])[:, target_dim]
        try:
            reg = Ridge(alpha=1.0).fit(X_sub, y_pred)
            y_lin = reg.predict(X_sub)
            return float(max(0.0, r2_score(y_pred, y_lin)))
        except Exception:
            return 0.0
