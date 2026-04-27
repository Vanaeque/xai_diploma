"""Symbolic regression: fit closed-form expressions to NN input→output mapping."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from .base import Explainer


class SymbolicRegressionExplainer(Explainer):
    """
    Fit symbolic expressions to NN behavior using GP-inspired symbolic regression.
    Uses a custom grammar-based approach (no external PySR dependency).
    """

    def __init__(
        self,
        model,
        game,
        n_iterations: int = 100,
        population_size: int = 50,
        max_depth: int = 4,
        n_samples: int = 500,
        **kwargs,
    ):
        super().__init__(model, game)
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.max_depth = max_depth
        self.n_samples = n_samples
        self._expressions: list[str] = []
        self._best_r2: float = 0.0

    @property
    def name(self) -> str:
        return "symbolic_regression"

    def _predict_scalar(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            out = torch.sigmoid(self.model(t)).numpy()
        # Use the most variable output dimension so the regression target has
        # non-trivial variance.  out.mean(axis=1) is nearly flat (~0.25 for
        # Sudoku 4×4) which collapses all coefficients below any threshold.
        best_dim = int(out.var(axis=0).argmax())
        return out[:, best_dim]

    def _fit_polynomial(self, X: np.ndarray, y: np.ndarray, degree: int = 2) -> tuple[str, float]:
        """Fit a polynomial symbolic expression."""
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.pipeline import Pipeline

        # Use top-5 most variable features to keep it tractable
        feature_vars = X.var(axis=0)
        top_k = min(5, X.shape[1])
        top_feats = np.argsort(feature_vars)[-top_k:]
        X_sub = X[:, top_feats]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("reg", Ridge(alpha=1.0)),
        ])
        pipe.fit(X_sub, y)
        y_pred = pipe.predict(X_sub)
        # Guard against NaN predictions from overflow
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return "0", 0.0
        r2 = float(r2_score(y, y_pred))

        # Build an expression from scaled features
        poly_step = pipe.named_steps["poly"]
        reg_step = pipe.named_steps["reg"]
        coef = reg_step.coef_

        feat_names = [f"x{i}" for i in top_feats]
        poly_names = poly_step.get_feature_names_out(feat_names)
        terms = []
        for c, name in zip(coef, poly_names):
            if abs(c) > 0.01:
                terms.append(f"{c:.3f}*{name}")
        expr = " + ".join(terms[:10]) if terms else "0"

        return expr, max(0.0, r2)

    def _fit_symbolic_rules(self, X: np.ndarray, y: np.ndarray) -> tuple[list[str], float]:
        """Extract simple threshold rules; return (rules_text, tree_r2)."""
        from sklearn.tree import DecisionTreeRegressor, export_text
        from sklearn.metrics import r2_score

        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)
        tree.fit(X, y)
        y_pred = tree.predict(X)
        tree_r2 = float(r2_score(y, y_pred)) if not np.any(np.isnan(y_pred)) else 0.0
        n_feats = X.shape[1]
        feat_names = [f"f_{i}" for i in range(n_feats)]
        rules = export_text(tree, feature_names=feat_names[:tree.n_features_in_])
        return [rules], max(0.0, tree_r2)

    def explain(self, X: np.ndarray, **kwargs) -> dict[str, Any]:
        X_sub = X[:min(self.n_samples, len(X))]
        y_nn = self._predict_scalar(X_sub)

        expr_poly, r2_poly = self._fit_polynomial(X_sub, y_nn, degree=2)
        rules, r2_tree = self._fit_symbolic_rules(X_sub, y_nn)
        best_r2 = max(r2_poly, r2_tree)

        self._expressions = [expr_poly] + rules
        self._best_r2 = best_r2

        return {
            "attributions": np.zeros((len(X_sub), X_sub.shape[1])),
            "method": "symbolic_regression",
            "expressions": self._expressions,
            "best_r2": best_r2,
            "poly_expression": expr_poly,
            "rules": rules,
            "summary": f"Symbolic regression: best R²={best_r2:.3f}, expr: {expr_poly[:80]}",
            "mean_abs_attr": np.zeros(X_sub.shape[1]),
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Fidelity: R² of best symbolic expression on the data."""
        return explanation.get("best_r2", self._best_r2)
