"""
Extended XAI evaluation metrics.

  Faithfulness  : comprehensiveness, sufficiency  (DeYoung et al. 2020)
  Stability     : max_sensitivity                  (Yeh et al. 2019)
  Sanity        : model_randomization_score        (Adebayo et al. 2018)
                  data_randomization_score         (ibid.)
  Recoverability: semantic_equivalence_z3
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_attributions(explanation: dict) -> np.ndarray | None:
    """Return (n_samples, n_features) attribution matrix, or None."""
    arr = explanation.get("attributions")
    if arr is not None:
        a = np.asarray(arr, dtype=np.float32)
        return a if a.ndim == 2 else None
    for key in ("mean_abs_attr", "feature_importances"):
        arr = explanation.get(key)
        if arr is not None:
            return np.asarray(arr, dtype=np.float32)[None, :]
    return None


def _predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# ---------------------------------------------------------------------------
# Faithfulness metrics
# ---------------------------------------------------------------------------

def comprehensiveness(
    model: nn.Module,
    explainer: Any,
    X: np.ndarray,
    top_k: int = 10,
    baseline: float = 0.0,
    max_samples: int = 100,
) -> float:
    """
    Drop top-k features; measure mean |f(x) - f(x_ablated)|.
    Higher → top features are genuinely informative.
    Ref: DeYoung et al. 2020 (ERASER benchmark).
    """
    n = min(max_samples, len(X))
    explanation = explainer.explain(X[:n])
    attrs = _extract_attributions(explanation)
    if attrs is None:
        return 0.0

    pred_orig = _predict_proba(model, X[:n])  # (n, output_dim)
    scores = []
    for i in range(min(n, len(attrs))):
        top_idx = np.argsort(np.abs(attrs[i]))[-top_k:]
        x_abl = X[i].copy()
        x_abl[top_idx] = baseline
        pred_abl = _predict_proba(model, x_abl[None])[0]
        scores.append(float(np.abs(pred_orig[i] - pred_abl).mean()))
    return float(np.mean(scores)) if scores else 0.0


def sufficiency(
    model: nn.Module,
    explainer: Any,
    X: np.ndarray,
    top_k: int = 10,
    baseline: float = 0.0,
    max_samples: int = 100,
) -> float:
    """
    Keep only top-k features; measure 1 - mean |f(x) - f(x_masked)|.
    Higher → top features alone are sufficient to recover the prediction.
    Ref: DeYoung et al. 2020 (ERASER benchmark).
    """
    n = min(max_samples, len(X))
    explanation = explainer.explain(X[:n])
    attrs = _extract_attributions(explanation)
    if attrs is None:
        return 0.0

    pred_orig = _predict_proba(model, X[:n])
    scores = []
    for i in range(min(n, len(attrs))):
        top_idx = np.argsort(np.abs(attrs[i]))[-top_k:]
        x_kept = np.full_like(X[i], baseline)
        x_kept[top_idx] = X[i][top_idx]
        pred_kept = _predict_proba(model, x_kept[None])[0]
        scores.append(1.0 - float(np.abs(pred_orig[i] - pred_kept).mean()))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Stability metric
# ---------------------------------------------------------------------------

def max_sensitivity(
    explainer: Any,
    X: np.ndarray,
    n_reruns: int = 10,
    noise_std: float = 0.01,
    max_samples: int = 20,
) -> float:
    """
    Max L2 distance between explanation attributions under small input
    perturbations (Yeh et al. 2019).  Lower → more stable.
    """
    n = min(max_samples, len(X))
    base_exp = explainer.explain(X[:n])
    attrs_base = _extract_attributions(base_exp)
    if attrs_base is None:
        return 0.0

    rng = np.random.default_rng(0)
    # Use actual attribution count, not n (single-vector methods return 1 row)
    n_attr = len(attrs_base)
    max_dists = np.zeros(n_attr)

    for _ in range(n_reruns):
        noise = rng.normal(0.0, noise_std, size=X[:n].shape).astype(np.float32)
        X_noisy = np.clip(X[:n] + noise, 0.0, 1.0)
        noisy_exp = explainer.explain(X_noisy)
        attrs_noisy = _extract_attributions(noisy_exp)
        if attrs_noisy is None:
            continue
        m = min(n_attr, len(attrs_noisy))
        diffs = attrs_base[:m] - attrs_noisy[:m]
        dists = np.linalg.norm(diffs, axis=1)
        max_dists[:m] = np.maximum(max_dists[:m], dists)

    return float(max_dists.mean())


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def model_randomization_score(
    model: nn.Module,
    explainer: Any,
    X: np.ndarray,
    attrs_real: np.ndarray,
    n_samples: int = 30,
) -> float:
    """
    Model parameter randomization test (Adebayo et al. 2018).

    Reinitialise model weights; compare attributions to the trained model.
    Returns mean |Spearman r| over samples.
    Low ≈ 0 → explanation depends on learned weights (good sanity).
    High ≈ 1 → explanation ignores the model (bad — same as random baseline).
    """
    from scipy.stats import spearmanr

    n = min(n_samples, len(X), len(attrs_real))
    saved_state = copy.deepcopy(model.state_dict())

    try:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        model.eval()
        rand_exp = explainer.explain(X[:n])
        attrs_rand = _extract_attributions(rand_exp)
    finally:
        model.load_state_dict(saved_state)
        model.eval()

    if attrs_rand is None:
        return 0.0

    corrs = []
    for i in range(min(n, len(attrs_rand))):
        a, b = np.abs(attrs_real[i]), np.abs(attrs_rand[i])
        if a.std() < 1e-8 or b.std() < 1e-8:
            corrs.append(0.0)
            continue
        r, _ = spearmanr(a, b)
        if not np.isnan(r):
            corrs.append(abs(float(r)))

    return float(np.mean(corrs)) if corrs else 0.0


def data_randomization_score(
    explainer: Any,
    X: np.ndarray,
    y_nn: np.ndarray,
    n_samples: int = 30,
) -> float:
    """
    Data randomization test (Adebayo et al. 2018, adapted for surrogates).

    Fits the surrogate decision tree on shuffled NN predictions and compares
    feature importances to those from the real predictions.
    Low |Spearman r| ≈ 0 → surrogate captures actual model patterns (good).
    High ≈ 1 → surrogate depends only on input structure, ignores labels (bad).

    Only defined for rule-extraction / decision-tree explainers.
    Returns NaN for attribution-based methods (retraining required).
    """
    from scipy.stats import spearmanr

    if not hasattr(explainer, "_tree") or explainer._tree is None:
        return float("nan")

    fi_real = explainer._tree.feature_importances_
    n = min(n_samples, len(X), len(y_nn))

    rng = np.random.default_rng(42)
    y_shuf = y_nn[:n].copy()
    rng.shuffle(y_shuf)

    y_bin: np.ndarray
    if y_shuf.ndim > 1:
        y_bin = (y_shuf[:, int(y_shuf.var(axis=0).argmax())] > 0.5).astype(int)
    else:
        y_bin = (y_shuf > 0.5).astype(int)

    from sklearn.tree import DecisionTreeClassifier
    tree_rand = DecisionTreeClassifier(
        max_depth=getattr(explainer, "max_depth", 5),
        min_samples_leaf=getattr(explainer, "min_samples_leaf", 10),
        random_state=42,
    )
    X_bin = (X[:n] > 0.5).astype(np.float32)
    try:
        tree_rand.fit(X_bin, y_bin)
    except Exception:
        return float("nan")

    fi_rand = tree_rand.feature_importances_
    r, _ = spearmanr(fi_real, fi_rand)
    return float(abs(r)) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Recoverability metric
# ---------------------------------------------------------------------------

def semantic_equivalence_z3(
    game: Any,
    model: nn.Module,
    X_test: np.ndarray,
    puzzles: list,
    n_samples: int = 50,
) -> float:
    """
    Semantic equivalence via Z3: fraction of cells where NN prediction agrees
    with the Z3 symbolic solver's ground-truth solution.

    This is the core recoverability metric for the diploma thesis:
      - Minesweeper: does the model recover "provably-mine" positions?
      - Sudoku: does the model recover correct digit placements?

    Uses game.solve_symbolic() as the oracle (Z3 when available, falls back to
    constraint propagation).
    """
    n = min(n_samples, len(X_test), len(puzzles))
    preds = _predict_proba(model, X_test[:n])  # (n, output_dim)

    agreements = []
    for i in range(n):
        try:
            z3_sol = game.solve_symbolic(puzzles[i])
        except Exception:
            continue
        if z3_sol is None:
            continue
        z3_flat = np.array(z3_sol, dtype=np.float32).reshape(-1)
        z3_bin = (z3_flat > 0.5).astype(int)
        pred_bin = (preds[i] > 0.5).astype(int)
        m = min(len(z3_bin), len(pred_bin))
        agreements.append(float((pred_bin[:m] == z3_bin[:m]).mean()))

    return float(np.mean(agreements)) if agreements else 0.0


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, bool] = {
    "fidelity": True,
    "comprehensiveness": False,
    "sufficiency": False,
    "max_sensitivity": True,
    "model_randomization": True,
    "data_randomization": True,
    "semantic_equivalence_z3": False,
}

_PER_METHOD: dict[str, dict[str, bool]] = {
    "lime":             {"comprehensiveness": True, "sufficiency": True},
    "shap":             {"comprehensiveness": True, "sufficiency": True},
    "lrp":              {"comprehensiveness": True, "sufficiency": True},
    "rule_extraction":  {"semantic_equivalence_z3": True},
    "symbolic_regression": {},
    "concept_probe":    {"max_sensitivity": False},
}


def resolve_metrics(xai_name: str, metrics_cfg: dict | None = None) -> dict:
    """Return the merged metric flags for the given XAI method."""
    cfg = metrics_cfg or {}
    flags = {**_DEFAULTS}
    flags.update(_PER_METHOD.get(xai_name, {}))
    # User overrides from config
    flags.update({k: v for k, v in cfg.items() if isinstance(v, bool)})
    per = cfg.get("per_method", {}).get(xai_name, {})
    flags.update(per)
    return flags
