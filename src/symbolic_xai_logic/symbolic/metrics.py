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


def _attrs_index_input_features(attrs: np.ndarray, X: np.ndarray) -> bool:
    """True iff the attribution columns index input features 1-to-1.

    Some explainers (notably concept_probe) put **hidden-layer activations**
    in the ``attributions`` field — those have arbitrary dimensionality
    unrelated to ``X.shape[1]`` and CANNOT be used for feature-ablation
    metrics (the top-k indices would point at activation columns, not
    input features, and ``x.flat[top_idx]`` either crashes or silently
    corrupts the input).  This guard lets the ablation metrics return
    NaN cleanly rather than throwing IndexError mid-fidelity.
    """
    if attrs is None or X is None:
        return False
    input_flat = int(np.prod(X.shape[1:])) if X.ndim >= 2 else int(X.shape[0])
    return attrs.ndim == 2 and attrs.shape[1] == input_flat


def _predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Forward-pass a numpy array through the model, return numpy probs.

    Reads the device off the model's parameters so this works regardless of
    whether the model lives on CPU or CUDA — without this, callers from
    explain.py (which moves models to --device cuda) hit a device-mismatch
    RuntimeError on the first comprehensiveness / sufficiency / MoRF call.
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=device)
        out = torch.sigmoid(model(t)).detach().cpu().numpy()
    return out


def surrogate_fidelity(
    model: nn.Module,
    explanation: dict,
    X: np.ndarray,
    top_k: int = 20,
    max_depth: int = 4,
    cv: int = 3,
) -> float:
    """Method-agnostic fidelity to the NN — comparable across XAI methods.

    Native ``Explainer.fidelity()`` returns whatever each method computes
    (cross-val accuracy for surrogate trees, gradient correlation for LRP,
    R² for symbolic regression, concept-probe accuracy for concept_probe).
    These are **not comparable** across methods — see audit notes.

    This metric uses the SAME formula for every XAI method:

        1. Take the explanation's top-k features (by mean |attribution|).
        2. Fit a depth-``max_depth`` decision tree on those features → NN labels.
        3. Return ``cross_val_score`` mean accuracy.

    Higher = the features the explainer flagged as important *actually are*
    predictive of the NN's output.  Returns NaN when no usable attributions
    are present (e.g. concept_probe with all-degenerate probes).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    attrs = _extract_attributions(explanation)
    if attrs is None or attrs.size == 0:
        return float("nan")

    mean_attr = np.abs(attrs).mean(axis=0) if attrs.ndim == 2 else np.abs(attrs).ravel()
    if mean_attr.size == 0 or mean_attr.sum() == 0.0:
        return float("nan")

    y_nn = _predict_proba(model, X)
    if y_nn.ndim > 1:
        # Multi-output → reduce to the highest-variance dim (most informative target)
        target_dim = int(y_nn.var(axis=0).argmax())
        y_target = (y_nn[:, target_dim] > 0.5).astype(int)
    else:
        y_target = (y_nn > 0.5).astype(int)

    n = min(len(X), len(y_target))
    if len(np.unique(y_target[:n])) < 2:
        return float("nan")  # single-class — CV undefined

    k = min(top_k, mean_attr.size, X.shape[1])
    top_idx = np.argsort(mean_attr)[-k:]
    X_sub = X[:n, top_idx]

    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=10, random_state=42
    )
    try:
        scores = cross_val_score(tree, X_sub, y_target[:n], cv=cv, scoring="accuracy")
    except Exception:
        return float("nan")
    val = float(scores.mean())
    return val if np.isfinite(val) else float("nan")


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
        return float("nan")
    # Attributions whose columns aren't input-feature scores (e.g. concept_probe
    # returns hidden activations) cannot drive ablation — bail out cleanly.
    if not _attrs_index_input_features(attrs, X):
        return float("nan")

    pred_orig = _predict_proba(model, X[:n])  # (n, output_dim)
    scores = []
    for i in range(min(n, len(attrs))):
        top_idx = np.argsort(np.abs(attrs[i]))[-top_k:]
        x_abl = X[i].copy()
        # Handle multidimensional arrays by flattening, indexing, and reshaping
        x_abl_shape = x_abl.shape
        x_abl.flat[top_idx] = baseline
        pred_abl = _predict_proba(model, x_abl[None])[0]
        scores.append(float(np.abs(pred_orig[i] - pred_abl).mean()))
    return float(np.mean(scores)) if scores else 0.0


def morf_curve(
    model: nn.Module,
    explainer: Any,
    X: np.ndarray,
    k_grid: list[int] | None = None,
    baseline: float = 0.0,
    max_samples: int = 50,
) -> dict:
    """Most-Relevant-First removal curve (Samek et al., 2017).

    Greedily ablates the top-k attributed features for k in ``k_grid``;
    measures mean |Δoutput| at each k.  Higher final value AND a steep early
    rise → attributions concentrate on truly important features.

    Returns
    -------
    {
      "k_grid": [...],
      "mean_drop": [...],   # parallel to k_grid; mean across samples
    }
    """
    n = min(max_samples, len(X))
    if k_grid is None:
        D = X.shape[1]
        # Geometric-ish grid up to half the features
        k_grid = sorted(set(
            [1, 2, 5, 10, 20, 50, 100, max(1, D // 4), max(1, D // 2)]
            if D > 4 else list(range(1, D + 1))
        ))
        k_grid = [k for k in k_grid if k <= D]

    explanation = explainer.explain(X[:n])
    attrs = _extract_attributions(explanation)
    if attrs is None or not _attrs_index_input_features(attrs, X):
        return {"k_grid": list(k_grid), "mean_drop": [0.0] * len(k_grid)}

    pred_orig = _predict_proba(model, X[:n])
    drops_per_k: list[float] = []
    for k in k_grid:
        sample_drops = []
        for i in range(min(n, len(attrs))):
            order = np.argsort(np.abs(attrs[i]))[::-1]  # most → least important
            top_idx = order[:k]
            x_abl = X[i].copy()
            # Handle multidimensional arrays by flattening, indexing, and reshaping
            x_abl.flat[top_idx] = baseline
            pred_abl = _predict_proba(model, x_abl[None])[0]
            sample_drops.append(float(np.abs(pred_orig[i] - pred_abl).mean()))
        drops_per_k.append(float(np.mean(sample_drops)) if sample_drops else 0.0)
    return {"k_grid": list(k_grid), "mean_drop": drops_per_k}


def lerf_curve(
    model: nn.Module,
    explainer: Any,
    X: np.ndarray,
    k_grid: list[int] | None = None,
    baseline: float = 0.0,
    max_samples: int = 50,
) -> dict:
    """Least-Relevant-First removal curve.  Mirror of ``morf_curve`` but
    ablates the *least*-attributed features first.  A faithful explanation
    should produce a low, flat early region and a steep rise only when the
    truly important features are reached at the end.
    """
    n = min(max_samples, len(X))
    if k_grid is None:
        D = X.shape[1]
        k_grid = sorted(set(
            [1, 2, 5, 10, 20, 50, 100, max(1, D // 4), max(1, D // 2)]
            if D > 4 else list(range(1, D + 1))
        ))
        k_grid = [k for k in k_grid if k <= D]

    explanation = explainer.explain(X[:n])
    attrs = _extract_attributions(explanation)
    if attrs is None or not _attrs_index_input_features(attrs, X):
        return {"k_grid": list(k_grid), "mean_drop": [0.0] * len(k_grid)}

    pred_orig = _predict_proba(model, X[:n])
    drops_per_k: list[float] = []
    for k in k_grid:
        sample_drops = []
        for i in range(min(n, len(attrs))):
            order = np.argsort(np.abs(attrs[i]))   # least → most important
            bot_idx = order[:k]
            x_abl = X[i].copy()
            # Handle multidimensional arrays by flattening, indexing, and reshaping
            x_abl.flat[bot_idx] = baseline
            pred_abl = _predict_proba(model, x_abl[None])[0]
            sample_drops.append(float(np.abs(pred_orig[i] - pred_abl).mean()))
        drops_per_k.append(float(np.mean(sample_drops)) if sample_drops else 0.0)
    return {"k_grid": list(k_grid), "mean_drop": drops_per_k}


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
        return float("nan")
    if not _attrs_index_input_features(attrs, X):
        return float("nan")

    pred_orig = _predict_proba(model, X[:n])
    scores = []
    for i in range(min(n, len(attrs))):
        top_idx = np.argsort(np.abs(attrs[i]))[-top_k:]
        x_kept = np.full_like(X[i], baseline)
        # Handle multidimensional arrays by flattening, indexing, and reshaping
        x_kept.flat[top_idx] = X[i].flat[top_idx]
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
    normalize: bool = True,
) -> float:
    """
    Max L2 distance between explanation attributions under small input
    perturbations (Yeh et al. 2019).  Lower → more stable.

    When ``normalize=True`` (default) the per-sample distance is divided by
    the base attribution's L2 norm, giving a *scale-invariant* sensitivity
    score in roughly [0, 1+] that is comparable across (game, model, xai).
    Without normalization the raw L2 distance reflects attribution
    magnitudes — LRP on a high-confidence sudoku4 model produces O(10)
    raw distances, LRP on minesweeper-CNN O(0.1), making cross-config
    comparison meaningless.
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
    base_norms = np.linalg.norm(attrs_base, axis=1)

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

    if normalize:
        # Per-sample scale invariance: ||Δattr|| / ||attr||.
        # Guard against zero-norm rows (no attribution = trivially stable).
        safe_norms = np.where(base_norms > 1e-8, base_norms, 1.0)
        normalized = max_dists / safe_norms
        # If the base attribution itself was ~0, treat that sample as stable (0).
        normalized = np.where(base_norms > 1e-8, normalized, 0.0)
        return float(normalized.mean())
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
    # Save any fitted state on the explainer that explain() will overwrite
    _EXP_ATTRS = ("_tree", "_target_label", "_rules", "_feature_names")
    saved_exp = {k: copy.deepcopy(getattr(explainer, k)) for k in _EXP_ATTRS if hasattr(explainer, k)}

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
        for k, v in saved_exp.items():
            setattr(explainer, k, v)

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

    from ..games.sudoku import SudokuGame

    agreements = []
    for i in range(n):
        try:
            z3_sol = game.solve_symbolic(puzzles[i])
        except Exception:
            continue
        if z3_sol is None:
            continue

        if isinstance(game, SudokuGame):
            # z3_sol is a 2D digit grid (1..n); NN output is (n*n*n,) probabilities.
            # Decode each cell as argmax over its n-dim block, then compare digits.
            sz = game.size
            pred_digits = preds[i].reshape(-1, sz).argmax(axis=-1) + 1  # (n*n,) ints 1..n
            z3_digits = np.array(z3_sol, dtype=np.int32).reshape(-1)    # (n*n,) ints 1..n
            agreements.append(float((pred_digits == z3_digits).mean()))
        else:
            # Minesweeper: NN output is (n*n,) mine probabilities; z3_sol is binary mine grid.
            z3_flat = np.array(z3_sol, dtype=np.float32).reshape(-1)
            z3_bin = (z3_flat > 0.5).astype(int)
            pred_bin = (preds[i] > 0.5).astype(int)
            m = min(len(z3_bin), len(pred_bin))
            agreements.append(float((pred_bin[:m] == z3_bin[:m]).mean()))

    return float(np.mean(agreements)) if agreements else 0.0


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

# All metrics ON by default (Task P0-4).  The historical defaults left
# comprehensiveness / sufficiency off for everything except LIME/SHAP/LRP and
# kept semantic_equivalence_z3 off for everything except rule_extraction; this
# produced reports where 4/6 metric columns were stuck at 0 for non-LRP
# methods, which made the cross-method comparison tables in the paper read as
# "everything is zero except LRP".  Reviewers will (correctly) flag that.
#
# Methods that genuinely cannot compute a given metric (e.g. data_randomization
# requires a fitted decision tree — not present for attribution methods) handle
# that internally by returning NaN; the report writer maps NaN → null.
_DEFAULTS: dict[str, bool] = {
    "fidelity": True,
    "comprehensiveness": True,
    "sufficiency": True,
    "max_sensitivity": True,
    "model_randomization": True,
    "data_randomization": True,
    "semantic_equivalence_z3": True,
    "morf_curve": True,   # Task P1-12
    "lerf_curve": True,   # Task P1-12
}

# Per-method opt-OUTS only.  If you really want to disable a metric for a
# specific XAI method (e.g. concept_probe makes max_sensitivity expensive),
# set it to False here.  Don't put method-specific *opt-ins* here — defaults
# are now opt-out.
_PER_METHOD: dict[str, dict[str, bool]] = {
    # concept_probe's "attributions" field is the **hidden-activation matrix**,
    # not per-input-feature scores.  Feature-ablation metrics (comprehensiveness,
    # sufficiency, MoRF, LeRF) can't index input features from activation
    # columns — they'd land out of bounds and crash with IndexError.
    # max_sensitivity is also expensive (re-fits the probe per perturbation).
    # All four are explicitly disabled here; the metric functions themselves
    # also return NaN as a defence-in-depth guard if this is ever overridden.
    "concept_probe": {
        "comprehensiveness": False,
        "sufficiency": False,
        "max_sensitivity": False,
        "morf_curve": False,
        "lerf_curve": False,
    },
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
