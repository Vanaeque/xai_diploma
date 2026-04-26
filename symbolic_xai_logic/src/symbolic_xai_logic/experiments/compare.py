"""Multi-model / multi-XAI comparison tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..symbolic.fidelity import FidelityReport

_COLS_ORDER = [
    "game", "model", "xai", "seed",
    # Faithfulness
    "nn_accuracy", "nn_csr", "fidelity_to_nn",
    "comprehensiveness", "sufficiency",
    # Stability
    "max_sensitivity",
    # Sanity
    "model_randomization", "data_randomization",
    # Recoverability
    "semantic_equivalence_z3", "canonical_match_rate",
    # Plausibility
    "agreement_gt",
    # Comprehensibility
    "rule_count", "rule_complexity",
]

_METRIC_COLS = [
    "nn_accuracy", "nn_csr", "fidelity_to_nn",
    "comprehensiveness", "sufficiency",
    "max_sensitivity",
    "model_randomization", "data_randomization",
    "semantic_equivalence_z3", "canonical_match_rate",
    "agreement_gt",
    "rule_count", "rule_complexity",
]


def compare_experiments(reports: list[FidelityReport]) -> pd.DataFrame:
    """Convert list of FidelityReport to a per-run comparison DataFrame."""
    rows = [r.to_dict() for r in reports]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    available = [c for c in _COLS_ORDER if c in df.columns]
    sort_keys = [k for k in ("game", "model", "xai", "seed") if k in df.columns]
    return df[available].sort_values(sort_keys).reset_index(drop=True)


def aggregate_seeds(reports: list[FidelityReport]) -> pd.DataFrame:
    """Aggregate per-run reports across seeds: mean ± std for each metric."""
    if not reports:
        return pd.DataFrame()

    df = compare_experiments(reports)
    group_keys = [k for k in ("game", "model", "xai") if k in df.columns]
    metric_cols = [c for c in _METRIC_COLS if c in df.columns]

    rows = []
    for keys, grp in df.groupby(group_keys):
        row: dict = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        row["n_seeds"] = len(grp)
        for col in metric_cols:
            vals = grp[col].dropna().astype(float)
            row[f"{col}_mean"] = round(float(vals.mean()), 4) if len(vals) else float("nan")
            row[f"{col}_std"] = round(float(vals.std()), 4) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)
