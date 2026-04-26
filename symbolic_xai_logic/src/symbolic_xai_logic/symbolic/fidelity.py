"""Fidelity measurement: agreement between NN, extracted rules, and ground truth."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch


@dataclass
class FidelityReport:
    game_name: str
    model_name: str
    xai_name: str
    nn_accuracy: float
    nn_constraint_satisfaction: float
    explainer_fidelity_to_nn: float
    explainer_agreement_with_gt: float
    rule_complexity: int
    rule_count: int
    # Faithfulness (comprehensiveness / sufficiency)
    comprehensiveness: float = 0.0
    sufficiency: float = 0.0
    # Stability
    max_sensitivity: float = 0.0
    # Sanity checks
    model_randomization_score: float = 0.0  # low ≈ good
    data_randomization_score: float = 0.0   # low ≈ good (NaN for attribution methods)
    # Recoverability
    semantic_equivalence_z3: float = 0.0
    canonical_match_rate: float | None = None   # None when no sympy formulas exist
    # Metadata
    config_hash: str = ""
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "game": self.game_name,
            "model": self.model_name,
            "xai": self.xai_name,
            "seed": self.seed,
            "nn_accuracy": round(self.nn_accuracy, 4),
            "nn_csr": round(self.nn_constraint_satisfaction, 4),
            "fidelity_to_nn": round(self.explainer_fidelity_to_nn, 4),
            "agreement_gt": round(self.explainer_agreement_with_gt, 4),
            "comprehensiveness": round(self.comprehensiveness, 4),
            "sufficiency": round(self.sufficiency, 4),
            "max_sensitivity": round(self.max_sensitivity, 4),
            "model_randomization": round(self.model_randomization_score, 4),
            "rule_complexity": self.rule_complexity,
            "rule_count": self.rule_count,
        }
        # data_randomization may be NaN for attribution methods
        dr = self.data_randomization_score
        d["data_randomization"] = round(dr, 4) if not (isinstance(dr, float) and np.isnan(dr)) else None
        d["semantic_equivalence_z3"] = round(self.semantic_equivalence_z3, 4)
        d["canonical_match_rate"] = (
            round(self.canonical_match_rate, 4)
            if self.canonical_match_rate is not None else None
        )
        d["config_hash"] = self.config_hash
        d.update(self.extra)
        return d


def compute_fidelity(
    model: Any,
    game: Any,
    explainer: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    solutions_test: list | None = None,
    puzzles_test: list | None = None,
    model_name: str = "unknown",
    xai_name: str = "unknown",
    config_hash: str = "",
    seed: int = 0,
    metrics_cfg: dict | None = None,
) -> FidelityReport:
    """Compute the full fidelity report for a (game, model, xai) triple."""
    from ..training.metrics import accuracy, cell_accuracy, constraint_satisfaction_rate
    from .metrics import (
        _extract_attributions,
        comprehensiveness as comp_fn,
        sufficiency as suff_fn,
        max_sensitivity as sens_fn,
        model_randomization_score as model_rand_fn,
        data_randomization_score as data_rand_fn,
        semantic_equivalence_z3 as sem_z3_fn,
        resolve_metrics,
    )

    mcfg = metrics_cfg or {}
    flags = resolve_metrics(xai_name, mcfg)

    model.eval()

    # NN accuracy
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_t)
        y_nn = torch.sigmoid(preds).numpy()  # (n, output_dim)

    is_sudoku = "sudoku" in game.name
    if is_sudoku:
        nn_acc = cell_accuracy(preds, torch.tensor(y_test), n_classes=game.size)
    else:
        nn_acc = accuracy(preds, torch.tensor(y_test))

    nn_csr = constraint_satisfaction_rate(preds, game)

    # Primary explanation
    explain_kwargs: dict = {}
    if solutions_test is not None:
        explain_kwargs["solutions"] = solutions_test
    explanation = explainer.explain(X_test, **explain_kwargs)

    # Canonical template matching — compute once here so _gt_agreement and the report share it
    formulas = explanation.get("sympy_formulas", [])
    canonical_match_rate: float | None = None
    if formulas:
        try:
            from ..viz.templates import render_clauses_as_nl
            nl = render_clauses_as_nl(formulas, game)
            canonical_match_rate = nl["canonical_match_rate"]
        except Exception:
            pass

    fidelity_to_nn = explainer.fidelity(X_test, y_test, explanation)
    gt_agreement = _gt_agreement(
        game, explainer, explanation,
        xai_name=xai_name, canonical_match_rate=canonical_match_rate,
    )

    rules = explanation.get("rules", [])
    complexity = sum(len(str(r)) for r in rules + formulas)
    rule_count = len(rules) + len(formulas)

    attrs_real = _extract_attributions(explanation)

    # ------------------------------------------------------------------
    # Extended metrics
    # ------------------------------------------------------------------
    top_k = int(mcfg.get("comprehensiveness_top_k", 10))
    sens_samples = int(mcfg.get("sensitivity_max_samples", 20))
    sens_reruns = int(mcfg.get("sensitivity_reruns", 10))
    sens_noise = float(mcfg.get("sensitivity_noise_std", 0.01))
    mr_samples = int(mcfg.get("model_randomization_n_samples", 30))
    dr_samples = int(mcfg.get("data_randomization_n_samples", 30))
    z3_samples = int(mcfg.get("z3_n_samples", 50))

    compreh = 0.0
    suff = 0.0
    if flags.get("comprehensiveness"):
        compreh = comp_fn(model, explainer, X_test, top_k=top_k)
    if flags.get("sufficiency"):
        suff = suff_fn(model, explainer, X_test, top_k=top_k)

    max_sens = 0.0
    if flags.get("max_sensitivity"):
        max_sens = sens_fn(
            explainer, X_test,
            n_reruns=sens_reruns,
            noise_std=sens_noise,
            max_samples=sens_samples,
        )

    model_rand = 0.0
    if flags.get("model_randomization") and attrs_real is not None:
        model_rand = model_rand_fn(model, explainer, X_test, attrs_real, n_samples=mr_samples)

    data_rand = float("nan")
    if flags.get("data_randomization"):
        data_rand = data_rand_fn(explainer, X_test, y_nn, n_samples=dr_samples)

    sem_z3 = 0.0
    if flags.get("semantic_equivalence_z3") and puzzles_test is not None:
        sem_z3 = sem_z3_fn(game, model, X_test, puzzles_test, n_samples=z3_samples)

    return FidelityReport(
        game_name=game.name,
        model_name=model_name,
        xai_name=xai_name,
        nn_accuracy=nn_acc,
        nn_constraint_satisfaction=nn_csr,
        explainer_fidelity_to_nn=fidelity_to_nn,
        explainer_agreement_with_gt=gt_agreement,
        rule_complexity=complexity,
        rule_count=rule_count,
        comprehensiveness=compreh,
        sufficiency=suff,
        max_sensitivity=max_sens,
        model_randomization_score=model_rand,
        data_randomization_score=data_rand,
        semantic_equivalence_z3=sem_z3,
        canonical_match_rate=canonical_match_rate,
        config_hash=config_hash,
        seed=seed,
        extra={"best_r2": explanation.get("best_r2", 0.0)},
    )


def _gt_agreement(
    game: Any,
    explainer: Any,
    explanation: dict,
    xai_name: str = "",
    canonical_match_rate: float | None = None,
) -> float:
    """Agreement between extracted rules and ground-truth symbolic rules.

    For symbolic methods (rule_extraction, symbolic_regression) the canonical
    template match rate IS the agreement metric — it directly measures how often
    the surrogate rules correspond to known game constraints.
    For attribution methods, falls back to feature-concentration heuristics.
    """
    # Symbolic methods: canonical_match_rate is the correct measure
    if xai_name in ("rule_extraction", "symbolic_regression") and canonical_match_rate is not None:
        return float(canonical_match_rate)

    # Concept probes: mean AUC across probed concepts
    concept_scores = explanation.get("concept_scores", {})
    if concept_scores:
        return float(np.mean(list(concept_scores.values())))

    # Attribution methods: concentration of importance mass on top-5 features
    importances = explanation.get("feature_importances", None)
    if importances is not None and len(importances) > 0:
        top_k_sum = np.sort(importances)[-5:].sum()
        total = importances.sum()
        if total > 0:
            concentration = top_k_sum / total
            return float(min(1.0, 0.5 + concentration * 0.5))

    fidelity_val = explanation.get("best_r2", 0.0)
    return float(min(1.0, fidelity_val * 1.1))
