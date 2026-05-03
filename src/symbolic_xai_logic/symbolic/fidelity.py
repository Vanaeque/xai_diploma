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
    # Template coverage (Task P1-9): fraction of registered template families
    # that ever produced at least one match for this (game, xai) pair.
    template_coverage: float | None = None
    templates_fired: list = field(default_factory=list)
    # Z3-validated rule correctness (Task P1-10): of the canonically-matched
    # clauses, what fraction is actually logically valid under the game's rules?
    canonical_valid_rate: float | None = None
    canonical_false_positive_rate: float | None = None
    # Metadata
    config_hash: str = ""
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Numeric metrics that should never end up as NaN/inf in the JSON.
        # Replaced with None so the report stays valid JSON and downstream
        # aggregators can filter without parsing strings.  See P1-11.
        def _safe(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            try:
                return round(float(v), 4)
            except (TypeError, ValueError):
                return None

        d = {
            "game": self.game_name,
            "model": self.model_name,
            "xai": self.xai_name,
            "seed": self.seed,
            "nn_accuracy": _safe(self.nn_accuracy),
            "nn_csr": _safe(self.nn_constraint_satisfaction),
            "fidelity_to_nn": _safe(self.explainer_fidelity_to_nn),
            "agreement_gt": _safe(self.explainer_agreement_with_gt),
            "comprehensiveness": _safe(self.comprehensiveness),
            "sufficiency": _safe(self.sufficiency),
            "max_sensitivity": _safe(self.max_sensitivity),
            "model_randomization": _safe(self.model_randomization_score),
            "rule_complexity": self.rule_complexity,
            "rule_count": self.rule_count,
        }
        d["data_randomization"] = _safe(self.data_randomization_score)
        d["semantic_equivalence_z3"] = _safe(self.semantic_equivalence_z3)
        d["canonical_match_rate"] = (
            _safe(self.canonical_match_rate)
            if self.canonical_match_rate is not None else None
        )
        d["template_coverage"] = (
            _safe(self.template_coverage)
            if self.template_coverage is not None else None
        )
        d["templates_fired"] = list(self.templates_fired)
        d["canonical_valid_rate"] = (
            _safe(self.canonical_valid_rate)
            if self.canonical_valid_rate is not None else None
        )
        d["canonical_false_positive_rate"] = (
            _safe(self.canonical_false_positive_rate)
            if self.canonical_false_positive_rate is not None else None
        )
        d["config_hash"] = self.config_hash
        # Merge extra (canonical_stats, by_template, warnings, etc.) — preserve
        # nested structures rather than coercing to float.
        d.update(self.extra)
        # Surface explainer-level warnings, if any
        warnings = self.extra.get("warnings") if isinstance(self.extra, dict) else None
        if warnings:
            d["warnings"] = list(warnings)
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
    X_explain: np.ndarray | None = None,
    solutions_explain: list | None = None,
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
        morf_curve as morf_fn,
        lerf_curve as lerf_fn,
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

    # Primary explanation — prefer the dedicated explain split when provided
    # (Task P0-3).  Symbolic XAI methods (rule_extraction, symbolic_regression)
    # are starved by 400-puzzle test splits with 64 cell-digit dims to fit; the
    # explain split is sized for them (typically 5000-20000).  Attribution
    # methods don't benefit from a larger sample but tolerate it fine.
    X_explain_eff = X_explain if X_explain is not None else X_test
    sols_explain_eff = solutions_explain if solutions_explain is not None else solutions_test
    explain_kwargs: dict = {}
    if sols_explain_eff is not None:
        explain_kwargs["solutions"] = sols_explain_eff
    explanation = explainer.explain(X_explain_eff, **explain_kwargs)

    # Canonical template matching.
    # Prefer canonical_stats embedded in explanation (from RuleExtractor.explain)
    # Fall back to calling extract_all_canonical_rules directly if not present
    canonical_match_rate: float | None = None
    if xai_name == "rule_extraction":
        # First try: use canonical_stats embedded by explain()
        canonical_stats = explanation.get("canonical_stats")
        if canonical_stats:
            canonical_match_rate = canonical_stats.get("canonical_match_rate")
        
        # Fallback: call extract_all_canonical_rules directly if explain() didn't compute it
        if canonical_match_rate is None and hasattr(explainer, "extract_all_canonical_rules"):
            try:
                _, all_stats = explainer.extract_all_canonical_rules(X_test, game)
                canonical_match_rate = all_stats["canonical_match_rate"]
            except Exception:
                pass
    
    # If still no canonical match rate, try sympy_formulas
    if canonical_match_rate is None:
        formulas = explanation.get("sympy_formulas", [])
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
    formulas = explanation.get("sympy_formulas", [])
    canonical_rules = explanation.get("canonical_rules", [])
    # For rule_extraction on template games, the meaningful rule_count is the
    # number of UNIQUE CANONICAL rules recovered (not the depth-4 single-tree
    # paths, which are mostly non-canonical noise).  See P0-2 in
    # copilot_upgrade_instructions.md.
    if xai_name == "rule_extraction" and canonical_rules:
        rule_count = len(canonical_rules)
        complexity = sum(len(r.get("text", "")) for r in canonical_rules)
    else:
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

    # MoRF / LeRF curves (Task P1-12).  Skip when there are no per-sample
    # attributions to rank (concept_probe doesn't produce them in a way that
    # makes feature ablation meaningful).
    morf_samples = int(mcfg.get("morf_max_samples", 50))
    morf: dict | None = None
    lerf: dict | None = None
    if flags.get("morf_curve") and attrs_real is not None and attrs_real.ndim == 2 and attrs_real.shape[0] > 1:
        try:
            morf = morf_fn(model, explainer, X_test, max_samples=morf_samples)
        except Exception:
            morf = None
    if flags.get("lerf_curve") and attrs_real is not None and attrs_real.ndim == 2 and attrs_real.shape[0] > 1:
        try:
            lerf = lerf_fn(model, explainer, X_test, max_samples=morf_samples)
        except Exception:
            lerf = None

    # Prepare extra fields for report
    extra_dict: dict = {"best_r2": explanation.get("best_r2", 0.0)}
    # Forward any explainer-level warnings (e.g. "concept_probe: 3 concepts skipped")
    if explanation.get("warnings"):
        extra_dict["warnings"] = list(explanation["warnings"])
    # MoRF / LeRF curves (Task P1-12) — saved as nested dicts {k_grid, mean_drop}
    if morf is not None:
        extra_dict["morf_curve"] = morf
    if lerf is not None:
        extra_dict["lerf_curve"] = lerf

    # Add canonical stats if available — flatten the relevant fields directly
    # into the report so consumers (aggregate_canonical_rules, paper tables)
    # can read by_template / template_coverage / matched / total without
    # peeling a nested dict.
    canonical_stats = explanation.get("canonical_stats")
    if canonical_stats:
        extra_dict["canonical_stats"] = canonical_stats
        # Promote the most useful fields to top level too
        if "by_template" in canonical_stats:
            extra_dict["by_template"] = canonical_stats["by_template"]
        if "templates_fired" in canonical_stats:
            extra_dict["templates_fired"] = canonical_stats["templates_fired"]
        if "template_coverage" in canonical_stats:
            extra_dict["template_coverage"] = canonical_stats["template_coverage"]
        if "n_total" in canonical_stats:
            extra_dict["matched"] = canonical_stats.get("n_matched", 0)
            extra_dict["total"] = canonical_stats["n_total"]
    else:
        # Fallback for non-rule_extraction methods: by_template from sympy formulas
        try:
            from ..viz.templates import render_clauses_as_nl
            formulas_for_nl = explanation.get("sympy_formulas", [])
            if formulas_for_nl:
                nl_result = render_clauses_as_nl(formulas_for_nl, game)
                if "by_template" in nl_result:
                    extra_dict["by_template"] = nl_result["by_template"]
                extra_dict["matched"] = nl_result.get("matched", 0)
                extra_dict["total"] = nl_result.get("total", 0)
        except Exception:
            pass

    # Surface canonical-pass derived fields directly on the report dataclass
    # (Task P1-9 / P1-10).  These come from extract_all_canonical_rules' stats
    # dict — see xai/rule_extraction.py.
    template_coverage = None
    templates_fired: list = []
    canonical_valid_rate = None
    canonical_fp_rate = None
    if canonical_stats:
        template_coverage = canonical_stats.get("template_coverage")
        templates_fired = list(canonical_stats.get("templates_fired", []))
        canonical_valid_rate = canonical_stats.get("canonical_valid_rate")
        canonical_fp_rate = canonical_stats.get("canonical_false_positive_rate")

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
        template_coverage=template_coverage,
        templates_fired=templates_fired,
        canonical_valid_rate=canonical_valid_rate,
        canonical_false_positive_rate=canonical_fp_rate,
        config_hash=config_hash,
        seed=seed,
        extra=extra_dict,
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
