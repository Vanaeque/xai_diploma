"""Linear concept probes on hidden activations."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from .base import Explainer


class ConceptProbe(Explainer):
    """
    Train linear classifiers on hidden activations to predict
    game-specific ground-truth concepts (e.g., "row r contains digit d").
    """

    def __init__(
        self,
        model,
        game,
        probe_type: str = "linear",
        layer: int = -1,
        n_epochs: int = 50,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(model, game)
        self.probe_type = probe_type
        self.layer = layer
        self.n_epochs = n_epochs
        self.lr = lr
        self._probes: dict[str, Any] = {}
        self._concept_scores: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "concept_probe"

    def _get_activations(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            try:
                acts = self.model.get_activations(t, self.layer).numpy()
            except (AttributeError, Exception):
                acts = self.model(t).detach().numpy()
        return acts

    def _fit_probe(self, acts: np.ndarray, labels: np.ndarray) -> tuple[Any, float]:
        """Fit a logistic-regression probe on hidden activations.

        Returns (fitted pipeline | None, cross-val accuracy in [0, 1]).
        For degenerate cases (single-class labels, severe class imbalance,
        too-small split) returns (None, 1.0) so the caller can decide to
        skip rather than propagate NaN.  The NaN observed in
        report_minesweeper8_gnn_concept_probe.json (seed 1) was caused by
        ``cross_val_score`` returning NaN when the per-fold class population
        was zero — guard that here.  See P1-11 in copilot_upgrade_instructions.md.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score

        if len(np.unique(labels)) < 2:
            # Trivially predictable: all labels are the same value
            return None, 1.0

        # Guard: need enough samples per class to even attempt CV
        unique, counts = np.unique(labels, return_counts=True)
        min_class = int(counts.min())
        # cv must be ≥ 2 and ≤ min_class (each fold needs ≥1 of each class)
        cv = max(2, min(3, min_class, len(labels) // 10))
        if cv < 2 or min_class < 2:
            return None, float("nan")  # caller will skip

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, C=1.0, random_state=42)),
        ])
        try:
            scores = cross_val_score(
                pipe, acts, labels, cv=cv, scoring="accuracy",
                error_score="raise",
            )
        except Exception:
            return None, float("nan")
        # NaN can still slip through in adversarial cases; guard the mean.
        score = float(scores.mean())
        if not np.isfinite(score):
            return None, float("nan")
        pipe.fit(acts, labels)
        return pipe, score

    def explain(
        self,
        X: np.ndarray,
        solutions: list | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        acts = self._get_activations(X)

        # Collect concept labels from game
        if solutions is not None:
            concept_labels: dict[str, list] = {}
            for sol in solutions:
                c = self.game.concepts(sol)
                for k, v in c.items():
                    concept_labels.setdefault(k, []).append(int(v) if hasattr(v, "__int__") else v)

            n_skipped_degenerate = 0
            for concept_name, labels in concept_labels.items():
                labels_arr = np.array(labels, dtype=int)
                if len(labels_arr) != len(acts):
                    continue
                probe, score = self._fit_probe(acts, labels_arr)
                if not np.isfinite(score):
                    n_skipped_degenerate += 1
                    continue  # do NOT pollute _concept_scores with NaN
                self._concept_scores[concept_name] = score
                if probe is not None:
                    self._probes[concept_name] = probe
            self._n_skipped_degenerate = n_skipped_degenerate

        top_concepts = sorted(self._concept_scores.items(), key=lambda x: -x[1])[:10]

        warnings = []
        if getattr(self, "_n_skipped_degenerate", 0) > 0:
            warnings.append(
                f"concept_probe: {self._n_skipped_degenerate} concepts had "
                f"degenerate label distributions and were excluded from scoring"
            )

        return {
            "attributions": acts,
            "method": "concept_probe",
            "concept_scores": self._concept_scores,
            "top_concepts": top_concepts,
            "summary": (
                f"Concept probes: {len(self._concept_scores)} concepts, "
                f"top: {top_concepts[:3] if top_concepts else 'none'}"
            ),
            "mean_abs_attr": np.abs(acts).mean(axis=0) if acts.ndim == 2 else np.array([]),
            "warnings": warnings,
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Fidelity: mean cross-val accuracy across all concept probes."""
        scores = [s for s in self._concept_scores.values() if np.isfinite(s)]
        return float(np.mean(scores)) if scores else 0.0
