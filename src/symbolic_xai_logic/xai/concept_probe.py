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
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score

        if len(np.unique(labels)) < 2:
            # Trivially predictable: all labels are the same value
            return None, 1.0

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, C=1.0, random_state=42)),
        ])
        scores = cross_val_score(pipe, acts, labels, cv=min(3, len(labels) // 10), scoring="accuracy")
        pipe.fit(acts, labels)
        return pipe, float(scores.mean())

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

            for concept_name, labels in concept_labels.items():
                labels_arr = np.array(labels, dtype=int)
                if len(labels_arr) != len(acts):
                    continue
                probe, score = self._fit_probe(acts, labels_arr)
                self._concept_scores[concept_name] = score
                if probe is not None:
                    self._probes[concept_name] = probe

        top_concepts = sorted(self._concept_scores.items(), key=lambda x: -x[1])[:10]

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
        }

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Fidelity: mean cross-val accuracy across all concept probes."""
        scores = list(self._concept_scores.values())
        return float(np.mean(scores)) if scores else 0.0
