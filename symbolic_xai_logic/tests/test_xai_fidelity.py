"""
XAI fidelity tests: explainers recover known-planted rules.

Planted rule test: a synthetic dataset has a known boolean rule
(e.g., output=1 iff feature_0 > 0.5 AND feature_1 > 0.5),
and the rule extractor must recover it with ≥ 0.95 fidelity.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from symbolic_xai_logic.xai.rule_extraction import RuleExtractor
from symbolic_xai_logic.xai.lrp import LRPExplainer
from symbolic_xai_logic.xai.concept_probe import ConceptProbe
from symbolic_xai_logic.xai.symbolic_regression import SymbolicRegressionExplainer
from symbolic_xai_logic.models.mlp import MLP
from symbolic_xai_logic.games.sudoku import SudokuGame


class PlantedRuleModel(nn.Module):
    """A model that perfectly implements: y = (x[0] > 0.5) AND (x[1] > 0.5)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = (x[:, 0:1] - 0.5) * 20
        f1 = (x[:, 1:2] - 0.5) * 20
        return torch.min(f0, f1)

    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        return x


class DummyGame:
    """Minimal game shim for testing."""
    name = "synthetic"

    def concepts(self, state):
        return {}

    def is_valid(self, state):
        return True


def make_planted_dataset(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 10).astype(np.float32)
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(np.float32)
    return X, y


class TestPlantedRuleFidelity:
    """Core planted-rule test: rule extractor must recover ≥ 0.95 fidelity."""

    def test_rule_extraction_fidelity_planted(self):
        X, y = make_planted_dataset(n=2000, seed=42)
        model = PlantedRuleModel()
        game = DummyGame()

        extractor = RuleExtractor(model=model, game=game, max_depth=4, min_samples_leaf=5, n_samples=2000)
        explanation = extractor.explain(X, feature_names=[f"x_{i}" for i in range(X.shape[1])])
        fidelity = extractor.fidelity(X, y, explanation)

        assert fidelity >= 0.95, (
            f"Rule extraction fidelity {fidelity:.3f} is below required 0.95 for planted rule dataset"
        )

    def test_rule_extraction_recovers_top_features(self):
        """Top-2 most important features should be x_0 and x_1."""
        X, y = make_planted_dataset(n=2000, seed=42)
        model = PlantedRuleModel()
        game = DummyGame()

        extractor = RuleExtractor(model=model, game=game, max_depth=4, min_samples_leaf=5)
        explanation = extractor.explain(X, feature_names=[f"x_{i}" for i in range(X.shape[1])])

        importances = explanation["feature_importances"]
        top2 = set(np.argsort(importances)[-2:])
        assert 0 in top2, f"Feature x_0 not in top-2: importances={importances[:5]}"
        assert 1 in top2, f"Feature x_1 not in top-2: importances={importances[:5]}"


class TestLRPExplainer:
    def test_lrp_produces_attributions(self):
        model = MLP(input_dim=16, output_dim=4, hidden_dims=[32, 16])
        game = DummyGame()
        game.name = "synthetic"

        X = np.random.rand(10, 16).astype(np.float32)
        explainer = LRPExplainer(model=model, game=game)
        explanation = explainer.explain(X)

        assert "attributions" in explanation
        assert explanation["attributions"].shape[0] == 10
        assert explanation["attributions"].shape[1] == 16

    def test_lrp_fidelity_range(self):
        model = MLP(input_dim=16, output_dim=4, hidden_dims=[32, 16])
        game = DummyGame()
        game.name = "synthetic"

        X = np.random.rand(20, 16).astype(np.float32)
        y = np.random.rand(20, 4).astype(np.float32)
        explainer = LRPExplainer(model=model, game=game)
        explanation = explainer.explain(X)
        fidelity = explainer.fidelity(X, y, explanation)

        assert -1.0 <= fidelity <= 1.0


class TestSymbolicRegression:
    def test_sr_runs(self):
        model = PlantedRuleModel()
        game = DummyGame()
        game.name = "synthetic"

        X, y = make_planted_dataset(n=500)
        explainer = SymbolicRegressionExplainer(model=model, game=game, n_samples=500)
        explanation = explainer.explain(X)

        assert "expressions" in explanation
        assert "best_r2" in explanation
        assert explanation["best_r2"] >= 0.0

    def test_sr_fidelity_planted(self):
        model = PlantedRuleModel()
        game = DummyGame()
        game.name = "synthetic"

        X, y = make_planted_dataset(n=500)
        explainer = SymbolicRegressionExplainer(model=model, game=game, n_samples=500)
        explanation = explainer.explain(X)
        fidelity = explainer.fidelity(X, y, explanation)

        assert fidelity >= 0.5, f"SR fidelity {fidelity:.3f} too low for planted rule"


class TestConceptProbe:
    def test_concept_probe_runs(self):
        model = MLP(input_dim=64, output_dim=16, hidden_dims=[32])
        game = SudokuGame(size=4)
        X = np.random.rand(50, 64).astype(np.float32)
        pairs = game.generate(50, seed=0)
        solutions = [sol for _, sol in pairs]

        explainer = ConceptProbe(model=model, game=game)
        explanation = explainer.explain(X, solutions=solutions)

        assert "concept_scores" in explanation
        assert len(explanation["concept_scores"]) > 0

    def test_concept_probe_fidelity_range(self):
        model = MLP(input_dim=64, output_dim=16, hidden_dims=[32])
        game = SudokuGame(size=4)
        X = np.random.rand(50, 64).astype(np.float32)
        y = np.random.rand(50, 64).astype(np.float32)
        pairs = game.generate(50, seed=0)
        solutions = [sol for _, sol in pairs]

        explainer = ConceptProbe(model=model, game=game)
        explanation = explainer.explain(X, solutions=solutions)
        fidelity = explainer.fidelity(X, y, explanation)

        assert 0.0 <= fidelity <= 1.0
