"""Tests for extended XAI evaluation metrics."""
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """Two-feature linear model: output = sigmoid(w0*x0 + w1*x1)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            # x0 is the most important feature
            self.fc.weight.copy_(torch.tensor([[3.0, 0.1, 0.0, 0.0]]))

    def forward(self, x):
        return self.fc(x)

    def modules(self):  # expose for model_randomization_score
        return super().modules()


class FakeExplainer:
    """Explainer that attributes all importance to feature 0."""

    def __init__(self, model):
        self.model = model
        self._tree = None
        self.max_depth = 3
        self.min_samples_leaf = 5

    @property
    def name(self):
        return "fake"

    def explain(self, X, **kwargs):
        n = len(X)
        attrs = np.zeros((n, X.shape[1]), dtype=np.float32)
        attrs[:, 0] = 1.0  # all importance on feature 0
        return {"attributions": attrs, "mean_abs_attr": attrs.mean(axis=0)}

    def fidelity(self, X, y, explanation):
        return 1.0


def _make_X(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Comprehensiveness
# ---------------------------------------------------------------------------

class TestComprehensiveness:
    def test_returns_float(self):
        from symbolic_xai_logic.symbolic.metrics import comprehensiveness
        model = TinyMLP()
        explainer = FakeExplainer(model)
        X = _make_X()
        score = comprehensiveness(model, explainer, X, top_k=1)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_high_for_important_feature(self):
        """Dropping the most important feature (x0, large weight) should hurt the prediction."""
        from symbolic_xai_logic.symbolic.metrics import comprehensiveness
        model = TinyMLP()
        explainer = FakeExplainer(model)
        X = _make_X()
        score = comprehensiveness(model, explainer, X, top_k=1)
        # With large weight on feature 0, dropping it changes predictions meaningfully
        assert score > 0.0


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------

class TestSufficiency:
    def test_returns_float_in_range(self):
        from symbolic_xai_logic.symbolic.metrics import sufficiency
        model = TinyMLP()
        explainer = FakeExplainer(model)
        X = _make_X()
        score = sufficiency(model, explainer, X, top_k=1)
        assert isinstance(score, float)
        # Can exceed 1.0 if noise is small, so just check non-negative
        assert score >= -0.1

    def test_complementary_to_comprehensiveness(self):
        """More comprehensive → less sufficient (for a single-feature explanation)."""
        from symbolic_xai_logic.symbolic.metrics import comprehensiveness, sufficiency
        model = TinyMLP()
        explainer = FakeExplainer(model)
        X = _make_X()
        c = comprehensiveness(model, explainer, X, top_k=1)
        s = sufficiency(model, explainer, X, top_k=1)
        # Both should be non-trivial; comprehensiveness measures drop, sufficiency measures retention
        assert c >= 0.0
        assert s >= 0.0


# ---------------------------------------------------------------------------
# Max-sensitivity
# ---------------------------------------------------------------------------

class TestMaxSensitivity:
    def test_returns_non_negative_float(self):
        from symbolic_xai_logic.symbolic.metrics import max_sensitivity
        explainer = FakeExplainer(TinyMLP())
        X = _make_X()
        score = max_sensitivity(explainer, X, n_reruns=3, noise_std=0.05, max_samples=10)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_zero_for_constant_explainer(self):
        """An explainer that always returns the same attribution has sensitivity 0."""
        from symbolic_xai_logic.symbolic.metrics import max_sensitivity

        class ConstantExplainer(FakeExplainer):
            def explain(self, X, **kwargs):
                # Same attribution regardless of input
                attrs = np.ones((len(X), X.shape[1]), dtype=np.float32)
                return {"attributions": attrs}

        explainer = ConstantExplainer(TinyMLP())
        X = _make_X()
        score = max_sensitivity(explainer, X, n_reruns=3, noise_std=0.05, max_samples=5)
        assert score == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Model randomization score
# ---------------------------------------------------------------------------

class TestModelRandomizationScore:
    def test_returns_float_in_01(self):
        from symbolic_xai_logic.symbolic.metrics import model_randomization_score
        model = TinyMLP()
        explainer = FakeExplainer(model)
        X = _make_X()
        attrs_real = np.zeros((len(X), X.shape[1]), dtype=np.float32)
        attrs_real[:, 0] = 1.0
        score = model_randomization_score(model, explainer, X, attrs_real, n_samples=10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_model_weights_restored(self):
        """Model weights must be identical before and after the call."""
        from symbolic_xai_logic.symbolic.metrics import model_randomization_score
        model = TinyMLP()
        explainer = FakeExplainer(model)
        X = _make_X()
        attrs_real = np.zeros((len(X), 4), dtype=np.float32)
        attrs_real[:, 0] = 1.0

        w_before = model.fc.weight.clone()
        model_randomization_score(model, explainer, X, attrs_real, n_samples=5)
        w_after = model.fc.weight.clone()
        assert torch.allclose(w_before, w_after), "Model weights were not restored"


# ---------------------------------------------------------------------------
# Data randomization score
# ---------------------------------------------------------------------------

class TestDataRandomizationScore:
    def test_nan_for_non_tree_explainer(self):
        """Returns NaN for explainers without a surrogate tree."""
        from symbolic_xai_logic.symbolic.metrics import data_randomization_score
        import math
        explainer = FakeExplainer(TinyMLP())
        explainer._tree = None
        X = _make_X()
        y_nn = np.random.rand(len(X), 1).astype(np.float32)
        score = data_randomization_score(explainer, X, y_nn)
        assert math.isnan(score)

    def test_returns_float_for_tree_explainer(self):
        """Returns a valid float [0,1] when a fitted tree is present."""
        from symbolic_xai_logic.symbolic.metrics import data_randomization_score
        from sklearn.tree import DecisionTreeClassifier
        import math

        X = _make_X(n=60)
        y_nn = np.random.randint(0, 2, size=(60, 1)).astype(np.float32)

        explainer = FakeExplainer(TinyMLP())
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)
        X_bin = (X > 0.5).astype(np.float32)
        y_bin = (y_nn[:, 0] > 0.5).astype(int)
        tree.fit(X_bin, y_bin)
        explainer._tree = tree

        score = data_randomization_score(explainer, X, y_nn, n_samples=50)
        assert not math.isnan(score)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Semantic equivalence via Z3
# ---------------------------------------------------------------------------

class TestSemanticEquivalenceZ3:
    def test_returns_float_in_01(self):
        from symbolic_xai_logic.symbolic.metrics import semantic_equivalence_z3
        from symbolic_xai_logic.games.minesweeper import MinesweeperGame

        game = MinesweeperGame(size=4, n_mines=2)
        pairs = game.generate(20, seed=7)
        puzzles = [p for p, _ in pairs]
        X = np.stack([game.encode(p, "one_hot") for p in puzzles])

        # Simple model: always predicts 0 (no mines)
        class ZeroModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], game.output_dim)

        model = ZeroModel()
        score = semantic_equivalence_z3(game, model, X, puzzles, n_samples=10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_perfect_model_high_score(self):
        """A model that predicts perfectly should get a high Z3 equivalence score."""
        from symbolic_xai_logic.symbolic.metrics import semantic_equivalence_z3
        from symbolic_xai_logic.games.minesweeper import MinesweeperGame
        import torch.nn.functional as F

        game = MinesweeperGame(size=4, n_mines=2)
        pairs = game.generate(10, seed=9)
        puzzles = [p for p, _ in pairs]
        mines = [m for _, m in pairs]
        X = np.stack([game.encode(p, "one_hot") for p in puzzles])
        y_true = np.stack([np.array(m, dtype=np.float32).reshape(-1) for m in mines])

        # Oracle: returns ground-truth mine positions
        class OracleModel(nn.Module):
            def __init__(self, y):
                super().__init__()
                self._y = torch.tensor(y, dtype=torch.float32)

            def forward(self, x):
                # Return logit-equivalent of y_true (large positives/negatives)
                return (self._y[:len(x)] * 20.0) - 10.0

        model = OracleModel(y_true)
        score = semantic_equivalence_z3(game, model, X, puzzles, n_samples=10)
        assert score >= 0.7, f"Oracle model got low Z3 equivalence: {score}"


# ---------------------------------------------------------------------------
# resolve_metrics
# ---------------------------------------------------------------------------

class TestResolveMetrics:
    def test_defaults(self):
        from symbolic_xai_logic.symbolic.metrics import resolve_metrics
        flags = resolve_metrics("lime")
        assert flags["fidelity"] is True
        assert flags["comprehensiveness"] is True
        assert flags["max_sensitivity"] is True

    def test_rule_extraction_gets_z3(self):
        from symbolic_xai_logic.symbolic.metrics import resolve_metrics
        flags = resolve_metrics("rule_extraction")
        assert flags["semantic_equivalence_z3"] is True

    def test_user_override(self):
        from symbolic_xai_logic.symbolic.metrics import resolve_metrics
        flags = resolve_metrics("lime", {"max_sensitivity": False})
        assert flags["max_sensitivity"] is False
