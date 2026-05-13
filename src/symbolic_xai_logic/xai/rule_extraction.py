"""Decision-tree / rule-list rule extraction from NN outputs."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from sympy import Symbol, And, Not
from .base import Explainer


class RuleExtractor(Explainer):
    """
    Train a surrogate decision tree on NN predictions,
    then convert to sympy logical formulas.

    One-hot features are binarized (threshold forced to 0.5) before fitting
    so that tree splits are always clean Boolean conditions and the DNF
    converter never emits contradictory literals like f ∧ ¬f.
    """

    def __init__(
        self,
        model,
        game,
        method: str = "decision_tree",
        max_depth: int = 4,
        min_samples_leaf: int = 30,
        n_samples: int = 5000,
        **kwargs,
    ):
        super().__init__(model, game)
        self.method = method
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_samples = n_samples
        self._tree = None
        self._feature_names: list[str] = []
        self._rules: list[str] = []
        self._target_label: str = ""   # human-readable description of what is being classified

    @property
    def name(self) -> str:
        return "rule_extraction"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nn_predictions(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            out = torch.sigmoid(self.model(t)).numpy()
        if out.shape[1] > 1:
            return (out > 0.5).astype(int)
        return (out[:, 0] > 0.5).astype(int)

    @staticmethod
    def _binarize(X: np.ndarray) -> np.ndarray:
        """
        Force one-hot / near-binary features to strict {0, 1}.
        Any value > 0.5 → 1, else → 0.
        This means the only meaningful decision-tree split is at 0.5,
        preventing multi-threshold paths that produce contradictions in DNF.
        """
        return (X > 0.5).astype(np.float32)

    def _pick_target_dim(self, y_nn: np.ndarray, X_bin: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """
        Choose the most informative output dimension as the binary target.
        For Sudoku, prefers dims where the corresponding cell is blank in most puzzles,
        focusing XAI on the interesting predictions rather than trivial given-digit copying.
        Falls back to global variance if no blank dim has positive variance.
        """
        if y_nn.ndim == 1:
            return y_nn, 0
        var = y_nn.var(axis=0)

        from ..games.sudoku import SudokuGame
        if isinstance(self.game, SudokuGame) and X_bin is not None:
            n = self.game.size
            blank_var = np.zeros_like(var)
            for d in range(len(var)):
                cell = d // n
                cell_feats = X_bin[:, cell * n : cell * n + n]
                if cell_feats.sum(axis=1).mean() < 0.5:  # usually blank
                    blank_var[d] = var[d]
            if blank_var.sum() > 0:
                candidates = np.where(blank_var > 0)[0]
                best = int(np.random.choice(candidates))
                return y_nn[:, best], best

        best_dim = int(var.argmax()) if var.sum() > 0 else 0
        return y_nn[:, best_dim], best_dim

    def _dim_to_label(self, dim: int) -> str:
        """Map an output-dimension index to a human-readable target description."""
        from ..games.sudoku import SudokuGame
        from ..games.minesweeper import MinesweeperGame

        if isinstance(self.game, SudokuGame):
            n = self.game.size
            cell = dim // n
            digit = (dim % n) + 1
            row, col = divmod(cell, n)
            return f"cell ({row},{col}) = digit {digit}"
        if isinstance(self.game, MinesweeperGame):
            n = self.game.size
            row, col = divmod(dim, n)
            return f"mine at cell ({row},{col})"
        return f"output dim {dim}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> None:
        from sklearn.tree import DecisionTreeClassifier

        X_sub = X[:min(self.n_samples, len(X))]
        # Binarize BEFORE fitting so splits are always at 0.5
        X_bin = self._binarize(X_sub)

        y_nn = self._get_nn_predictions(X_sub)
        y_target, best_dim = self._pick_target_dim(y_nn, X_bin)
        self._target_label = self._dim_to_label(best_dim)

        self._feature_names = feature_names or [f"f_{i}" for i in range(X_bin.shape[1])]
        self._tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self._tree.fit(X_bin, y_target)
        self._rules = self._extract_rules()

    def _extract_rules(self) -> list[str]:
        """Format the decision tree as a readable text block with a target header."""
        from sklearn.tree import export_text
        if self._tree is None:
            return []
        header = f"Target: {self._target_label}\n"
        body = export_text(
            self._tree,
            feature_names=self._feature_names[:self._tree.n_features_in_],
        )
        return [header + body]

    def to_sympy(self) -> list[Any]:
        """
        Convert positive-class leaf paths to sympy AND-formulas.

        Because features are binarized before fitting, every split is at 0.5.
        Each condition on feature f becomes:
          left  branch (≤ 0.5) → ¬f   (feature is 0)
          right branch (> 0.5) → f    (feature is 1)

        We deduplicate literals per feature within a path: if the same feature
        appears twice (shouldn't happen after binarization but kept as safety
        net), contradictions f ∧ ¬f collapse to ⊥ and the whole path is dropped.
        """
        if self._tree is None:
            return []

        tree = self._tree
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        value = tree.tree_.value
        n_features = len(self._feature_names)

        def _safe_sym(feat_idx: int) -> Symbol:
            name = (
                self._feature_names[feat_idx]
                if feat_idx < n_features
                else f"f_{feat_idx}"
            )
            return Symbol(name.replace(" ", "_").replace(".", "_").replace("-", "_"))

        formulas = []

        def recurse(node: int, pos: dict[int, bool]) -> None:
            """
            pos maps feat_idx → True (feature is 1) / False (feature is 0).
            We track per-feature polarity to detect within-path contradictions.
            """
            if children_left[node] == children_right[node]:  # leaf
                cls = int(value[node].argmax())
                if cls == 1 and pos:
                    parts = []
                    for feat_idx, is_positive in pos.items():
                        sym = _safe_sym(feat_idx)
                        parts.append(sym if is_positive else Not(sym))
                    formula = parts[0] if len(parts) == 1 else And(*parts)
                    formulas.append(formula)
                return

            feat = feature[node]

            # Left child: feature ≤ threshold  → for binarized data this means feature = 0
            new_pos_left = dict(pos)
            if feat in new_pos_left:
                if new_pos_left[feat] is True:
                    # contradiction: already committed to feat=1, now feat=0 → drop path
                    pass
                # else consistent (feat=0 again) — nothing to add
            else:
                new_pos_left[feat] = False
            if not (feat in pos and pos[feat] is True):
                recurse(children_left[node], new_pos_left)

            # Right child: feature > threshold → feature = 1
            new_pos_right = dict(pos)
            if feat in new_pos_right:
                if new_pos_right[feat] is False:
                    # contradiction: already committed to feat=0, now feat=1 → drop path
                    pass
                # else consistent
            else:
                new_pos_right[feat] = True
            if not (feat in pos and pos[feat] is False):
                recurse(children_right[node], new_pos_right)

        recurse(0, {})
        return formulas

    def explain(self, X: np.ndarray, feature_names: list[str] | None = None, **kwargs) -> dict[str, Any]:
        self.fit(X, feature_names)
        formulas = self.to_sympy()

        importances = self._tree.feature_importances_ if self._tree else np.zeros(X.shape[1])

        result = {
            "attributions": np.tile(importances, (len(X), 1)),
            "method": "rule_extraction",
            "target_label": self._target_label,
            "rules": self._rules,
            "sympy_formulas": formulas,
            "summary": (
                f"Target: {self._target_label} — "
                f"extracted {len(formulas)} rules (depth={self.max_depth})"
            ),
            "feature_importances": importances,
            "n_rules": len(formulas),
            "rule_complexity": sum(len(str(f)) for f in formulas),
            "mean_abs_attr": importances,
        }

        # For template games, also extract canonical rules using depth-2 trees
        # with flip_target=True (mixed-polarity clauses)
        from ..games.sudoku import SudokuGame
        from ..games.minesweeper import MinesweeperGame
        
        if isinstance(self.game, (SudokuGame, MinesweeperGame)):
            try:
                canonical_rules, canonical_stats = self.extract_all_canonical_rules(X, self.game)
                result["canonical_rules"] = canonical_rules
                result["canonical_stats"] = canonical_stats
            except Exception as e:
                # If canonical extraction fails, just log and continue with single-tree results
                from ..utils.logging import get_logger
                logger = get_logger(__name__)
                logger.warning(f"[{self.game.name}] Canonical rule extraction failed: {e}")

        return result

    def fit_for_dim(
        self,
        X: np.ndarray,
        target_dim: int,
        feature_names: list[str] | None = None,
        flip_target: bool = False,
        max_depth_override: int | None = None,
    ) -> None:
        """Fit a decision tree predicting a specific output dimension.

        flip_target=True inverts the binary target so the positive class becomes
        "cell does NOT have digit d".  This produces mixed-polarity leaf paths
        (one positive + one negative literal) that satisfy canonical templates.
        """
        from sklearn.tree import DecisionTreeClassifier

        X_sub = X[:min(self.n_samples, len(X))]
        X_bin = self._binarize(X_sub)
        y_nn = self._get_nn_predictions(X_sub)
        y_target = y_nn[:, target_dim] if y_nn.ndim > 1 else y_nn
        if flip_target:
            y_target = 1 - y_target
        self._target_label = self._dim_to_label(target_dim)
        self._feature_names = feature_names or [f"f_{i}" for i in range(X_bin.shape[1])]
        depth = max_depth_override if max_depth_override is not None else self.max_depth
        self._tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self._tree.fit(X_bin, y_target)
        self._rules = self._extract_rules()

    def extract_all_canonical_rules(
        self, X: np.ndarray, game: Any
    ) -> tuple[list[dict], dict]:
        """Fit one tree per blank cell × digit and collect matched canonical NL rules.

        Uses flip_target=True (positive class = "cell does NOT hold digit d") and
        max_depth=2 so leaf paths are exactly 2 literals — the mixed-polarity shape
        that canonical template matchers require.

        Returns
        -------
        rules : list[dict]
            Unique matched rules with keys: template, text, target_cell.
        stats : dict
            n_total   — total formulas generated across all (cell, digit) trees
            n_matched — formulas that matched a canonical template (before dedup)
            canonical_match_rate — n_matched / n_total (0.0 when n_total == 0)
        """
        from ..games.sudoku import SudokuGame
        from ..games.minesweeper import MinesweeperGame
        from ..viz.templates import render_clauses_as_nl, _formula_to_atoms, match_clause
        from ..viz.templates import select_decoder
        from ..utils.logging import get_logger

        logger = get_logger(__name__)

        # Pick a feature decoder consistent with the model's input layout.
        # CNNs trained on spatial encoding use channel-first features; everything
        # else uses cell-major one-hot.  Without this, CNN clauses decode to
        # nonsensical (row, col, digit) triples and no template ever fires.
        decoder = select_decoder(self.model, game)

        X_sub = X[:min(self.n_samples, len(X))]
        X_bin = self._binarize(X_sub)
        y_nn = self._get_nn_predictions(X_sub)
        feature_names = [f"f_{i}" for i in range(X_bin.shape[1])]

        # Collect (dim, flip_target, X_local, pass_name) triples.
        # Sudoku: TWO passes per cell.
        #   1. blank-only pass — tree splits on row/col neighbours of the blank
        #      cell, recovers row/column_uniqueness templates.
        #   2. all-samples pass — tree splits on the cell's OWN digit features
        #      (which vary because the cell is filled in different samples),
        #      recovers cell_uniqueness clauses ("if cell holds d_a it doesn't
        #      hold d_b").  Without this pass cell_uniqueness can NEVER fire
        #      because the blank-only filter zeroes out the target cell's
        #      one-hot block.
        # Other games (Minesweeper): top-variance output dims, single pass.
        target_triples: list[tuple[int, bool, np.ndarray, str]] = []
        if isinstance(game, SudokuGame):
            n = game.size
            # Use lower threshold for canonical extraction (depth-2 trees need flexibility)
            min_blank = max(10, 15)  # Reduced from min_samples_leaf * 2 (60) to 15
            skipped_cells = 0
            included_cells = 0
            for cell in range(n * n):
                cell_feats = X_bin[:, cell * n: cell * n + n]
                blank_mask = cell_feats.sum(axis=1) == 0  # True where cell is blank
                filled_mask = ~blank_mask
                n_blank = int(blank_mask.sum())
                n_filled = int(filled_mask.sum())
                if n_blank < min_blank and n_filled < min_blank:
                    skipped_cells += 1
                    continue  # cell has no usable samples — skip
                included_cells += 1
                # Pass 1: row/col uniqueness — blank-only samples
                if n_blank >= min_blank:
                    X_blank = X[blank_mask]
                    for digit in range(n):
                        target_triples.append(
                            (cell * n + digit, True, X_blank, "row_col_uniq")
                        )
                # Pass 2: cell uniqueness — all samples so the cell's own
                # one-hot block varies and the tree can split on it.  flip
                # so leaf paths are mixed-polarity, which the cell_uniqueness
                # template (two same-cell different-digit atoms) requires.
                if n_filled >= min_blank:
                    for digit in range(n):
                        target_triples.append(
                            (cell * n + digit, True, X, "cell_uniq")
                        )

            logger.info(f"[canonical] Sudoku {n}×{n}: included {included_cells}/{n*n} cells "
                       f"(skipped {skipped_cells}), {len(target_triples)} (cell,digit,pass) triples")
        else:
            # For other games (Minesweeper, etc.): use top-variance output dims
            # with flip_target=True to learn patterns for "uncertain predictions"
            var = y_nn.var(axis=0) if y_nn.ndim > 1 else np.array([y_nn.var()])
            for d in np.argsort(var)[::-1][:16]:
                target_triples.append((int(d), True, X, "variance"))  # flip_target=True for uncertainty
            logger.info(f"[canonical] {game.name}: {len(target_triples)} top-variance dims (flipped); "
                       f"decoder={decoder.__name__}")

        seen_texts: set[str] = set()
        all_rules: list[dict] = []
        n_total = 0
        n_matched = 0
        n_processed = 0
        n_no_formulas = 0

        # Temporarily reduce min_samples_leaf for canonical extraction (depth-2 trees)
        orig_min_samples_leaf = self.min_samples_leaf
        self.min_samples_leaf = max(5, self.min_samples_leaf // 4)  # Reduce to ~7-8 from 30

        for dim, flip, X_local, pass_name in target_triples:
            n_processed += 1
            self.fit_for_dim(X_local, dim, feature_names, flip_target=flip, max_depth_override=2)
            formulas = self.to_sympy()
            if not formulas:
                n_no_formulas += 1
                continue
            try:
                nl = render_clauses_as_nl(formulas, game, decoder=decoder)
            except Exception:
                continue
            n_total += len(formulas)
            for formula, text in zip(formulas, nl["nl_rules"]):
                if text != "(non-canonical)":
                    n_matched += 1
                if text == "(non-canonical)" or text in seen_texts:
                    continue
                seen_texts.add(text)
                try:
                    atoms = _formula_to_atoms(formula, game, decoder=decoder)
                    rule_obj = match_clause(atoms, game) if atoms is not None else None
                except Exception:
                    atoms = None
                    rule_obj = None
                # Keep atoms alongside text so the Z3 validity check (P1-10)
                # has structural data to work with.  Atom is a frozen
                # dataclass; convert to plain dicts for JSON-serialisable
                # downstream consumers.
                atom_dicts = []
                if atoms is not None:
                    for a in atoms:
                        atom_dicts.append({
                            "kind": a.kind,
                            "payload": dict(a.payload),
                            "polarity": a.polarity,
                        })
                all_rules.append({
                    "template": rule_obj.template if rule_obj else "unknown",
                    "text": text,
                    "target_cell": self._target_label,
                    "atoms": atom_dicts,
                    "pass": pass_name,
                })

        # Restore original min_samples_leaf
        self.min_samples_leaf = orig_min_samples_leaf

        logger.info(f"[canonical] Processed {n_processed} (cell,digit) targets: "
                   f"{n_no_formulas} produced no class=1 leaves, "
                   f"{n_total} total formulas extracted, {n_matched} matched canonical")

        # Per-template breakdown of unique canonical rules (after dedup).
        # "non_canonical" is excluded from all_rules (the dedup loop above filtered it),
        # so by_template here counts only canonical hits.
        from collections import Counter
        by_template = dict(Counter(r["template"] for r in all_rules))

        # Per-blank-cell distribution: how many unique canonical rules fired at each
        # target_cell label.  Used by Task P1-8 (per-cell heatmap) downstream.
        by_target_cell = dict(Counter(r["target_cell"] for r in all_rules))

        # Template coverage = how many of the game's registered template families
        # ever produced at least one canonical match (Task P1-9).
        from ..viz.templates import _game_templates
        template_fns = _game_templates(game)
        n_template_families = len(template_fns)
        template_names_fired = sorted({r["template"] for r in all_rules
                                        if r["template"] != "unknown"})
        template_coverage = (
            len(template_names_fired) / n_template_families
            if n_template_families > 0 else 0.0
        )

        stats = {
            "n_total": n_total,
            "n_matched": n_matched,
            "n_unique_rules": len(all_rules),
            "canonical_match_rate": n_matched / n_total if n_total > 0 else 0.0,
            "by_template": by_template,
            "by_target_cell": by_target_cell,
            "templates_fired": template_names_fired,
            "n_template_families": n_template_families,
            "template_coverage": template_coverage,
        }

        # Z3-based validity (Task P1-10): for every canonically-formed clause,
        # ask Z3 whether it's logically entailed by the game's rule set.  This
        # separates "looks like row_uniqueness" from "actually IS a valid
        # row_uniqueness consequence".
        try:
            from ..symbolic.validation import validate_canonical_rules
            v_stats = validate_canonical_rules(game, all_rules)
            stats.update(v_stats)
        except Exception as e:
            logger.warning(f"[canonical] Z3 validation skipped: {e}")

        return all_rules, stats

    def fidelity(self, X: np.ndarray, y: np.ndarray, explanation: dict) -> float:
        """Fidelity: cross-validated agreement between surrogate tree and NN on binarized features."""
        if self._tree is None:
            return 0.0

        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        X_bin = self._binarize(X)
        nn_preds = self._get_nn_predictions(X)
        nn_preds, _ = self._pick_target_dim(nn_preds)

        if len(np.unique(nn_preds)) < 2:
            return 1.0 if (self._tree.predict(X_bin) == nn_preds).all() else 0.0

        cv_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        n_folds = min(5, max(2, len(X_bin) // 20))
        scores = cross_val_score(cv_tree, X_bin, nn_preds, cv=n_folds, scoring="accuracy")
        return float(scores.mean())
