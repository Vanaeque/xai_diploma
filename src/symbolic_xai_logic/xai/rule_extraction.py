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
        canonical_max_depth: int | None = None,
        **kwargs,
    ):
        super().__init__(model, game)
        self.method = method
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_samples = n_samples
        # Depth used by extract_all_canonical_rules.  Depth-2 caps clause length
        # at 2 atoms so only 2-atom templates (row/col/box/cell uniqueness) can
        # ever fire — naked-single, hidden-single, naked-pair, and the multi-atom
        # minesweeper clue rules all need more.  Default 4 = covers all sudoku4
        # naked/hidden single clauses (3-4 atoms each) and most minesweeper
        # local_count / local_exhaustion patterns for clues up to ~3.
        #
        # Sudoku9 special case: the canonical pass fits 1458 trees per run
        # (81 cells × 9 digits × 2 passes); at depth-4 each tree emits up to
        # 16 leaves, generating ~12k clauses and ~2 min of sympy + ~1 min of
        # template-match work per run.  Depth-3 produces ~6k clauses (~half
        # the wall time) and we lose only full naked_pair patterns — full
        # naked_single on sudoku9 needs depth 9 anyway, so the quality drop
        # is negligible while wall time roughly halves.  Override per-game
        # via the ``canonical_max_depth`` constructor argument.
        if canonical_max_depth is not None:
            self.canonical_max_depth = canonical_max_depth
        else:
            # Game-adaptive default: shallower for sudoku9 to keep run cost reasonable.
            try:
                game_size = getattr(game, "size", None)
                game_name = getattr(game, "name", "")
            except Exception:
                game_size, game_name = None, ""
            self.canonical_max_depth = 3 if (
                isinstance(game_size, int) and game_size >= 9 and "sudoku" in game_name
            ) else 4
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
        """Forward-pass the NN on X and return binary (n, output_dim) labels.

        Moves the input tensor onto the model's device so this works when the
        model lives on CUDA (see scripts/explain.py's --device flag).
        """
        self.model.eval()
        # Infer device from model parameters (no-op if the model is on CPU).
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=device)
            out = torch.sigmoid(self.model(t)).detach().cpu().numpy()
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
        precomputed_y_nn: np.ndarray | None = None,
    ) -> None:
        """Fit a decision tree predicting a specific output dimension.

        flip_target=True inverts the binary target so the positive class becomes
        "cell does NOT have digit d".  This produces mixed-polarity leaf paths
        (one positive + one negative literal) that satisfy canonical templates.

        ``precomputed_y_nn`` lets the caller pass cached NN predictions to avoid
        re-running the (potentially expensive) forward pass — critical when this
        method is called hundreds of times in extract_all_canonical_rules.
        """
        from sklearn.tree import DecisionTreeClassifier

        X_sub = X[:min(self.n_samples, len(X))]
        X_bin = self._binarize(X_sub)
        if precomputed_y_nn is not None:
            # Use the cached predictions; trim to current X_sub length if needed.
            y_nn = precomputed_y_nn[:len(X_sub)]
        else:
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
        max_depth defaults to ``self.canonical_max_depth`` (4 by default).
        Depth-2 only captures row/col/box/cell-uniqueness (2-atom rules);
        depth-4+ unlocks naked-single, hidden-single, naked-pair, and the
        full minesweeper local_count / local_exhaustion patterns.

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
        elif isinstance(game, MinesweeperGame):
            # Per-cell pass over the grid, but with a **local-neighbourhood
            # feature mask**.  Without this, the depth-4 trees pick splits on
            # cells far from the target (e.g. opposite-corner correlations) —
            # statistically valid but logically nonsense.  Real minesweeper
            # rules are local; restricting the tree's available features to
            # the 3×3 area around the target cell forces it to discover
            # local_count / local_exhaustion / zero_safe_neighbours shapes.
            n = game.size
            from ..games.minesweeper import N_SPATIAL_CHANNELS
            # For one-hot MLP/Transformer/RL encoding the feature layout is
            # (cell-major, channel-minor) of width N_SPATIAL_CHANNELS per cell.
            # CNN spatial encoding uses a different layout — skip the mask
            # there (decoder_spatial handles the decoded atoms differently;
            # local masking on spatial channels would need its own indexing).
            from ..models.cnn import CNN as _CNN
            mask_features = not isinstance(self.model, _CNN)

            def _neighbourhood(r: int, c: int) -> list[tuple[int, int]]:
                out = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < n and 0 <= cc < n:
                            out.append((rr, cc))
                return out

            def _feature_indices_for(cells: list[tuple[int, int]]) -> list[int]:
                idxs = []
                for (rr, cc) in cells:
                    flat = rr * n + cc
                    base = flat * N_SPATIAL_CHANNELS
                    idxs.extend(range(base, base + N_SPATIAL_CHANNELS))
                return idxs

            X_full = X  # full feature matrix; we zero columns per-cell
            n_feat = X_full.shape[1] if X_full.ndim == 2 else 0
            for r in range(n):
                for c in range(n):
                    cell_idx = r * n + c
                    nbrs = _neighbourhood(r, c)
                    if mask_features and n_feat > 0:
                        # Build a boolean mask vectorised: True = "keep",
                        # False = "zero out".  Massively faster than the
                        # per-column Python loop on 8×8×11 = 704 features.
                        keep = np.zeros(n_feat, dtype=bool)
                        idxs = _feature_indices_for(nbrs)
                        idxs = [i for i in idxs if 0 <= i < n_feat]
                        keep[idxs] = True
                        X_local_in = X_full * keep.astype(X_full.dtype)
                    else:
                        X_local_in = X_full
                    # Two passes — safety rules (flip_target=True → class=1
                    # means "NOT a mine") and mine rules (flip_target=False
                    # → class=1 means "IS a mine"):
                    #   * zero_safe_neighbours, flagged_satisfies_clue,
                    #     isolated_clue, local_count — safety patterns.
                    #   * local_exhaustion — mine pattern (shows-N + N hidden).
                    # Without the False pass the tree never learns clauses
                    # describing where mines DEFINITELY are.
                    target_triples.append((cell_idx, True, X_local_in, "local_safe"))
                    target_triples.append((cell_idx, False, X_local_in, "local_mine"))
            logger.info(f"[canonical] {game.name}: {len(target_triples)} per-cell triples "
                       f"(masked={mask_features}); decoder={decoder.__name__}")
        else:
            # Fallback for any other game type: top-variance output dims.
            var = y_nn.var(axis=0) if y_nn.ndim > 1 else np.array([y_nn.var()])
            for d in np.argsort(var)[::-1][:16]:
                target_triples.append((int(d), True, X, "variance"))
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

        # Cache y_nn per unique X_local — for sudoku, each cell's blank-only
        # subset is reused across all `n` digits (9 for sudoku9), and the full
        # X is reused across every cell in the cell_uniq pass.  Without this
        # cache, fit_for_dim re-runs the entire NN forward pass ~1458 times
        # per sudoku9 run; with it, only ~82 forward passes are needed.
        # That alone is ~18× speedup on the canonical pass.
        y_nn_cache: dict[int, np.ndarray] = {}

        def _y_nn_for(X_local: np.ndarray) -> np.ndarray:
            key = id(X_local)
            cached = y_nn_cache.get(key)
            if cached is not None:
                return cached
            X_sub_local = X_local[:min(self.n_samples, len(X_local))]
            y = self._get_nn_predictions(X_sub_local)
            y_nn_cache[key] = y
            return y

        for dim, flip, X_local, pass_name in target_triples:
            n_processed += 1
            self.fit_for_dim(
                X_local, dim, feature_names,
                flip_target=flip, max_depth_override=self.canonical_max_depth,
                precomputed_y_nn=_y_nn_for(X_local),
            )
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

    # ------------------------------------------------------------------
    # Per-prediction explanation (Approach B in the design doc)
    # ------------------------------------------------------------------

    def _decode_path_atoms(self, x_bin_row: np.ndarray) -> list[Any]:
        """Walk ``self._tree`` for a single sample and decode the leaf-path
        literals into ``Atom``s using the game-appropriate decoder.

        Each tree split is on a binarized feature; left branch ⇒ "feature
        is 0" (negative atom), right branch ⇒ "feature is 1" (positive
        atom).  Returns the ordered list of decoded atoms along the root-
        to-leaf path for the input sample.
        """
        from ..viz.templates import select_decoder
        if self._tree is None:
            return []
        t = self._tree.tree_
        decoder = select_decoder(self.model, self.game)

        atoms: list[Any] = []
        node = 0
        # Children left/right are -1 for leaves
        while t.children_left[node] != -1:
            feature_idx = int(t.feature[node])
            threshold = float(t.threshold[node])
            value = float(x_bin_row[feature_idx])
            # Binarized features split at 0.5; ≤ → left → "is 0" (negative).
            if value <= threshold:
                polarity = False
                node = int(t.children_left[node])
            else:
                polarity = True
                node = int(t.children_right[node])
            fname = (
                self._feature_names[feature_idx]
                if feature_idx < len(self._feature_names)
                else f"f_{feature_idx}"
            )
            atom = decoder(fname, polarity, self.game)
            if atom is not None:
                atoms.append(atom)
        return atoms

    def _leaf_class(self, x_bin_row: np.ndarray) -> tuple[int, float]:
        """Walk the tree for one sample; return (predicted_class, confidence)."""
        if self._tree is None:
            return (0, 0.0)
        t = self._tree.tree_
        node = 0
        while t.children_left[node] != -1:
            feature_idx = int(t.feature[node])
            threshold = float(t.threshold[node])
            if x_bin_row[feature_idx] <= threshold:
                node = int(t.children_left[node])
            else:
                node = int(t.children_right[node])
        values = t.value[node][0]
        total = float(values.sum())
        if total == 0:
            return (0, 0.0)
        cls = int(values.argmax())
        return (cls, float(values[cls]) / total)

    def explain_prediction(
        self,
        x: np.ndarray,
        target_cell: tuple[int, int],
        target_digit: int | None = None,
        X_train: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Generate a per-puzzle explanation for one NN prediction.

        Parameters
        ----------
        x : np.ndarray of shape (input_dim,)
            A single puzzle's encoded input.
        target_cell : (row, col)
            Which cell to explain.  For Sudoku, also requires ``target_digit``
            unless you want to pick the NN's argmax digit automatically.
        target_digit : int | None
            For Sudoku: the digit (1-indexed) whose mine/non-mine prediction
            we explain.  ``None`` picks the NN's most-likely digit for that
            cell.  Ignored for Minesweeper (only one mine/no-mine output per
            cell).
        X_train : np.ndarray | None
            Background data to fit the surrogate tree on.  When ``None``,
            the caller must have already fit a tree via ``fit_for_dim`` for
            the desired target dim.  Passing ``X_train`` is the common path:
            the function fits a depth-``canonical_max_depth`` tree on demand.

        Returns
        -------
        dict with keys:
            prediction:        the NN's binary prediction for this cell/digit
            confidence:        NN's sigmoid output in [0, 1]
            target_label:      human-readable target description
            target_dim:        flat output index
            path_atoms:        list of Atom objects on the tree's root→leaf path
            matched_template:  template name (e.g. "hidden_single") or None
            matched_rule_text: NL text of the matched rule, or None
            surrogate_class:   surrogate tree's predicted class for this puzzle
            surrogate_conf:    surrogate's leaf-purity confidence
            explanation_text:  human-readable summary stitched together
        """
        from ..games.sudoku import SudokuGame
        from ..games.minesweeper import MinesweeperGame
        from ..viz.templates import match_clause

        r, c = target_cell

        # 1) Map (target_cell, target_digit) -> output dim, picking the
        #    argmax digit when target_digit is None for sudoku.
        n = self.game.size
        x_arr = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            t = torch.tensor(x_arr[None], dtype=torch.float32, device=device)
            nn_out = torch.sigmoid(self.model(t)).detach().cpu().numpy()[0]

        if isinstance(self.game, SudokuGame):
            cell_idx = r * n + c
            if target_digit is None:
                # argmax over the n digits at this cell
                cell_slice = nn_out[cell_idx * n : cell_idx * n + n]
                target_digit = int(cell_slice.argmax()) + 1
            target_dim = cell_idx * n + (target_digit - 1)
        elif isinstance(self.game, MinesweeperGame):
            target_dim = r * n + c
        else:
            target_dim = r * n + c

        nn_confidence = float(nn_out[target_dim])
        nn_prediction = int(nn_confidence > 0.5)

        # 2) Fit a surrogate tree for this dim on X_train (or trust an
        #    already-fit tree if X_train is None).
        if X_train is not None:
            self.fit_for_dim(
                X_train, target_dim,
                flip_target=False,
                max_depth_override=self.canonical_max_depth,
            )
        if self._tree is None:
            raise RuntimeError(
                "No surrogate tree fit. Call explain_prediction(..., X_train=X) "
                "or run fit_for_dim() yourself before requesting an explanation."
            )

        # 3) Walk the tree for THIS puzzle to get the active path.
        x_bin = self._binarize(x_arr[None])[0]
        path_atoms = self._decode_path_atoms(x_bin)
        surr_class, surr_conf = self._leaf_class(x_bin)

        # 4) Try to label the path with a canonical template.
        matched = match_clause(path_atoms, self.game)

        # 5) Stitch a natural-language summary.
        target_label = self._dim_to_label(target_dim)
        nl_lines = [
            f"NN predicts: {target_label} → "
            f"{'YES' if nn_prediction else 'no'} (confidence {nn_confidence:.2f})",
            f"Surrogate tree agrees: "
            f"{'YES' if surr_class == nn_prediction else 'NO'} "
            f"(leaf purity {surr_conf:.2f})",
        ]
        if matched is not None:
            nl_lines.append(f"Reason — {matched.template}:")
            nl_lines.append(f"  {matched.text}")
        elif path_atoms:
            atom_lines = []
            for a in path_atoms:
                cell_str = f"({a.payload.get('row')},{a.payload.get('col')})"
                sign = "" if a.polarity else "¬"
                if a.kind == "cell_digit":
                    detail = f"d{a.payload.get('digit')}"
                elif a.kind == "cell_state":
                    detail = f"ch{a.payload.get('channel')}"
                else:
                    detail = str(a.payload)
                atom_lines.append(f"  {sign}cell{cell_str} {detail}")
            nl_lines.append(
                f"Reason — surrogate path uses {len(path_atoms)} features "
                f"(no recognised template):"
            )
            nl_lines.extend(atom_lines)
        else:
            nl_lines.append(
                "Reason — surrogate path is empty (NN's decision is "
                "well-modelled by the class prior at this dim)."
            )

        explanation_text = "\n".join(nl_lines)

        return {
            "target_cell": (int(r), int(c)),
            "target_digit": target_digit,
            "target_dim": int(target_dim),
            "target_label": target_label,
            "prediction": nn_prediction,
            "confidence": nn_confidence,
            "path_atoms": path_atoms,
            "matched_template": matched.template if matched else None,
            "matched_rule_text": matched.text if matched else None,
            "surrogate_class": surr_class,
            "surrogate_conf": surr_conf,
            "explanation_text": explanation_text,
        }

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
