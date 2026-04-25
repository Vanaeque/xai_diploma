# Task for Code Copilot — Natural-Language Rules + Four-Method Default

## Goal

Two changes to the existing `symbolic_xai_logic` repo, plus one small ergonomics fix:

1. **Add canonical-rule template matching** so extracted DNF clauses are translated into human-readable game rules (e.g. *"if cell (0,1) holds 2, then cell (0,2) does not hold 2"*). Wire it into the experiment pipeline so the rendered explanation always includes a "Natural-Language Rules" section, and report a new metric `canonical_match_rate` in every run's JSON.
2. **Standardize on four XAI methods** for the diploma: `rule_extraction`, `symbolic_regression`, `concept_probe`, `lrp`. Remove `lime` and `shap` from default reproduction paths but keep the modules importable so a user can still run `--xai lime` explicitly.
3. **Split training and explanation cleanly** so a checkpoint produced by `scripts/train.py` can be re-explained any number of times by `scripts/explain.py` without retraining.

Don't break existing tests. Don't refactor unrelated modules. Make changes minimally invasive.

---

## Current state — what already exists, what's missing

You can rely on this; I audited the repo before writing the task.

**Already implemented (do not rebuild):**

- `src/symbolic_xai_logic/xai/rule_extraction.py` — binarizes one-hot features at 0.5, fits a sklearn DecisionTreeClassifier on NN predictions, extracts DNF via tree-walk, deduplicates literals per feature, emits a `target_label` like `"cell (2,1) = digit 3"` via `_dim_to_label()`.
- `src/symbolic_xai_logic/xai/symbolic_regression.py`, `concept_probe.py`, `lrp.py`, `lime_explainer.py`, `shap_explainer.py`, `base.py`.
- `src/symbolic_xai_logic/games/{sudoku,nqueens,knights_knaves,sat3,minesweeper}.py`.
- `src/symbolic_xai_logic/experiments/runner.py` — full pipeline orchestrator with a `run_all_xai()` method.
- `scripts/train.py`, `scripts/explain.py`, `scripts/run_experiment.py`.
- `src/symbolic_xai_logic/viz/rule_render.py` — generic pretty-printer for the explanation dict; **no decoder, no template matching, no NL rendering.**

**Missing (build these):**

- Any feature-index → `(cell, digit)` / `(cell, channel)` decoder usable from outside `rule_extraction.py`.
- Canonical-rule templates per game (Sudoku row / column / box / cell uniqueness; Minesweeper local-count / local-exhaustion).
- Pattern matcher that takes a sympy clause and returns the matching template + a natural-language string, or None.
- Plumbing of `canonical_match_rate` into `compute_fidelity()` / the report JSON.
- A `--no-xai` flag on `scripts/train.py` so it produces only a checkpoint.

**Known issue you should also fix (≤5 lines):** `scripts/train.py:18-19` lists game choices `["sudoku","nqueens","knights_knaves","sat3"]` and model choices `["mlp","gnn","transformer"]`. Add `"minesweeper"` and `"cnn"`. The runner already handles them; the CLI just doesn't expose them.

---

## Files you will create or modify

```
src/symbolic_xai_logic/viz/
├── rule_render.py                 # MODIFY — add NL-rules section to render_explanation()
└── templates/                     # NEW package
    ├── __init__.py                # NEW — registry, public match_clause(), render_clauses_as_nl()
    ├── atom.py                    # NEW — Atom dataclass, NaturalLanguageRule dataclass
    ├── feature_decoder.py         # NEW — game-aware feature-index → Atom decoder
    ├── sudoku.py                  # NEW — 4 templates: cell/row/col/box uniqueness
    └── minesweeper.py             # NEW — 2 templates: local_count, local_exhaustion

src/symbolic_xai_logic/experiments/runner.py   # MODIFY — call render_clauses_as_nl, write canonical_match_rate to report
src/symbolic_xai_logic/symbolic/fidelity.py    # MODIFY — accept and pass through canonical_match_rate
scripts/train.py                                # MODIFY — add --no-xai, add minesweeper/cnn to choices
scripts/explain.py                              # MODIFY — print NL section (already prints render_explanation, just verify)
scripts/reproduce_all.py                        # MODIFY (or CREATE) — use the four-method default
src/symbolic_xai_logic/experiments/runner.py:148  # MODIFY — DEFAULT_XAI_METHODS constant

tests/test_rule_templates.py                    # NEW — unit tests for decoder + templates
tests/test_xai_fidelity.py                      # MODIFY only if it iterates the old default list
README.md                                       # MODIFY — add "Natural-Language Rules" + four-method note
```

---

## Detailed specs

### `templates/atom.py`

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Atom:
    """One literal in a DNF clause, decoded back to a game-level fact."""
    kind: str          # "cell_digit" | "mine" | "queen" | "queen_attacks" | ...
    payload: dict[str, Any]   # game-specific, e.g. {"row": 0, "col": 1, "digit": 2}
    polarity: bool     # True = positive literal, False = negated

@dataclass(frozen=True)
class NaturalLanguageRule:
    template: str          # e.g. "row_uniqueness"
    text: str              # human-readable rendering
    atoms: tuple[Atom, ...]  # the matched atoms, for traceability
```

### `templates/feature_decoder.py`

Single public function:

```python
def decode(feature_name: str, polarity: bool, game) -> Atom | None:
    """
    Map a sympy Symbol's name (e.g. "f_38") plus polarity into an Atom for the given game.
    Returns None if the feature name is unrecognized for this game.
    """
```

Implementation per game type — reuse the math currently inlined in `rule_extraction.py` `_dim_to_label`:

- **Sudoku** (n×n): parse `f_{i}`. `cell = i // n`; `row, col = divmod(cell, n)`; `digit = (i % n) + 1`. Return `Atom("cell_digit", {"row":row,"col":col,"digit":digit}, polarity)`.
- **Minesweeper**: read `games/minesweeper.py` to determine the encoding; with `n_channels = N_SPATIAL_CHANNELS` the feature layout is `(row, col, channel)`. Return `Atom("cell_state", {"row":row,"col":col,"channel":ch}, polarity)`. Channel semantics: 0=hidden, 1=flagged, 2..10=number revealed.
- For the others (nqueens / knights_knaves / sat3): out of scope for this task — return None.

**Refactor opportunity:** move the decoding math out of `rule_extraction._dim_to_label` and have it call this decoder. Keep the existing public string output by formatting the Atom. Don't change the string format.

### `templates/sudoku.py`

Four matchers. Each takes `atoms: list[Atom]` (one DNF conjunction) and the game; returns `NaturalLanguageRule | None`.

- **`cell_uniqueness(atoms, game)`** — exactly two atoms, same `(row, col)`, different digits, mixed polarity. Text: `"if cell ({r},{c}) holds {d_pos}, it does not hold {d_neg}"`.
- **`row_uniqueness(atoms, game)`** — exactly two atoms, same row, same digit, different columns, mixed polarity. Text: `"if cell ({r},{c1}) holds {d}, then cell ({r},{c2}) does not hold {d}"`.
- **`column_uniqueness(atoms, game)`** — analogous on column.
- **`box_uniqueness(atoms, game)`** — same digit, different cells in the same √n × √n box, mixed polarity. Compute box id as `(r // bs)*bs + (c // bs)` where `bs = int(sqrt(game.size))`. Only define for `game.size in (4, 9, 16)`.

Return None for any clause that doesn't match.

### `templates/minesweeper.py`

- **`local_count(atoms, game)`** — pattern: a "shows N" atom (channel 2..10) on cell c plus N flagged-channel atoms on neighbors of c, plus negative hidden-channel atoms on the remaining neighbors. Text: `"if cell ({r},{c}) shows {n} and {n} of its neighbors are mines, the others are safe"`.
- **`local_exhaustion(atoms, game)`** — pattern: a "shows N" atom plus negated flag-atoms making remaining unknowns exactly the missing mines. Text: `"if cell ({r},{c}) shows {n} and only {k} unknown neighbors remain that account for the missing mines, all {k} are mines"`.

These are stricter to detect; if the atom set doesn't fit cleanly, return None — false negatives are fine, false positives are not.

### `templates/__init__.py`

```python
from .atom import Atom, NaturalLanguageRule
from .feature_decoder import decode
from . import sudoku, minesweeper

# Per-game template registry
TEMPLATES = {
    "sudoku":     [sudoku.cell_uniqueness, sudoku.row_uniqueness,
                   sudoku.column_uniqueness, sudoku.box_uniqueness],
    "minesweeper": [minesweeper.local_count, minesweeper.local_exhaustion],
}

def match_clause(atoms: list[Atom], game) -> NaturalLanguageRule | None:
    for template_fn in TEMPLATES.get(game.name, []):
        rule = template_fn(atoms, game)
        if rule is not None:
            return rule
    return None

def render_clauses_as_nl(formulas, game) -> dict:
    """
    formulas: list of sympy Boolean expressions (And/Not/Symbol).
    Returns {
        "nl_rules": list[str],     # one entry per clause, "(non-canonical)" if no match
        "matched":  int,
        "total":    int,
        "canonical_match_rate": float,
        "by_template": dict[str, int],  # how many clauses matched each template
    }
    """
```

The `render_clauses_as_nl` function decomposes each sympy formula into its top-level conjuncts, calls `decode()` on each Symbol/Not(Symbol) to build a list[Atom], runs `match_clause`, and assembles the dict. Handle the degenerate "single literal" formula (e.g. `¬f_38`) — that's one atom, treat it as a one-atom clause.

### `viz/rule_render.py` modification

Extend `render_explanation(explanation, game_name="", xai_name="", game=None)` (add the `game=None` parameter). When `game` is provided AND `explanation` has `sympy_formulas`, call `render_clauses_as_nl(explanation["sympy_formulas"], game)` and append a section:

```
### Natural-Language Rules (canonical match rate: 4/6 = 0.67):
1. ✓ [cell_uniqueness] if cell (0,1) holds 2, it does not hold 4
2. ✓ [row_uniqueness] if cell (0,1) holds 2, then cell (0,2) does not hold 2
3. ✗ (non-canonical)  ¬f_38 ∧ f_5
...

### By template:
  cell_uniqueness:    2
  row_uniqueness:     2
  column_uniqueness:  0
  box_uniqueness:     0
  non_canonical:      2
```

Backwards compatibility: when `game=None`, behavior is unchanged.

### `experiments/runner.py` modification

Around line 138 (after `compute_fidelity`), call `render_clauses_as_nl` if the explanation has `sympy_formulas`, and merge `canonical_match_rate`, `matched`, `total`, `by_template` into the report dict before saving the JSON. Same value should be available in the printed output.

`compute_fidelity` (in `symbolic/fidelity.py`) will need to accept and store `canonical_match_rate` as a field on `FidelityReport`. Default to `None` if not applicable (e.g. for LRP / SHAP / LIME where there are no symbolic formulas).

### `scripts/train.py` modification

Add `--no-xai` flag. When set, build the same config but skip the explainer step in the runner. The simplest implementation: don't use `ExperimentRunner.run()`; instead expose a new `ExperimentRunner.train_only()` method that runs everything up to and including `trainer.train(...)` and `save_json(history, ...)` and returns the checkpoint path. Then `train.py` calls `train_only()` when `--no-xai` is set.

Also add `"minesweeper"` to the game choices and `"cnn"` to the model choices.

### `scripts/explain.py` modification

Verify (or fix) that `render_explanation` is called with the `game=` argument so the NL section actually prints. The script already loads the game — pass it through.

### `scripts/reproduce_all.py` (create if absent)

Iterate the four-method default over both Sudoku 4×4 (MLP) and Minesweeper 8×8 (CNN). 5 seeds for the headline configs, 3 for the rest. Each combination produces one row in `results/summary.csv`.

```python
DEFAULT_XAI_METHODS = ["rule_extraction", "symbolic_regression", "concept_probe", "lrp"]
```

Make this constant the single source of truth. `experiments/runner.py:148` should import it. The argparse `choices=` lists in `scripts/train.py` and `scripts/explain.py` should still allow LIME/SHAP for individual runs (no regression) — only the default iteration excludes them.

---

## Tests

### NEW `tests/test_rule_templates.py`

Cover the decoder and at least each Sudoku template once positively and once negatively:

- `decode("f_38", True, sudoku4)` returns `Atom("cell_digit", {"row":2,"col":1,"digit":3}, True)`.
- A clause `[Atom("cell_digit",{"row":0,"col":1,"digit":2},True), Atom("cell_digit",{"row":0,"col":1,"digit":4},False)]` matches `cell_uniqueness` and the rendered text contains `"cell (0,1)"`, `"holds 2"`, `"does not hold 4"`.
- A clause `[Atom("cell_digit",{"row":0,"col":1,"digit":2},True), Atom("cell_digit",{"row":1,"col":2,"digit":3},False)]` matches no template and `match_clause` returns None.
- For Minesweeper: build at least one positive `local_count` example and one input that should NOT match.

### `tests/test_xai_fidelity.py`

If it currently iterates over the old default list, update it to `DEFAULT_XAI_METHODS`. Otherwise leave alone.

---

## Acceptance criteria

- `pytest -q` is green (existing + new tests).
- `python scripts/run_experiment.py --game sudoku --size 4 --xai rule_extraction` prints a "Natural-Language Rules" block at the end and `results/report_sudoku_mlp_rule_extraction.json` contains `canonical_match_rate`, `matched`, `total`, `by_template`.
- `python scripts/explain.py --checkpoint <path> --xai rule_extraction` produces the same NL section.
- `python scripts/train.py --game sudoku --size 4 --no-xai` trains and saves a checkpoint without invoking any explainer; subsequent `scripts/explain.py` calls work against that checkpoint.
- `python scripts/reproduce_all.py` iterates exactly the four standard methods.
- No regressions: existing `### Rules:` and `### Symbolic Formulas:` sections in `render_explanation` are unchanged; the NL block is purely additive.

## What NOT to change

- The decision-tree fitting logic in `rule_extraction.py`.
- Any existing config keys or `--xai` CLI flag semantics.
- LIME / SHAP / N-Queens / Knights & Knaves / 3-SAT modules — leave them intact even though they're not in the default path.
- The `target_label` string format used inside `rule_extraction.py` — the new decoder should reuse the math but not change the existing text output.

## Working style

- Make changes one file at a time and run the relevant tests after each. Get the decoder + Sudoku templates green first, then add Minesweeper templates, then wire into `runner.py` last.
- Commit in logical chunks with descriptive messages: `viz/templates: scaffold + atoms`, `viz/templates: sudoku matchers + tests`, `runner: emit canonical_match_rate`, `train.py: add --no-xai`, etc.
- No `pass`-only stubs, no `raise NotImplementedError`. Empty function bodies fail review.
