"""Canonical-rule template registry for logic games."""
from __future__ import annotations
from typing import Any, Callable

from .atom import Atom, NaturalLanguageRule
from .feature_decoder import decode, decode_spatial
from . import sudoku, minesweeper, universal

# Decoder type: maps (feature_name, polarity, game) -> Atom | None.
# Default is the one_hot/cell-major layout used by MLP / Transformer / RL / GNN.
# Spatial CNN models must use ``decode_spatial`` instead.
DecoderFn = Callable[[str, bool, Any], "Atom | None"]


def select_decoder(model: Any | None, game: Any | None = None) -> DecoderFn:
    """Pick the right feature decoder for a given model.

    CNN models trained with spatial encoding produce trees whose feature
    indices follow a channel-first layout; everything else uses cell-major
    one-hot. Without routing CNN through ``decode_spatial`` the atoms come
    back nonsensical and no canonical template ever matches.
    """
    if model is None:
        return decode
    try:
        from ..models.cnn import CNN
    except Exception:  # pragma: no cover — defensive import
        return decode
    if isinstance(model, CNN):
        return decode_spatial
    return decode

# Per-game template list — each function takes (atoms, game) → NaturalLanguageRule | None.
#
# Three-tier ordering (match_clause returns on first hit, so order = priority):
#   Tier 1: strict game-specific templates — recognised solving techniques.
#   Tier 2: relaxed game-specific templates — partial / fragmentary forms
#           of the same techniques (e.g. partial_naked_single = naked_single
#           without the explicit positive atom).
#   Tier 3: universal structural templates — game-agnostic patterns
#           (same-cell, same-row, spatial-locality, mixed-polarity).
#
# Downstream aggregators can split canonical_match_rate by tier — strict-only
# for headline "the NN learned a recognised technique" numbers, full set for
# "the NN learned ANY genuine dependency" numbers.
_UNIVERSAL = [
    universal.same_cell_clause,
    universal.same_row_clause,
    universal.same_column_clause,
    universal.spatial_locality_clause,
    universal.mixed_polarity_clause,
    universal.long_conjunction_clause,
]

TEMPLATES: dict[str, list] = {
    "sudoku4": [
        # Tier 1: strict
        sudoku.naked_single, sudoku.hidden_single, sudoku.naked_pair,
        sudoku.pointing_pair,
        sudoku.cell_uniqueness, sudoku.row_uniqueness,
        sudoku.column_uniqueness, sudoku.box_uniqueness,
        # Tier 2: relaxed
        sudoku.partial_naked_single, sudoku.partial_hidden_single,
        # Tier 3: universal
        *_UNIVERSAL,
    ],
    "sudoku9": [
        sudoku.naked_single, sudoku.hidden_single, sudoku.naked_pair,
        sudoku.pointing_pair,
        sudoku.cell_uniqueness, sudoku.row_uniqueness,
        sudoku.column_uniqueness, sudoku.box_uniqueness,
        sudoku.partial_naked_single, sudoku.partial_hidden_single,
        *_UNIVERSAL,
    ],
    "minesweeper8": [
        # Tier 1: strict
        minesweeper.zero_safe_neighbours,
        minesweeper.flagged_satisfies_clue,
        minesweeper.isolated_clue,
        minesweeper.local_count, minesweeper.local_exhaustion,
        minesweeper.two_clue_chain,
        # Tier 2: relaxed
        minesweeper.safe_low_clue, minesweeper.mine_high_clue,
        minesweeper.neighbour_clue_signal,
        # Tier 3: universal
        *_UNIVERSAL,
    ],
    "minesweeper16": [
        minesweeper.zero_safe_neighbours,
        minesweeper.flagged_satisfies_clue,
        minesweeper.isolated_clue,
        minesweeper.local_count, minesweeper.local_exhaustion,
        minesweeper.two_clue_chain,
        minesweeper.safe_low_clue, minesweeper.mine_high_clue,
        minesweeper.neighbour_clue_signal,
        *_UNIVERSAL,
    ],
}

# Generic key lookup: strip trailing digits for prefix matching
def _game_templates(game: Any) -> list:
    name = getattr(game, "name", "")
    if name in TEMPLATES:
        return TEMPLATES[name]
    # Prefix match: "sudoku" matches "sudoku4", "sudoku9", etc.
    for key, fns in TEMPLATES.items():
        if key.rstrip("0123456789") == name.rstrip("0123456789"):
            return fns
    return []


def match_clause(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """Try each registered template for the game; return first match or None."""
    for template_fn in _game_templates(game):
        rule = template_fn(atoms, game)
        if rule is not None:
            return rule
    return None


def _formula_to_atoms(
    formula: Any,
    game: Any,
    decoder: DecoderFn | None = None,
) -> list[Atom] | None:
    """
    Decompose a sympy expression into a list of decoded Atoms.
    Returns None if any literal is undecodable or the formula shape is unsupported.

    ``decoder`` defaults to the one-hot ``decode``; pass ``decode_spatial``
    (or use ``select_decoder(model)``) for CNN/spatial-encoded models.
    """
    from sympy import And, Not, Symbol

    dec = decoder if decoder is not None else decode

    if isinstance(formula, Symbol):
        atom = dec(str(formula), True, game)
        return [atom] if atom is not None else None

    if isinstance(formula, Not):
        inner = formula.args[0]
        if not isinstance(inner, Symbol):
            return None
        atom = dec(str(inner), False, game)
        return [atom] if atom is not None else None

    if isinstance(formula, And):
        atoms: list[Atom] = []
        for lit in formula.args:
            if isinstance(lit, Symbol):
                atom = dec(str(lit), True, game)
            elif isinstance(lit, Not) and isinstance(lit.args[0], Symbol):
                atom = dec(str(lit.args[0]), False, game)
            else:
                return None
            if atom is None:
                return None
            atoms.append(atom)
        return atoms if atoms else None

    return None


def render_clauses_as_nl(
    formulas: list,
    game: Any,
    decoder: DecoderFn | None = None,
) -> dict:
    """
    Translate a list of sympy Boolean formulas into natural-language rule strings.

    Returns:
        nl_rules           : one string per formula; "(non-canonical)" when no template matched
        matched            : count of successfully matched clauses
        total              : total clause count
        canonical_match_rate : matched / total (0.0 when total == 0)
        by_template        : {template_name: count} including "non_canonical"
    """
    nl_rules: list[str] = []
    matched = 0
    by_template: dict[str, int] = {}

    for formula in formulas:
        atoms = _formula_to_atoms(formula, game, decoder=decoder)
        rule = match_clause(atoms, game) if atoms is not None else None

        if rule is not None:
            nl_rules.append(rule.text)
            matched += 1
            by_template[rule.template] = by_template.get(rule.template, 0) + 1
        else:
            nl_rules.append("(non-canonical)")
            by_template["non_canonical"] = by_template.get("non_canonical", 0) + 1

    total = len(formulas)
    return {
        "nl_rules": nl_rules,
        "matched": matched,
        "total": total,
        "canonical_match_rate": matched / total if total > 0 else 0.0,
        "by_template": by_template,
    }
