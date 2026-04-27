"""Canonical-rule template registry for logic games."""
from __future__ import annotations
from typing import Any

from .atom import Atom, NaturalLanguageRule
from .feature_decoder import decode
from . import sudoku, minesweeper

# Per-game template list — each function takes (atoms, game) → NaturalLanguageRule | None
TEMPLATES: dict[str, list] = {
    "sudoku4":       [sudoku.cell_uniqueness, sudoku.row_uniqueness,
                      sudoku.column_uniqueness, sudoku.box_uniqueness],
    "sudoku9":       [sudoku.cell_uniqueness, sudoku.row_uniqueness,
                      sudoku.column_uniqueness, sudoku.box_uniqueness],
    "minesweeper8":  [minesweeper.local_count, minesweeper.local_exhaustion],
    "minesweeper16": [minesweeper.local_count, minesweeper.local_exhaustion],
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


def _formula_to_atoms(formula: Any, game: Any) -> list[Atom] | None:
    """
    Decompose a sympy expression into a list of decoded Atoms.
    Returns None if any literal is undecodable or the formula shape is unsupported.
    """
    from sympy import And, Not, Symbol

    if isinstance(formula, Symbol):
        atom = decode(str(formula), True, game)
        return [atom] if atom is not None else None

    if isinstance(formula, Not):
        inner = formula.args[0]
        if not isinstance(inner, Symbol):
            return None
        atom = decode(str(inner), False, game)
        return [atom] if atom is not None else None

    if isinstance(formula, And):
        atoms: list[Atom] = []
        for lit in formula.args:
            if isinstance(lit, Symbol):
                atom = decode(str(lit), True, game)
            elif isinstance(lit, Not) and isinstance(lit.args[0], Symbol):
                atom = decode(str(lit.args[0]), False, game)
            else:
                return None
            if atom is None:
                return None
            atoms.append(atom)
        return atoms if atoms else None

    return None


def render_clauses_as_nl(formulas: list, game: Any) -> dict:
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
        atoms = _formula_to_atoms(formula, game)
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
