"""Game-agnostic ("universal") canonical-rule templates.

These templates DON'T encode any particular game's solving technique — they
recognise structural patterns in extracted clauses that are evidence the
underlying NN has learned *some* genuine dependency.  They're deliberately
permissive so a thesis can report:

  "NN learned a recognised solving rule"     (strict templates fire)
  "NN learned a structured local pattern"    (universal templates fire)
  "NN learned only noise"                    (nothing fires)

Order them AFTER the strict game-specific templates in the TEMPLATES dict;
``match_clause`` tries templates in order and returns on the first hit, so
strict matches take precedence and universals act as a fallback layer.

Each template returns a ``NaturalLanguageRule`` whose ``template`` field is
prefixed with ``universal/`` so downstream aggregators can split the two
populations cleanly.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable

from .atom import Atom, NaturalLanguageRule


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _by_polarity(atoms: list[Atom]) -> tuple[list[Atom], list[Atom]]:
    pos = [a for a in atoms if a.polarity]
    neg = [a for a in atoms if not a.polarity]
    return pos, neg


def _cell_of(a: Atom) -> tuple[int, int] | None:
    """Return (row, col) if atom has spatial payload, else None."""
    p = a.payload
    if "row" in p and "col" in p:
        return (int(p["row"]), int(p["col"]))
    return None


def _chebyshev(c1: tuple[int, int], c2: tuple[int, int]) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))


def _describe_atom(a: Atom) -> str:
    """Compact human-readable representation of one atom."""
    cell = _cell_of(a)
    sign = "" if a.polarity else "¬"
    p = a.payload
    if a.kind == "cell_digit":
        return f"{sign}cell{cell}=d{p.get('digit')}"
    if a.kind == "cell_state":
        return f"{sign}cell{cell}@ch{p.get('channel')}"
    return f"{sign}{a.kind}({p})"


# ---------------------------------------------------------------------------
# Layer A — Structural patterns (medium specificity)
# ---------------------------------------------------------------------------

def same_cell_clause(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """≥2 atoms about the SAME (row, col).

    Captures any per-cell constraint the NN learned regardless of what the
    constraint specifically is — e.g. "cell holds d1 → cell channel X" or
    "cell channel A ∧ cell channel B".  Strictly weaker than cell_uniqueness
    (no mixed-polarity requirement, no different-digit requirement).
    """
    by_cell: dict[tuple[int, int], list[Atom]] = defaultdict(list)
    for a in atoms:
        cell = _cell_of(a)
        if cell is not None:
            by_cell[cell].append(a)
    for cell, group in by_cell.items():
        if len(group) >= 2:
            desc = " ∧ ".join(_describe_atom(a) for a in group)
            text = f"per-cell constraint at cell {cell}: {desc}"
            return NaturalLanguageRule("universal/same_cell", text, tuple(group))
    return None


def same_row_clause(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """≥2 atoms in the same row across DIFFERENT cells.

    Picks up any row-confined dependency without requiring same digit or
    mixed polarity (which row_uniqueness needs).
    """
    by_row: dict[int, list[Atom]] = defaultdict(list)
    for a in atoms:
        cell = _cell_of(a)
        if cell is not None:
            by_row[cell[0]].append(a)
    for row, group in by_row.items():
        cells = {_cell_of(a) for a in group}
        if len(cells) >= 2:
            desc = " ∧ ".join(_describe_atom(a) for a in group)
            text = f"row-{row} dependency: {desc}"
            return NaturalLanguageRule("universal/same_row", text, tuple(group))
    return None


def same_column_clause(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """≥2 atoms in the same column across DIFFERENT cells."""
    by_col: dict[int, list[Atom]] = defaultdict(list)
    for a in atoms:
        cell = _cell_of(a)
        if cell is not None:
            by_col[cell[1]].append(a)
    for col, group in by_col.items():
        cells = {_cell_of(a) for a in group}
        if len(cells) >= 2:
            desc = " ∧ ".join(_describe_atom(a) for a in group)
            text = f"column-{col} dependency: {desc}"
            return NaturalLanguageRule("universal/same_column", text, tuple(group))
    return None


def spatial_locality_clause(
    atoms: list[Atom],
    game: Any,
    radius: int = 2,
) -> NaturalLanguageRule | None:
    """All atoms reference cells within Chebyshev distance ``radius`` of each other.

    A "local dependency" — the clause's premises are all geographically close.
    For minesweeper this captures any neighbourhood-of-neighbourhood pattern
    even when the clue/hidden shape doesn't match local_count or
    local_exhaustion exactly.  For sudoku it captures box-local patterns.
    """
    cells = [c for a in atoms if (c := _cell_of(a)) is not None]
    if len(cells) < 2:
        return None
    # Diameter test: pairwise max distance ≤ radius.
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if _chebyshev(cells[i], cells[j]) > radius:
                return None
    desc = " ∧ ".join(_describe_atom(a) for a in atoms if _cell_of(a) is not None)
    text = (
        f"local dependency (radius ≤ {radius}) across cells "
        f"{sorted(set(cells))}: {desc}"
    )
    return NaturalLanguageRule(
        "universal/spatial_locality", text,
        tuple(a for a in atoms if _cell_of(a) is not None),
    )


def mixed_polarity_clause(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """≥1 positive atom AND ≥1 negative atom in the clause.

    Detects "implication-shaped" rules: in DNF a class=1 leaf path with both
    polarities means "if these features are present AND those are absent,
    then class".  This is the loosest possible structural template — any
    non-trivial conditional rule has mixed polarity.
    """
    pos, neg = _by_polarity(atoms)
    if not pos or not neg:
        return None
    pos_desc = " ∧ ".join(_describe_atom(a) for a in pos)
    neg_desc = " ∧ ".join(_describe_atom(a) for a in neg)
    text = f"implication: ({pos_desc}) ∧ ¬({neg_desc})"
    return NaturalLanguageRule(
        "universal/mixed_polarity", text, tuple(pos + neg),
    )


def long_conjunction_clause(
    atoms: list[Atom],
    game: Any,
    min_atoms: int = 3,
) -> NaturalLanguageRule | None:
    """Any ≥``min_atoms`` clause that didn't match anything else.

    Last-resort catch-all to declare "the NN encoded a multi-feature
    dependency here", without claiming it's any known solving technique.
    """
    if len(atoms) < min_atoms:
        return None
    desc = " ∧ ".join(_describe_atom(a) for a in atoms)
    text = f"complex {len(atoms)}-atom dependency: {desc}"
    return NaturalLanguageRule("universal/complex", text, tuple(atoms))
