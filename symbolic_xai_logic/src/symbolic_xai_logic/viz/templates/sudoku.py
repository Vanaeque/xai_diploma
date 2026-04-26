"""Canonical template matchers for Sudoku rules.

Each matcher searches for a canonical 2-atom sub-pattern inside the full clause.
Extra atoms in the clause are treated as context; the `atoms` field of the
returned NaturalLanguageRule contains ONLY the matched pair for traceability.

False positives are not acceptable; false negatives remain acceptable.
"""
from __future__ import annotations
import math
from itertools import combinations
from typing import Any, Iterable

from .atom import Atom, NaturalLanguageRule


def _iter_pairs(atoms: list[Atom]) -> Iterable[tuple[Atom, Atom]]:
    cd = [a for a in atoms if a.kind == "cell_digit"]
    for i in range(len(cd)):
        for j in range(i + 1, len(cd)):
            yield cd[i], cd[j]


def cell_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r,c) holds digit d_pos, it does not hold digit d_neg.

    Sub-pattern: 2 cell_digit atoms, same (row, col), different digits, mixed polarity.
    """
    for a, b in _iter_pairs(atoms):
        if a.payload["row"] != b.payload["row"] or a.payload["col"] != b.payload["col"]:
            continue
        if a.payload["digit"] == b.payload["digit"]:
            continue
        if a.polarity == b.polarity:
            continue
        pos = a if a.polarity else b
        neg = b if a.polarity else a
        r, c = pos.payload["row"], pos.payload["col"]
        text = (
            f"if cell ({r},{c}) holds {pos.payload['digit']}, "
            f"it does not hold {neg.payload['digit']}"
        )
        # TODO: support multi-rule clauses (return first match only for now)
        return NaturalLanguageRule("cell_uniqueness", text, (pos, neg))
    return None


def row_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r,c1) holds d, then cell (r,c2) does not hold d.

    Sub-pattern: 2 cell_digit atoms, same row and digit, different columns, mixed polarity.
    """
    for a, b in _iter_pairs(atoms):
        if a.payload["row"] != b.payload["row"]:
            continue
        if a.payload["digit"] != b.payload["digit"]:
            continue
        if a.payload["col"] == b.payload["col"]:
            continue
        if a.polarity == b.polarity:
            continue
        pos = a if a.polarity else b
        neg = b if a.polarity else a
        r = pos.payload["row"]
        text = (
            f"if cell ({r},{pos.payload['col']}) holds {pos.payload['digit']}, "
            f"then cell ({r},{neg.payload['col']}) does not hold {pos.payload['digit']}"
        )
        # TODO: support multi-rule clauses
        return NaturalLanguageRule("row_uniqueness", text, (pos, neg))
    return None


def column_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r1,c) holds d, then cell (r2,c) does not hold d.

    Sub-pattern: 2 cell_digit atoms, same column and digit, different rows, mixed polarity.
    """
    for a, b in _iter_pairs(atoms):
        if a.payload["col"] != b.payload["col"]:
            continue
        if a.payload["digit"] != b.payload["digit"]:
            continue
        if a.payload["row"] == b.payload["row"]:
            continue
        if a.polarity == b.polarity:
            continue
        pos = a if a.polarity else b
        neg = b if a.polarity else a
        c = pos.payload["col"]
        text = (
            f"if cell ({pos.payload['row']},{c}) holds {pos.payload['digit']}, "
            f"then cell ({neg.payload['row']},{c}) does not hold {pos.payload['digit']}"
        )
        # TODO: support multi-rule clauses
        return NaturalLanguageRule("column_uniqueness", text, (pos, neg))
    return None


def box_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r1,c1) holds d, then cell (r2,c2) in the same box does not hold d.

    Sub-pattern: 2 cell_digit atoms, same digit, same √n×√n box, different cells, mixed polarity.
    Only defined for game.size in (4, 9, 16).
    """
    if game.size not in (4, 9, 16):
        return None
    bs = int(math.isqrt(game.size))

    def box_id(r: int, c: int) -> int:
        return (r // bs) * bs + (c // bs)

    for a, b in _iter_pairs(atoms):
        if a.payload["digit"] != b.payload["digit"]:
            continue
        if a.payload["row"] == b.payload["row"] and a.payload["col"] == b.payload["col"]:
            continue
        if a.polarity == b.polarity:
            continue
        if box_id(a.payload["row"], a.payload["col"]) != box_id(b.payload["row"], b.payload["col"]):
            continue
        pos = a if a.polarity else b
        neg = b if a.polarity else a
        text = (
            f"if cell ({pos.payload['row']},{pos.payload['col']}) holds {pos.payload['digit']}, "
            f"then cell ({neg.payload['row']},{neg.payload['col']}) in the same box "
            f"does not hold {pos.payload['digit']}"
        )
        # TODO: support multi-rule clauses
        return NaturalLanguageRule("box_uniqueness", text, (pos, neg))
    return None
