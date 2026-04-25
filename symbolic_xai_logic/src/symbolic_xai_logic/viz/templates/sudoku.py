"""Canonical template matchers for Sudoku rules.

Each matcher takes a list[Atom] (one DNF conjunction) and the game object.
Returns NaturalLanguageRule if the pattern matches, else None.

Patterns are strict: every false negative is acceptable; false positives are not.
"""
from __future__ import annotations
import math
from typing import Any

from .atom import Atom, NaturalLanguageRule


def _cell_digit_atoms(atoms: list[Atom]) -> list[Atom]:
    return [a for a in atoms if a.kind == "cell_digit"]


def cell_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r,c) holds digit d_pos, it does not hold digit d_neg.

    Pattern: exactly 2 cell_digit atoms, same (row, col), different digits,
    one polarity True and one False.
    """
    cd = _cell_digit_atoms(atoms)
    if len(cd) != 2 or len(atoms) != 2:
        return None
    a, b = cd
    if a.payload["row"] != b.payload["row"] or a.payload["col"] != b.payload["col"]:
        return None
    if a.payload["digit"] == b.payload["digit"]:
        return None
    if a.polarity == b.polarity:
        return None
    pos = a if a.polarity else b
    neg = b if a.polarity else a
    r, c = pos.payload["row"], pos.payload["col"]
    text = (
        f"if cell ({r},{c}) holds {pos.payload['digit']}, "
        f"it does not hold {neg.payload['digit']}"
    )
    return NaturalLanguageRule("cell_uniqueness", text, tuple(atoms))


def row_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r,c1) holds d, then cell (r,c2) does not hold d.

    Pattern: exactly 2 cell_digit atoms, same row and digit, different columns,
    one polarity True and one False.
    """
    cd = _cell_digit_atoms(atoms)
    if len(cd) != 2 or len(atoms) != 2:
        return None
    a, b = cd
    if a.payload["row"] != b.payload["row"]:
        return None
    if a.payload["digit"] != b.payload["digit"]:
        return None
    if a.payload["col"] == b.payload["col"]:
        return None
    if a.polarity == b.polarity:
        return None
    pos = a if a.polarity else b
    neg = b if a.polarity else a
    r = pos.payload["row"]
    text = (
        f"if cell ({r},{pos.payload['col']}) holds {pos.payload['digit']}, "
        f"then cell ({r},{neg.payload['col']}) does not hold {pos.payload['digit']}"
    )
    return NaturalLanguageRule("row_uniqueness", text, tuple(atoms))


def column_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r1,c) holds d, then cell (r2,c) does not hold d.

    Pattern: exactly 2 cell_digit atoms, same column and digit, different rows,
    one polarity True and one False.
    """
    cd = _cell_digit_atoms(atoms)
    if len(cd) != 2 or len(atoms) != 2:
        return None
    a, b = cd
    if a.payload["col"] != b.payload["col"]:
        return None
    if a.payload["digit"] != b.payload["digit"]:
        return None
    if a.payload["row"] == b.payload["row"]:
        return None
    if a.polarity == b.polarity:
        return None
    pos = a if a.polarity else b
    neg = b if a.polarity else a
    c = pos.payload["col"]
    text = (
        f"if cell ({pos.payload['row']},{c}) holds {pos.payload['digit']}, "
        f"then cell ({neg.payload['row']},{c}) does not hold {pos.payload['digit']}"
    )
    return NaturalLanguageRule("column_uniqueness", text, tuple(atoms))


def box_uniqueness(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    If cell (r1,c1) holds d, then cell (r2,c2) in the same box does not hold d.

    Pattern: exactly 2 cell_digit atoms, same digit, same √n×√n box, different cells,
    one polarity True and one False.
    Only defined for game.size in (4, 9, 16).
    """
    if game.size not in (4, 9, 16):
        return None
    bs = int(math.isqrt(game.size))

    cd = _cell_digit_atoms(atoms)
    if len(cd) != 2 or len(atoms) != 2:
        return None
    a, b = cd
    if a.payload["digit"] != b.payload["digit"]:
        return None
    if a.payload["row"] == b.payload["row"] and a.payload["col"] == b.payload["col"]:
        return None
    if a.polarity == b.polarity:
        return None

    def box_id(r: int, c: int) -> int:
        return (r // bs) * bs + (c // bs)

    if box_id(a.payload["row"], a.payload["col"]) != box_id(b.payload["row"], b.payload["col"]):
        return None

    pos = a if a.polarity else b
    neg = b if a.polarity else a
    text = (
        f"if cell ({pos.payload['row']},{pos.payload['col']}) holds {pos.payload['digit']}, "
        f"then cell ({neg.payload['row']},{neg.payload['col']}) in the same box "
        f"does not hold {pos.payload['digit']}"
    )
    return NaturalLanguageRule("box_uniqueness", text, tuple(atoms))
