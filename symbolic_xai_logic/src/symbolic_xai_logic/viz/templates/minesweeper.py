"""Canonical template matchers for Minesweeper rules.

Channel semantics (N_SPATIAL_CHANNELS = 11):
  0  = hidden (unrevealed cell)
  1  = flagged
  2  = revealed, shows 0
  3  = revealed, shows 1
  ...
  10 = revealed, shows 8

local_count     : shows-N cell + N flagged neighbors + NOT-hidden remaining neighbors
local_exhaustion: shows-N cell + exactly N hidden neighbors (all must be mines)
"""
from __future__ import annotations
from typing import Any

from .atom import Atom, NaturalLanguageRule

_CH_HIDDEN = 0
_CH_FLAGGED = 1
_CH_NUMBER_BASE = 2


def _nbrs(game: Any, r: int, c: int) -> set[tuple[int, int]]:
    return set(game._neighbours(r, c))


def local_count(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Pattern: exactly one "shows N" atom (ch 2..10, polarity True) at cell (r,c),
    plus N flagged atoms (ch 1, polarity True) at N neighbors of (r,c),
    plus (|neighbors| − N) negated-hidden atoms (ch 0, polarity False) at the
    remaining neighbors.

    Extra non-cell_state atoms are treated as background context and ignored.
    Text: "if cell (r,c) shows N and N of its neighbors are mines, the others are safe"
    """
    cs = [a for a in atoms if a.kind == "cell_state"]

    shows = [a for a in cs if _CH_NUMBER_BASE <= a.payload["channel"] <= 10 and a.polarity]
    if len(shows) != 1:
        return None
    s = shows[0]
    n_mines = s.payload["channel"] - _CH_NUMBER_BASE
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)

    flagged = [a for a in cs if a.payload["channel"] == _CH_FLAGGED and a.polarity]
    if len(flagged) != n_mines:
        return None
    flagged_cells = {(a.payload["row"], a.payload["col"]) for a in flagged}
    if not flagged_cells.issubset(nbrs):
        return None

    remaining = nbrs - flagged_cells
    neg_hidden = [a for a in cs if a.payload["channel"] == _CH_HIDDEN and not a.polarity]
    if len(neg_hidden) != len(remaining):
        return None
    neg_hidden_cells = {(a.payload["row"], a.payload["col"]) for a in neg_hidden}
    if neg_hidden_cells != remaining:
        return None

    text = (
        f"if cell ({cr},{cc}) shows {n_mines} and {n_mines} of its neighbors are mines, "
        f"the others are safe"
    )
    return NaturalLanguageRule("local_count", text, tuple(atoms))


def local_exhaustion(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Pattern: exactly one "shows N" atom (ch 2..10, polarity True) at cell (r,c),
    plus exactly N hidden atoms (ch 0, polarity True) at N neighbors of (r,c).

    Extra non-cell_state atoms are treated as background context and ignored.
    This is the saturated-clue rule: the clue equals the unknown-neighbor count,
    so all unknown neighbors must be mines.

    Text: "if cell (r,c) shows N and only N unknown neighbors remain that account
           for the missing mines, all N are mines"
    """
    cs = [a for a in atoms if a.kind == "cell_state"]

    shows = [a for a in cs if _CH_NUMBER_BASE <= a.payload["channel"] <= 10 and a.polarity]
    if len(shows) != 1:
        return None
    s = shows[0]
    n_mines = s.payload["channel"] - _CH_NUMBER_BASE
    if n_mines == 0:
        return None  # shows-0 is handled by local_count (all safe); exhaustion needs ≥1 mine
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)

    hidden = [a for a in cs if a.payload["channel"] == _CH_HIDDEN and a.polarity]
    if len(hidden) != n_mines:
        return None
    hidden_cells = {(a.payload["row"], a.payload["col"]) for a in hidden}
    if not hidden_cells.issubset(nbrs):
        return None
    # TODO: support multi-rule clauses

    text = (
        f"if cell ({cr},{cc}) shows {n_mines} and only {n_mines} unknown "
        f"neighbor{'s' if n_mines > 1 else ''} remain that account for the missing "
        f"mine{'s' if n_mines > 1 else ''}, all {n_mines} are mines"
    )
    return NaturalLanguageRule("local_exhaustion", text, tuple(atoms))
