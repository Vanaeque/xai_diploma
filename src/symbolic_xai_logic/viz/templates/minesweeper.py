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


def zero_safe_neighbours(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Pattern: exactly one "shows 0" atom (channel 2) at cell (r,c), plus ≥1
    negated-hidden atoms (channel 0, polarity False) at neighbours of (r,c).

    Semantics: when a cell shows 0, every adjacent cell is provably safe
    (no mine can be there).  This is the simplest, most-fired minesweeper
    rule and is what tree-cascading expansions ride on.  Doesn't require
    the full neighbourhood — even a 2-atom version (shows-0 + one safe
    neighbour) is the canonical kernel of cascade unfolding.
    """
    cs = [a for a in atoms if a.kind == "cell_state"]

    shows_zero = [
        a for a in cs
        if a.payload["channel"] == _CH_NUMBER_BASE  # channel 2 == "shows 0"
        and a.polarity
    ]
    if len(shows_zero) != 1:
        return None
    s = shows_zero[0]
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)

    safe_atoms = [
        a for a in cs
        if a.payload["channel"] == _CH_HIDDEN
        and not a.polarity                              # NOT hidden = revealed / safe
        and (a.payload["row"], a.payload["col"]) in nbrs
    ]
    if not safe_atoms:
        return None

    safe_cells = ", ".join(
        f"({a.payload['row']},{a.payload['col']})" for a in safe_atoms
    )
    text = (
        f"if cell ({cr},{cc}) shows 0, then neighbour{'s' if len(safe_atoms) > 1 else ''} "
        f"{{{safe_cells}}} {'are' if len(safe_atoms) > 1 else 'is'} safe"
    )
    return NaturalLanguageRule("zero_safe_neighbours", text, tuple([s] + safe_atoms))


def flagged_satisfies_clue(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Pattern: exactly one "shows N" atom at (r,c) with N ≥ 1, plus ≥N
    flagged atoms (channel 1, polarity True) at N distinct neighbours of
    (r,c), plus ≥1 negated-hidden atom at OTHER neighbours of (r,c).

    Semantics: the clue is *already satisfied* by existing flags, so every
    other unrevealed neighbour can be safely opened.  Strictly stronger
    than local_count because it tolerates extra context atoms and only
    requires ≥1 inferred-safe neighbour, not all of them.

    Critical companion to local_exhaustion — together they cover the
    two saturated regimes (clue == flags vs clue == hidden + flags).
    """
    cs = [a for a in atoms if a.kind == "cell_state"]

    shows = [
        a for a in cs
        if _CH_NUMBER_BASE <= a.payload["channel"] <= 10 and a.polarity
    ]
    if len(shows) != 1:
        return None
    s = shows[0]
    n_mines = s.payload["channel"] - _CH_NUMBER_BASE
    if n_mines < 1:
        return None
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)

    flagged = [
        a for a in cs
        if a.payload["channel"] == _CH_FLAGGED
        and a.polarity
        and (a.payload["row"], a.payload["col"]) in nbrs
    ]
    flagged_cells = {(a.payload["row"], a.payload["col"]) for a in flagged}
    if len(flagged_cells) < n_mines:
        return None

    # At least one *other* neighbour proven safe
    safe = [
        a for a in cs
        if a.payload["channel"] == _CH_HIDDEN
        and not a.polarity
        and (a.payload["row"], a.payload["col"]) in nbrs
        and (a.payload["row"], a.payload["col"]) not in flagged_cells
    ]
    if not safe:
        return None

    safe_cells_str = ", ".join(
        f"({a.payload['row']},{a.payload['col']})" for a in safe
    )
    text = (
        f"if cell ({cr},{cc}) shows {n_mines} and {n_mines} of its neighbours "
        f"are already flagged, then unrevealed neighbour{'s' if len(safe) > 1 else ''} "
        f"{{{safe_cells_str}}} {'are' if len(safe) > 1 else 'is'} safe"
    )
    return NaturalLanguageRule("flagged_satisfies_clue", text, tuple([s] + flagged + safe))


def two_clue_chain(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Two clues at adjacent revealed cells whose neighbourhoods overlap.

    Pattern: ≥2 positive shows-N atoms at distinct cells (r1,c1) and
    (r2,c2) that are themselves neighbours (Chebyshev distance ≤ 2), plus
    ≥1 hidden-channel atom in the shared neighbourhood.

    Captures the "chord" / "1-2-1" / "1-2-X" family of minesweeper
    deductions — combining two clues to constrain shared unknowns.  Doesn't
    enforce the specific arithmetic (that's local_count/local_exhaustion
    territory), only the geometric shape.
    """
    cs = [a for a in atoms if a.kind == "cell_state"]
    clues = [
        a for a in cs
        if _CH_NUMBER_BASE <= a.payload["channel"] <= 10 and a.polarity
    ]
    if len(clues) < 2:
        return None
    # Find an adjacent pair of clues
    for i in range(len(clues)):
        for j in range(i + 1, len(clues)):
            r1, c1 = clues[i].payload["row"], clues[i].payload["col"]
            r2, c2 = clues[j].payload["row"], clues[j].payload["col"]
            if max(abs(r1 - r2), abs(c1 - c2)) > 2:
                continue
            nbrs1 = _nbrs(game, r1, c1)
            nbrs2 = _nbrs(game, r2, c2)
            shared = nbrs1 & nbrs2
            if not shared:
                continue
            hidden_in_shared = [
                a for a in cs
                if a.payload["channel"] == _CH_HIDDEN
                and (a.payload["row"], a.payload["col"]) in shared
            ]
            if not hidden_in_shared:
                continue
            n1 = clues[i].payload["channel"] - _CH_NUMBER_BASE
            n2 = clues[j].payload["channel"] - _CH_NUMBER_BASE
            text = (
                f"two-clue chain: clues ({r1},{c1})={n1} and ({r2},{c2})={n2} "
                f"share {len(shared)} neighbour cell{'s' if len(shared) > 1 else ''}, "
                f"constraining their mine pattern"
            )
            return NaturalLanguageRule(
                "two_clue_chain", text,
                tuple([clues[i], clues[j]] + hidden_in_shared),
            )
    return None


def safe_low_clue(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    A low-numbered clue (0–2) plus already-revealed neighbours strongly
    suggests the remaining unknowns are safer than average.

    Pattern: exactly one shows-N atom with N ∈ {0, 1, 2}, plus ≥(|nbrs|−N)
    negated-hidden atoms (= revealed neighbours).  This is the partial
    version of flagged_satisfies_clue / local_count where flags aren't
    explicit but enough cells are revealed.
    """
    cs = [a for a in atoms if a.kind == "cell_state"]
    shows = [
        a for a in cs
        if _CH_NUMBER_BASE <= a.payload["channel"] <= _CH_NUMBER_BASE + 2
        and a.polarity
    ]
    if len(shows) != 1:
        return None
    s = shows[0]
    n_mines = s.payload["channel"] - _CH_NUMBER_BASE
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)
    revealed = [
        a for a in cs
        if a.payload["channel"] == _CH_HIDDEN
        and not a.polarity
        and (a.payload["row"], a.payload["col"]) in nbrs
    ]
    if len(revealed) < max(1, len(nbrs) - n_mines - 1):
        # Not enough revealed neighbours to make a strong safety claim
        return None
    text = (
        f"safe low clue: cell ({cr},{cc}) shows {n_mines} and most of its "
        f"neighbours are revealed, so the rest are probably safe"
    )
    return NaturalLanguageRule("safe_low_clue", text, tuple([s] + revealed))


def mine_high_clue(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    A high-numbered clue (≥3) with many hidden neighbours says the hidden
    ones are mine-dense.

    Pattern: exactly one shows-N atom with N ≥ 3, plus ≥1 positive-hidden
    atom at a neighbour.  Partial form of local_exhaustion — without
    requiring N == count(hidden), it still indicates the hidden cell is
    mine-likely under the clue's pressure.
    """
    cs = [a for a in atoms if a.kind == "cell_state"]
    shows = [
        a for a in cs
        if _CH_NUMBER_BASE + 3 <= a.payload["channel"] <= 10
        and a.polarity
    ]
    if len(shows) != 1:
        return None
    s = shows[0]
    n_mines = s.payload["channel"] - _CH_NUMBER_BASE
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)
    hidden = [
        a for a in cs
        if a.payload["channel"] == _CH_HIDDEN
        and a.polarity
        and (a.payload["row"], a.payload["col"]) in nbrs
    ]
    if not hidden:
        return None
    cell_list = ", ".join(
        f"({a.payload['row']},{a.payload['col']})" for a in hidden
    )
    text = (
        f"mine-pressure clue: cell ({cr},{cc}) shows {n_mines}, with hidden "
        f"neighbour{'s' if len(hidden) > 1 else ''} {{{cell_list}}} under high "
        f"mine-pressure"
    )
    return NaturalLanguageRule("mine_high_clue", text, tuple([s] + hidden))


def neighbour_clue_signal(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Loosest minesweeper template — catches what the depth-4 trees actually
    learn: "a clue at one neighbouring cell + a hidden/revealed atom at
    another nearby cell is predictive of the target cell's mine status."

    Pattern: at least one positive "shows-N" atom at SOME cell (the clue),
    plus at least one "hidden"-channel atom (positive or negative) at a
    DIFFERENT cell that is a neighbour of the clue cell.

    This is a relaxed local_count/local_exhaustion — we don't require the
    exact count match, just that the clue and the unknown cell are
    geometrically adjacent (logical-form precondition for ANY minesweeper
    deduction).  Catches the bulk of what NN-distilled trees produce when
    they correctly localise their reasoning.
    """
    cs = [a for a in atoms if a.kind == "cell_state"]
    clues = [
        a for a in cs
        if _CH_NUMBER_BASE <= a.payload["channel"] <= 10 and a.polarity
    ]
    hidden_like = [a for a in cs if a.payload["channel"] == _CH_HIDDEN]

    for clue in clues:
        cr, cc = clue["row"] if False else clue.payload["row"], clue.payload["col"]
        nbrs = _nbrs(game, cr, cc)
        adjacent = [
            a for a in hidden_like
            if (a.payload["row"], a.payload["col"]) in nbrs
        ]
        if not adjacent:
            continue
        n_mines = clue.payload["channel"] - _CH_NUMBER_BASE
        adj_cells = ", ".join(
            f"({a.payload['row']},{a.payload['col']})" for a in adjacent
        )
        # Polarity-aware text: positive hidden = "is hidden", negative = "is revealed".
        text = (
            f"clue at ({cr},{cc}) showing {n_mines} constrains the mine "
            f"probability of adjacent cell{'s' if len(adjacent) > 1 else ''} "
            f"{{{adj_cells}}}"
        )
        return NaturalLanguageRule("neighbour_clue_signal", text, tuple([clue] + adjacent))
    return None


def isolated_clue(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Pattern: exactly one "shows N" atom at (r,c) with N ≥ 1, plus ≥(|nbrs|−N)
    negated-hidden atoms (channel 0, polarity False) at distinct neighbours
    of (r,c).  When too many neighbours are already revealed-safe, the
    remaining hidden ones must hold ALL N mines — symmetric counterpart
    to local_exhaustion stated via the safe-cell complement.

    Captures cases where the surrogate tree's path counted out the safe
    neighbours rather than the mine neighbours.
    """
    cs = [a for a in atoms if a.kind == "cell_state"]

    shows = [
        a for a in cs
        if _CH_NUMBER_BASE <= a.payload["channel"] <= 10 and a.polarity
    ]
    if len(shows) != 1:
        return None
    s = shows[0]
    n_mines = s.payload["channel"] - _CH_NUMBER_BASE
    if n_mines < 1:
        return None
    cr, cc = s.payload["row"], s.payload["col"]
    nbrs = _nbrs(game, cr, cc)

    safe = [
        a for a in cs
        if a.payload["channel"] == _CH_HIDDEN
        and not a.polarity
        and (a.payload["row"], a.payload["col"]) in nbrs
    ]
    safe_cells = {(a.payload["row"], a.payload["col"]) for a in safe}
    if len(safe_cells) < len(nbrs) - n_mines:
        return None
    remaining = nbrs - safe_cells
    if len(remaining) != n_mines:
        return None

    rem_str = ", ".join(f"({r},{c})" for (r, c) in sorted(remaining))
    text = (
        f"if cell ({cr},{cc}) shows {n_mines} and only {n_mines} of its "
        f"neighbours remain unrevealed (the rest are safe), then {{{rem_str}}} "
        f"{'are' if n_mines > 1 else 'is'} mine{'s' if n_mines > 1 else ''}"
    )
    return NaturalLanguageRule("isolated_clue", text, tuple([s] + safe))
