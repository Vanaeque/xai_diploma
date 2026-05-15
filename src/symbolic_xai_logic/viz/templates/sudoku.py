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


def naked_single(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Naked single — a cell with only one candidate digit left.

    Pattern: a single (r, c) appears with k ≥ 2 cell_digit atoms where exactly
    one polarity is positive and the rest are negative, each on a *different*
    digit.  The positive atom names the unique remaining digit; the negatives
    enumerate eliminations that locked it in.

    For the full game-theoretic naked single we'd need n−1 negative atoms
    (3 for sudoku4, 8 for sudoku9).  We accept any k ≥ 2 because that's still
    a stricter elimination than plain cell_uniqueness (which only proves one
    eliminated digit) — it's a *partial naked single* when k < n−1.

    Distinguishing it from cell_uniqueness: at least TWO negative same-cell
    atoms, all on different digits, plus one positive same-cell atom on a
    different digit again.
    """
    cd = [a for a in atoms if a.kind == "cell_digit"]
    # Group atoms by (row, col)
    from collections import defaultdict
    by_cell: dict[tuple[int, int], list[Atom]] = defaultdict(list)
    for a in cd:
        by_cell[(a.payload["row"], a.payload["col"])].append(a)

    for (r, c), group in by_cell.items():
        positives = [a for a in group if a.polarity]
        negatives = [a for a in group if not a.polarity]
        if len(positives) != 1 or len(negatives) < 2:
            continue
        pos_digit = positives[0].payload["digit"]
        neg_digits = [a.payload["digit"] for a in negatives]
        # All atoms in the group must be on different digits
        all_digits = neg_digits + [pos_digit]
        if len(set(all_digits)) != len(all_digits):
            continue
        eliminated = ", ".join(str(d) for d in sorted(set(neg_digits)))
        n = game.size
        suffix = (
            f"the only remaining candidate in cell ({r},{c})"
            if len(neg_digits) == n - 1
            else f"after eliminating {{{eliminated}}}, cell ({r},{c}) must hold {pos_digit}"
        )
        text = (
            f"naked single: with {{{eliminated}}} excluded for cell ({r},{c}), "
            f"digit {pos_digit} is forced"
            if len(neg_digits) < n - 1
            else f"naked single: cell ({r},{c}) holds {pos_digit} (all other digits eliminated)"
        )
        return NaturalLanguageRule("naked_single", text, tuple(positives + negatives))
    return None


def hidden_single(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Hidden single — a digit that fits in only one cell within a unit.

    Pattern: ≥2 cell_digit atoms with the SAME digit, all in the same row,
    column, or box, where exactly one polarity is positive and the rest are
    negative on distinct cells.  The positive atom names the unique cell in
    that unit that can hold the digit; the negatives are the eliminations.

    Strict form (n−1 negatives) means a full hidden single.  Partial forms
    (k < n−1) are still informative locked-candidate patterns.
    """
    cd = [a for a in atoms if a.kind == "cell_digit"]
    bs = int(math.isqrt(game.size)) if game.size in (4, 9, 16) else 0

    def box_id(r: int, c: int) -> int:
        return (r // bs) * bs + (c // bs) if bs else -1

    # Group by digit then test each unit class
    from collections import defaultdict
    by_digit: dict[int, list[Atom]] = defaultdict(list)
    for a in cd:
        by_digit[a.payload["digit"]].append(a)

    for digit, group in by_digit.items():
        positives = [a for a in group if a.polarity]
        negatives = [a for a in group if not a.polarity]
        if len(positives) != 1 or len(negatives) < 2:
            continue
        pos = positives[0]
        pr, pc = pos.payload["row"], pos.payload["col"]
        # Try each unit type
        for unit_name, unit_match in (
            ("row",    lambda a: a.payload["row"] == pr),
            ("column", lambda a: a.payload["col"] == pc),
            ("box",    lambda a: bs > 0 and box_id(a.payload["row"], a.payload["col"]) == box_id(pr, pc)),
        ):
            in_unit_negs = [a for a in negatives if unit_match(a)]
            if len(in_unit_negs) < 2:
                continue
            # Cells in the unit must all be distinct from the positive cell
            neg_cells = {(a.payload["row"], a.payload["col"]) for a in in_unit_negs}
            if (pr, pc) in neg_cells:
                continue
            cell_list = ", ".join(f"({a.payload['row']},{a.payload['col']})" for a in in_unit_negs)
            text = (
                f"hidden single: digit {digit} cannot go in {{{cell_list}}}, "
                f"so cell ({pr},{pc}) — the only remaining {unit_name} cell — must hold {digit}"
            )
            return NaturalLanguageRule(
                "hidden_single", text, tuple([pos] + in_unit_negs)
            )
    return None


def naked_pair(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Naked pair — two cells in the same unit share the same 2 candidate digits.

    Pattern: for the SAME digit d, two negative cell_digit atoms at two
    DIFFERENT cells (cellA, cellB) in the SAME unit (row/col/box), plus for
    another digit d', two negative atoms at the SAME two cells.  Standard
    naked-pair shape elides explicit positive atoms — the assertion is "d and
    d' are confined to these two cells, so other digits in those cells, and
    d/d' anywhere else in the unit, are excluded".

    We accept the partial form: ≥4 cell_digit atoms, all negative, on exactly
    2 distinct cells in the same unit, on exactly 2 distinct digits.  This
    is the surrogate-tree-friendly version that captures the pattern even
    when the cleanup eliminations don't all land in the same leaf path.
    """
    cd = [a for a in atoms if a.kind == "cell_digit"]
    negatives = [a for a in cd if not a.polarity]
    if len(negatives) < 4:
        return None

    cells = sorted({(a.payload["row"], a.payload["col"]) for a in negatives})
    digits = sorted({a.payload["digit"] for a in negatives})
    if len(cells) != 2 or len(digits) != 2:
        return None
    # All 4 (cell, digit) combinations must be represented exactly once
    pairs = {(a.payload["row"], a.payload["col"], a.payload["digit"]) for a in negatives}
    expected = {(r, c, d) for (r, c) in cells for d in digits}
    if pairs != expected:
        return None

    # Are the two cells in the same unit?
    (r1, c1), (r2, c2) = cells
    bs = int(math.isqrt(game.size)) if game.size in (4, 9, 16) else 0
    same_row = r1 == r2
    same_col = c1 == c2
    same_box = (bs > 0
                and (r1 // bs) == (r2 // bs)
                and (c1 // bs) == (c2 // bs))
    if not (same_row or same_col or same_box):
        return None
    unit = "row" if same_row else "column" if same_col else "box"

    d1, d2 = digits
    text = (
        f"naked pair: cells ({r1},{c1}) and ({r2},{c2}) in the same {unit} "
        f"are confined to digits {{{d1},{d2}}}, so neither holds any other digit"
    )
    return NaturalLanguageRule("naked_pair", text, tuple(negatives))


def pointing_pair(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Pointing pair / locked candidates — for digit d, all candidates in a box
    lie in a single row (or column), forcing d to be excluded from that row
    OUTSIDE the box.

    Pattern: same digit d, ≥3 cell_digit atoms, all negative, where the cells
    are: ≥1 cell in box B that is in row R outside box B (excluded), and ≥2
    cells in box B that are NOT in row R (also excluded).  The two-pronged
    elimination implicitly locks d to row-R-∩-box-B.

    We accept the surrogate-friendly relaxation: ≥3 negative same-digit atoms
    spanning two distinct rows where at least 2 share a box and at least one
    different-row atom is in the box's row-projection outside the box.
    """
    cd = [a for a in atoms if a.kind == "cell_digit"]
    if game.size not in (4, 9, 16):
        return None
    bs = int(math.isqrt(game.size))

    from collections import defaultdict
    by_digit: dict[int, list[Atom]] = defaultdict(list)
    for a in cd:
        if not a.polarity:
            by_digit[a.payload["digit"]].append(a)

    for digit, negs in by_digit.items():
        if len(negs) < 3:
            continue
        # Group by box
        by_box: dict[tuple[int, int], list[Atom]] = defaultdict(list)
        for a in negs:
            r, c = a.payload["row"], a.payload["col"]
            by_box[(r // bs, c // bs)].append(a)
        for (br, bc), box_negs in by_box.items():
            if len(box_negs) < 2:
                continue
            # Are the in-box negatives aligned to a single row OR single column?
            in_box_rows = {a.payload["row"] for a in box_negs}
            in_box_cols = {a.payload["col"] for a in box_negs}
            if len(in_box_rows) == 1:
                row = next(iter(in_box_rows))
                # Need at least one same-digit negative in the same row OUTSIDE this box
                outside = [
                    a for a in negs
                    if a.payload["row"] == row
                    and (a.payload["row"] // bs, a.payload["col"] // bs) != (br, bc)
                ]
                if outside:
                    text = (
                        f"pointing pair: digit {digit} in box ({br},{bc}) is "
                        f"confined to row {row}, so it cannot appear in row {row} "
                        f"outside that box (e.g. cell "
                        f"({outside[0].payload['row']},{outside[0].payload['col']}))"
                    )
                    return NaturalLanguageRule("pointing_pair", text, tuple(box_negs + outside))
            if len(in_box_cols) == 1:
                col = next(iter(in_box_cols))
                outside = [
                    a for a in negs
                    if a.payload["col"] == col
                    and (a.payload["row"] // bs, a.payload["col"] // bs) != (br, bc)
                ]
                if outside:
                    text = (
                        f"pointing pair: digit {digit} in box ({br},{bc}) is "
                        f"confined to column {col}, so it cannot appear in "
                        f"column {col} outside that box (e.g. cell "
                        f"({outside[0].payload['row']},{outside[0].payload['col']}))"
                    )
                    return NaturalLanguageRule("pointing_pair", text, tuple(box_negs + outside))
    return None


def partial_naked_single(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Relaxed naked single — ≥2 same-cell negative atoms on different digits,
    without requiring an explicit positive atom for the remaining digit.

    Catches the "the NN learned that several digits are eliminated for this
    cell" pattern even when the corresponding positive atom didn't end up in
    the same leaf path.  Strictly weaker than naked_single but stronger than
    cell_uniqueness (≥2 negatives, not just 1).
    """
    cd = [a for a in atoms if a.kind == "cell_digit"]
    from collections import defaultdict
    by_cell: dict[tuple[int, int], list[Atom]] = defaultdict(list)
    for a in cd:
        if not a.polarity:
            by_cell[(a.payload["row"], a.payload["col"])].append(a)
    for (r, c), negs in by_cell.items():
        digits = {a.payload["digit"] for a in negs}
        if len(digits) >= 2:
            elim = ", ".join(str(d) for d in sorted(digits))
            text = (
                f"partial naked single: cell ({r},{c}) cannot hold any of "
                f"{{{elim}}}"
            )
            return NaturalLanguageRule("partial_naked_single", text, tuple(negs))
    return None


def partial_hidden_single(atoms: list[Atom], game: Any) -> NaturalLanguageRule | None:
    """
    Relaxed hidden single — ≥2 same-digit negative atoms on different cells
    that share a row, column, or box.  No positive atom required.

    Picks up "digit d is eliminated from several cells of the same unit"
    even when the surrogate didn't surface the locked positive cell.
    """
    cd = [a for a in atoms if a.kind == "cell_digit"]
    bs = int(math.isqrt(game.size)) if game.size in (4, 9, 16) else 0

    from collections import defaultdict
    by_digit: dict[int, list[Atom]] = defaultdict(list)
    for a in cd:
        if not a.polarity:
            by_digit[a.payload["digit"]].append(a)

    for digit, negs in by_digit.items():
        if len(negs) < 2:
            continue
        cells = [(a.payload["row"], a.payload["col"]) for a in negs]
        # Try unit cohorts
        for unit_name, group_key in (
            ("row",    lambda rc: rc[0]),
            ("column", lambda rc: rc[1]),
            ("box",    lambda rc: (rc[0] // bs, rc[1] // bs) if bs else None),
        ):
            from collections import Counter
            counts = Counter(group_key(rc) for rc in cells if group_key(rc) is not None)
            for key, n in counts.items():
                if n >= 2:
                    members = [a for a in negs if group_key((a.payload["row"], a.payload["col"])) == key]
                    cell_list = ", ".join(
                        f"({a.payload['row']},{a.payload['col']})" for a in members
                    )
                    text = (
                        f"partial hidden single: digit {digit} cannot appear in "
                        f"{{{cell_list}}} (same {unit_name})"
                    )
                    return NaturalLanguageRule("partial_hidden_single", text, tuple(members))
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
