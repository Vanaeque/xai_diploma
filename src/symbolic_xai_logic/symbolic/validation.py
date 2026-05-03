"""Z3-based logical validation of extracted canonical rules (Task P1-10).

A canonical rule extracted by ``RuleExtractor`` has a *form* that matches one
of the registered template families (e.g. ``row_uniqueness``).  But a clause
can match a template *syntactically* and still be logically wrong — for
example "if cell (0,0) holds 1 then cell (1,1) does not hold 2" syntactically
fits a 2-cell-digit pattern but isn't entailed by Sudoku rules at all.

This module asks Z3 whether each canonical clause is *valid* under the game's
constraint set.  A clause C is valid iff under the conjunction of game
constraints, C is logically entailed — equivalently, the conjunction
G ∧ ¬C is unsatisfiable.

Returns
-------
canonical_valid_rate            : valid_clauses / total_clauses
canonical_false_positive_rate   : invalid_clauses / canonical_form_clauses

We only validate rules whose ``template`` is a known family — "unknown"
templates are left out of the denominator (they were never going to be
canonically meaningful in the first place).
"""
from __future__ import annotations
from typing import Any


def _sudoku_solver_with_constraints(game: Any):
    """Build a fresh Z3 solver loaded with sudoku rules of the given size.

    The cells are unbound — we'll add the rule literal(s) at validation time.
    Returns (solver, cells_2d) or (None, None) if z3 is not installed.
    """
    try:
        from z3 import Int, Distinct, Solver
    except ImportError:
        return None, None

    n = game.size
    bs = game.box_size
    cells = [[Int(f"x_{r}_{c}") for c in range(n)] for r in range(n)]
    s = Solver()
    for r in range(n):
        for c in range(n):
            s.add(cells[r][c] >= 1, cells[r][c] <= n)
    for r in range(n):
        s.add(Distinct(cells[r]))
    for c in range(n):
        s.add(Distinct([cells[r][c] for r in range(n)]))
    for br in range(0, n, bs):
        for bc in range(0, n, bs):
            box = [cells[br + dr][bc + dc] for dr in range(bs) for dc in range(bs)]
            s.add(Distinct(box))
    return s, cells


def _atom_to_z3_predicate(atom_dict: dict, cells_2d):
    """Translate a serialised cell_digit atom to a Z3 predicate.

    ``atom_dict`` is a plain dict (as stored by extract_all_canonical_rules):
        {"kind": "cell_digit", "payload": {"row": r, "col": c, "digit": d}, "polarity": True/False}

    Polarity True  → cell == digit
    Polarity False → cell != digit
    """
    if atom_dict.get("kind") != "cell_digit":
        return None
    p = atom_dict["payload"]
    cell_var = cells_2d[p["row"]][p["col"]]
    eq = (cell_var == p["digit"])
    if atom_dict["polarity"]:
        return eq
    # Wrap in z3.Not — imported lazily so non-z3 paths don't break
    from z3 import Not
    return Not(eq)


def validate_sudoku_rule(game: Any, atoms: list[dict]) -> bool | None:
    """Check whether a 2-atom Sudoku canonical clause is logically valid.

    The clause we extracted is a conjunction of literals (e.g.
    ``cell(r,c1)=d ∧ ¬cell(r,c2)=d``).  This conjunction is "valid" in the
    XAI sense if its truth is consistent with — and entailed by — Sudoku
    rules.  Concretely, for the cell_uniqueness / row_uniqueness /
    column_uniqueness / box_uniqueness families, the positive literal alone
    forces the negated literal under Sudoku rules.  We test that by asking
    Z3: under sudoku ∧ positive_literal, can the negated literal's *positive
    form* still hold?  If unsat → the negation is forced → rule is valid.

    Returns
    -------
    True   : rule is logically valid under Sudoku rules
    False  : rule is NOT entailed (Z3 found a satisfying assignment that
             violates it)
    None   : Z3 unavailable or rule shape unsupported
    """
    if not atoms or len(atoms) != 2:
        return None
    s, cells = _sudoku_solver_with_constraints(game)
    if s is None:
        return None

    pos = next((a for a in atoms if a.get("polarity")), None)
    neg = next((a for a in atoms if not a.get("polarity")), None)
    if pos is None or neg is None:
        return None
    if pos.get("kind") != "cell_digit" or neg.get("kind") != "cell_digit":
        return None

    pos_pred = _atom_to_z3_predicate(pos, cells)
    # The clause asserts: pos AND ¬neg_inner.  It is valid iff under sudoku ∧ pos,
    # neg_inner cannot hold.  Equivalent: sudoku ∧ pos ∧ neg_inner is UNSAT.
    neg_inner = _atom_to_z3_predicate(
        {**neg, "polarity": True}, cells   # un-negate to get the inner predicate
    )
    s.add(pos_pred)
    s.add(neg_inner)
    from z3 import sat
    result = s.check()
    return result != sat  # unsat → rule is logically forced


def validate_canonical_rules(game: Any, rules: list[dict]) -> dict:
    """Validate every canonical rule in ``rules``.

    Returns a stats dict with:
      n_canonical_form  : rules whose template is a known family
      n_valid           : valid under Z3
      n_invalid         : explicitly invalidated by Z3
      n_unknown_check   : Z3 unavailable / unsupported shape
      canonical_valid_rate            : n_valid / n_canonical_form
      canonical_false_positive_rate   : n_invalid / n_canonical_form
    """
    from ..games.sudoku import SudokuGame

    is_sudoku = isinstance(game, SudokuGame)
    n_form = 0
    n_valid = 0
    n_invalid = 0
    n_unknown = 0

    for r in rules:
        if r.get("template") in (None, "unknown", ""):
            continue
        n_form += 1
        atoms = r.get("atoms")
        if not atoms:
            n_unknown += 1
            continue
        verdict: bool | None = None
        if is_sudoku:
            verdict = validate_sudoku_rule(game, atoms)
        # Minesweeper validation not implemented yet — clauses are too long
        # for the 2-atom z3 setup above.  Marked as "unknown" rather than
        # silently passing as valid.
        if verdict is True:
            n_valid += 1
        elif verdict is False:
            n_invalid += 1
        else:
            n_unknown += 1

    rate_valid = (n_valid / n_form) if n_form else None
    rate_fp = (n_invalid / n_form) if n_form else None

    return {
        "n_canonical_form": n_form,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_unknown_check": n_unknown,
        "canonical_valid_rate": rate_valid,
        "canonical_false_positive_rate": rate_fp,
    }
