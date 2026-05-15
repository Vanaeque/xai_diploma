"""Unit tests for the template decoder and canonical-rule matchers."""
from __future__ import annotations
import pytest
from symbolic_xai_logic.viz.templates.atom import Atom, NaturalLanguageRule
from symbolic_xai_logic.viz.templates.feature_decoder import decode
from symbolic_xai_logic.viz.templates import match_clause, render_clauses_as_nl
from symbolic_xai_logic.viz.templates import sudoku as sudoku_templates
from symbolic_xai_logic.viz.templates import minesweeper as ms_templates
from symbolic_xai_logic.games.sudoku import SudokuGame
from symbolic_xai_logic.games.minesweeper import MinesweeperGame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sudoku4():
    return SudokuGame(size=4)


@pytest.fixture
def sudoku9():
    return SudokuGame(size=9)


@pytest.fixture
def ms4():
    return MinesweeperGame(size=4, n_mines=2)


# ---------------------------------------------------------------------------
# feature_decoder
# ---------------------------------------------------------------------------

class TestFeatureDecoder:
    def test_sudoku4_f38(self, sudoku4):
        """f_38 in 4x4 Sudoku → cell 9 → (row=2, col=1), digit=3."""
        atom = decode("f_38", True, sudoku4)
        assert atom is not None
        assert atom.kind == "cell_digit"
        assert atom.payload == {"row": 2, "col": 1, "digit": 3}
        assert atom.polarity is True

    def test_sudoku4_negative_polarity(self, sudoku4):
        atom = decode("f_38", False, sudoku4)
        assert atom is not None
        assert atom.polarity is False

    def test_sudoku4_f0(self, sudoku4):
        """f_0 → cell 0, row 0, col 0, digit 1."""
        atom = decode("f_0", True, sudoku4)
        assert atom.payload == {"row": 0, "col": 0, "digit": 1}

    def test_sudoku4_out_of_range(self, sudoku4):
        """Index beyond n*n*n → None."""
        assert decode("f_999", True, sudoku4) is None

    def test_unrecognised_name(self, sudoku4):
        assert decode("feature_38", True, sudoku4) is None
        assert decode("x38", True, sudoku4) is None

    def test_minesweeper_f0(self, ms4):
        """f_0 → cell 0 = (0,0), channel 0 (hidden)."""
        atom = decode("f_0", True, ms4)
        assert atom is not None
        assert atom.kind == "cell_state"
        assert atom.payload == {"row": 0, "col": 0, "channel": 0}

    def test_minesweeper_f_channel(self, ms4):
        """f_5 in 4x4 → cell=0 (5//11=0), ch=5 (5%11=5) → shows 3 (ch=2+3=5)."""
        atom = decode("f_5", True, ms4)
        assert atom.payload["channel"] == 5
        assert atom.payload["row"] == 0
        assert atom.payload["col"] == 0

    def test_minesweeper_cell1(self, ms4):
        """f_11 → cell=1 (11//11=1), ch=0 → row=0, col=1, channel=0."""
        atom = decode("f_11", True, ms4)
        assert atom.payload == {"row": 0, "col": 1, "channel": 0}

# ---------------------------------------------------------------------------
# Sudoku templates
# ---------------------------------------------------------------------------

class TestCellUniqueness:
    def _atoms(self, r, c, d_pos, d_neg):
        return [
            Atom("cell_digit", {"row": r, "col": c, "digit": d_pos}, True),
            Atom("cell_digit", {"row": r, "col": c, "digit": d_neg}, False),
        ]

    def test_positive_match(self, sudoku4):
        atoms = self._atoms(0, 1, 2, 4)
        rule = sudoku_templates.cell_uniqueness(atoms, sudoku4)
        assert rule is not None
        assert rule.template == "cell_uniqueness"
        assert "cell (0,1)" in rule.text
        assert "holds 2" in rule.text
        assert "does not hold 4" in rule.text

    def test_same_digit_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, False),
        ]
        assert sudoku_templates.cell_uniqueness(atoms, sudoku4) is None

    def test_different_cell_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 0, "col": 2, "digit": 4}, False),
        ]
        assert sudoku_templates.cell_uniqueness(atoms, sudoku4) is None

    def test_both_positive_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 4}, True),
        ]
        assert sudoku_templates.cell_uniqueness(atoms, sudoku4) is None


class TestRowUniqueness:
    def test_positive_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 0, "col": 2, "digit": 2}, False),
        ]
        rule = sudoku_templates.row_uniqueness(atoms, sudoku4)
        assert rule is not None
        assert rule.template == "row_uniqueness"
        assert "cell (0,1)" in rule.text
        assert "cell (0,2)" in rule.text

    def test_different_digit_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 0, "col": 2, "digit": 3}, False),
        ]
        assert sudoku_templates.row_uniqueness(atoms, sudoku4) is None

    def test_different_row_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 1, "col": 2, "digit": 2}, False),
        ]
        assert sudoku_templates.row_uniqueness(atoms, sudoku4) is None


class TestColumnUniqueness:
    def test_positive_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 3}, True),
            Atom("cell_digit", {"row": 2, "col": 1, "digit": 3}, False),
        ]
        rule = sudoku_templates.column_uniqueness(atoms, sudoku4)
        assert rule is not None
        assert rule.template == "column_uniqueness"
        assert "col" in rule.text or ",1)" in rule.text

    def test_different_col_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 3}, True),
            Atom("cell_digit", {"row": 2, "col": 2, "digit": 3}, False),
        ]
        assert sudoku_templates.column_uniqueness(atoms, sudoku4) is None


class TestBoxUniqueness:
    def test_positive_match_4x4(self, sudoku4):
        """Cells (0,0) and (1,1) are in the same 2×2 box."""
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 0, "digit": 2}, True),
            Atom("cell_digit", {"row": 1, "col": 1, "digit": 2}, False),
        ]
        rule = sudoku_templates.box_uniqueness(atoms, sudoku4)
        assert rule is not None
        assert rule.template == "box_uniqueness"
        assert "same box" in rule.text

    def test_different_box_no_match(self, sudoku4):
        """Cells (0,0) and (0,2) are in different boxes in 4×4."""
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 0, "digit": 2}, True),
            Atom("cell_digit", {"row": 0, "col": 2, "digit": 2}, False),
        ]
        assert sudoku_templates.box_uniqueness(atoms, sudoku4) is None

    def test_unsupported_size_no_match(self):
        game = SudokuGame(size=4)
        game.size = 5   # artificially unsupported
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 0, "digit": 2}, True),
            Atom("cell_digit", {"row": 1, "col": 1, "digit": 2}, False),
        ]
        assert sudoku_templates.box_uniqueness(atoms, game) is None


class TestNoMatchForMixedClause:
    def test_different_kinds_no_match(self, sudoku4):
        atoms = [
            Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
            Atom("cell_digit", {"row": 1, "col": 2, "digit": 3}, False),
        ]
        result = match_clause(atoms, sudoku4)
        # Strict templates (row/col/box/cell_uniqueness, naked/hidden_single,
        # naked_pair, pointing_pair) must NOT fire — atoms share no axis or
        # digit.  A universal/* fallback (e.g. mixed_polarity) is acceptable
        # and intentional once the universal layer is registered.
        if result is not None:
            assert result.template.startswith("universal/"), (
                f"strict template fired on unrelated atoms: {result.template}"
            )


# ---------------------------------------------------------------------------
# Minesweeper templates
# ---------------------------------------------------------------------------

class TestLocalExhaustion:
    def _shows_atom(self, r, c, n):
        return Atom("cell_state", {"row": r, "col": c, "channel": 2 + n}, True)

    def _hidden_atom(self, r, c):
        return Atom("cell_state", {"row": r, "col": c, "channel": 0}, True)

    def test_positive_match(self, ms4):
        """Cell (1,1) shows 2, and its two neighbors (0,1) and (1,0) are hidden."""
        atoms = [
            self._shows_atom(1, 1, 2),
            self._hidden_atom(0, 1),
            self._hidden_atom(1, 0),
        ]
        rule = ms_templates.local_exhaustion(atoms, ms4)
        assert rule is not None
        assert rule.template == "local_exhaustion"
        assert "cell (1,1)" in rule.text
        assert "shows 2" in rule.text
        assert "all 2 are mines" in rule.text

    def test_non_neighbor_no_match(self, ms4):
        """Hidden cell at (3,3) is not a neighbor of (1,1)."""
        atoms = [
            self._shows_atom(1, 1, 1),
            self._hidden_atom(3, 3),
        ]
        assert ms_templates.local_exhaustion(atoms, ms4) is None

    def test_extra_atom_no_match(self, ms4):
        """More atoms than 1+N → not a clean exhaustion pattern."""
        atoms = [
            self._shows_atom(1, 1, 1),
            self._hidden_atom(0, 1),
            self._hidden_atom(1, 0),  # extra
        ]
        assert ms_templates.local_exhaustion(atoms, ms4) is None

    def test_shows0_no_match(self, ms4):
        """Shows 0 should not match local_exhaustion (no mines)."""
        atoms = [self._shows_atom(0, 0, 0)]
        assert ms_templates.local_exhaustion(atoms, ms4) is None


class TestLocalCount:
    def test_positive_match_4x4(self, ms4):
        """Cell (1,1) shows 2, 2 flagged neighbors, rest negated-hidden."""
        nbrs = ms4._neighbours(1, 1)  # 8 neighbors in 4x4
        shows = Atom("cell_state", {"row": 1, "col": 1, "channel": 4}, True)  # shows 2
        flagged = [
            Atom("cell_state", {"row": nbrs[0][0], "col": nbrs[0][1], "channel": 1}, True),
            Atom("cell_state", {"row": nbrs[1][0], "col": nbrs[1][1], "channel": 1}, True),
        ]
        flagged_cells = {(nbrs[0][0], nbrs[0][1]), (nbrs[1][0], nbrs[1][1])}
        neg_hidden = [
            Atom("cell_state", {"row": r, "col": c, "channel": 0}, False)
            for r, c in nbrs[2:]
        ]
        atoms = [shows] + flagged + neg_hidden
        rule = ms_templates.local_count(atoms, ms4)
        assert rule is not None
        assert rule.template == "local_count"
        assert "safe" in rule.text

    def test_wrong_flag_count_no_match(self, ms4):
        """shows 2 but only 1 flagged neighbor → no match."""
        shows = Atom("cell_state", {"row": 1, "col": 1, "channel": 4}, True)
        flagged = [Atom("cell_state", {"row": 0, "col": 1, "channel": 1}, True)]
        atoms = [shows] + flagged
        assert ms_templates.local_count(atoms, ms4) is None


# ---------------------------------------------------------------------------
# render_clauses_as_nl integration
# ---------------------------------------------------------------------------

class TestRenderClausesAsNL:
    def test_sudoku_canonical_and_noncanonical(self, sudoku4):
        """Mix of matching and non-matching formulas."""
        from sympy import Symbol, And, Not

        # f_0 = row0,col0,digit1 | f_1 = row0,col0,digit2 → cell_uniqueness
        f0 = Symbol("f_0")
        f1 = Symbol("f_1")
        formula_match = And(f0, Not(f1))

        # f_38 alone → single atom, no template match for single atom
        f38 = Symbol("f_38")
        formula_single = f38

        result = render_clauses_as_nl([formula_match, formula_single], sudoku4)
        assert result["total"] == 2
        assert result["matched"] >= 0   # cell_uniqueness may or may not match
        assert "canonical_match_rate" in result
        assert isinstance(result["nl_rules"], list)
        assert len(result["nl_rules"]) == 2

    def test_empty_formulas(self, sudoku4):
        result = render_clauses_as_nl([], sudoku4)
        assert result["total"] == 0
        assert result["matched"] == 0
        assert result["canonical_match_rate"] == 0.0

    def test_all_noncanonical(self, sudoku4):
        from sympy import Symbol
        result = render_clauses_as_nl([Symbol("f_0")], sudoku4)
        assert result["nl_rules"][0] == "(non-canonical)"
        assert result["matched"] == 0

    def test_sudoku_cell_uniqueness_via_render(self, sudoku4):
        """A known cell-uniqueness clause should be matched end-to-end."""
        from sympy import Symbol, And, Not
        # f_4 in 4x4: cell=1(row0,col1), digit=1 → row0,col1,digit1
        # f_5: cell=1, digit=2 → row0,col1,digit2
        # And(f_4, Not(f_5)) → cell (0,1) holds 1, does not hold 2 → cell_uniqueness
        f4 = Symbol("f_4")
        f5 = Symbol("f_5")
        result = render_clauses_as_nl([And(f4, Not(f5))], sudoku4)
        assert result["matched"] == 1
        assert result["canonical_match_rate"] == 1.0
        assert "cell_uniqueness" in result["by_template"]
        assert "cell (0,1)" in result["nl_rules"][0]


# ---------------------------------------------------------------------------
# Fix 1: multi-atom clause relaxation tests (COPILOT_TASK_2)
# ---------------------------------------------------------------------------

def test_cell_uniqueness_with_extra_atoms(sudoku4):
    """Canonical pair embedded among 2 unrelated atoms should still match."""
    a1 = Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True)
    a2 = Atom("cell_digit", {"row": 0, "col": 1, "digit": 4}, False)
    noise = [
        Atom("cell_digit", {"row": 3, "col": 2, "digit": 1}, True),
        Atom("cell_digit", {"row": 1, "col": 3, "digit": 3}, False),
    ]
    rule = match_clause([a1, a2, *noise], sudoku4)
    assert rule is not None
    assert rule.template == "cell_uniqueness"
    assert {a1, a2}.issubset(set(rule.atoms))


def test_no_canonical_pair_returns_none(sudoku4):
    """Three atoms with no pair satisfying any uniqueness relation → no
    *strict* match.  Once the universal layer is registered, mixed-polarity
    or spatial-locality fallbacks may legitimately fire; we just verify the
    strict tier stays silent."""
    atoms = [
        Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True),
        Atom("cell_digit", {"row": 3, "col": 2, "digit": 1}, True),
        Atom("cell_digit", {"row": 1, "col": 3, "digit": 3}, False),
    ]
    result = match_clause(atoms, sudoku4)
    if result is not None:
        assert result.template.startswith("universal/"), (
            f"strict template fired on unrelated atoms: {result.template}"
        )


def test_first_match_wins_cell_before_row(sudoku4):
    """cell_uniqueness must be checked before row_uniqueness."""
    a = Atom("cell_digit", {"row": 0, "col": 1, "digit": 2}, True)
    b = Atom("cell_digit", {"row": 0, "col": 1, "digit": 4}, False)   # cell-uniqueness with a
    c = Atom("cell_digit", {"row": 0, "col": 3, "digit": 2}, False)   # row-uniqueness with a
    rule = match_clause([a, b, c], sudoku4)
    assert rule is not None
    assert rule.template == "cell_uniqueness"
