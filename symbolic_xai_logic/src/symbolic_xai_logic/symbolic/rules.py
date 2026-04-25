"""Canonical rule sets for each game, expressed in human-readable and sympy forms."""
from __future__ import annotations
from typing import Any


CANONICAL_RULES: dict[str, list[str]] = {
    "sudoku4": [
        "Each row contains each digit 1-4 exactly once",
        "Each column contains each digit 1-4 exactly once",
        "Each 2x2 box contains each digit 1-4 exactly once",
        "Each cell contains exactly one digit",
    ],
    "sudoku9": [
        "Each row contains each digit 1-9 exactly once",
        "Each column contains each digit 1-9 exactly once",
        "Each 3x3 box contains each digit 1-9 exactly once",
        "Each cell contains exactly one digit",
    ],
    "nqueens4": [
        "Each row contains exactly one queen",
        "Each column contains exactly one queen",
        "No two queens share a diagonal",
        "Total number of queens equals board size",
    ],
    "nqueens6": [
        "Each row contains exactly one queen",
        "Each column contains exactly one queen",
        "No two queens share a diagonal",
        "Total number of queens equals board size",
    ],
    "nqueens8": [
        "Each row contains exactly one queen",
        "Each column contains exactly one queen",
        "No two queens share a diagonal",
    ],
    "knights_knaves": [
        "Knights always tell the truth",
        "Knaves always lie",
        "If agent i is a knight, all statements by i are true",
        "If agent i is a knave, all statements by i are false",
    ],
    "sat3": [
        "Every clause must have at least one satisfied literal",
        "Each variable is either true or false",
        "Each clause has exactly 3 literals",
    ],
    "minesweeper8": [
        "Total number of mines equals n_mines",
        "Each revealed number equals the count of adjacent mines",
        "Each cell is either a mine or safe (binary)",
        "Revealed safe cells are not mines",
    ],
}

# Feature-level rules for concept probing (maps concept name -> boolean formula description)
CONCEPT_RULES: dict[str, dict[str, str]] = {
    "sudoku4": {
        "row_uniqueness": "∀r,d: exactly one cell in row r has value d",
        "col_uniqueness": "∀c,d: exactly one cell in col c has value d",
        "box_uniqueness": "∀b,d: exactly one cell in box b has value d",
    },
    "sudoku9": {
        "row_uniqueness": "∀r,d: exactly one cell in row r has value d",
        "col_uniqueness": "∀c,d: exactly one cell in col c has value d",
        "box_uniqueness": "∀b,d: exactly one cell in box b has value d",
    },
    "nqueens6": {
        "one_per_row": "∀r: exactly one queen in row r",
        "one_per_col": "∀c: exactly one queen in col c",
        "no_diagonal": "∀i≠j: |row_i - row_j| ≠ |col_i - col_j|",
    },
    "knights_knaves": {
        "truth_teller": "knight(i) ↔ all statements by i are true",
        "liar": "knave(i) ↔ all statements by i are false",
    },
    "sat3": {
        "clause_satisfaction": "∀clause: at least one literal is satisfied",
    },
    "minesweeper8": {
        "mine_count": "total mines == n_mines",
        "clue_consistency": "∀revealed cell (r,c): count(adjacent mines) == clue(r,c)",
        "row_mine_presence": "∀r: row r has at least one mine (for most rows)",
    },
}


def get_canonical_rules(game_name: str) -> list[str]:
    for key in CANONICAL_RULES:
        if key in game_name or game_name in key:
            return CANONICAL_RULES[key]
    return CANONICAL_RULES.get(game_name, ["No canonical rules defined"])


def get_concept_rules(game_name: str) -> dict[str, str]:
    for key in CONCEPT_RULES:
        if key in game_name or game_name in key:
            return CONCEPT_RULES[key]
    return CONCEPT_RULES.get(game_name, {})
