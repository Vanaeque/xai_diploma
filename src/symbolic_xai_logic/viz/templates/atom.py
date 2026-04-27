"""Atom and NaturalLanguageRule dataclasses."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(eq=True, frozen=True)
class Atom:
    """One literal in a DNF clause, decoded back to a game-level fact."""
    kind: str           # "cell_digit" | "mine" | "cell_state" | "queen" | ...
    payload: dict       # game-specific, e.g. {"row": 0, "col": 1, "digit": 2}
    polarity: bool      # True = positive literal, False = negated

    def __hash__(self) -> int:
        return hash((self.kind, tuple(sorted(self.payload.items())), self.polarity))


@dataclass
class NaturalLanguageRule:
    """A DNF clause that matched a canonical game template."""
    template: str           # e.g. "row_uniqueness"
    text: str               # human-readable rendering
    atoms: tuple            # tuple[Atom, ...] — matched atoms for traceability
