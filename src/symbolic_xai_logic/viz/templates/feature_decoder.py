"""Game-aware feature-index → Atom decoder.

Feature names are the default names produced by RuleExtractor: "f_{i}" where
i is the flat feature index into the encoded input.

Encoding layouts assumed:
  Sudoku one_hot : index i → cell = i // n, digit = (i % n) + 1
  Minesweeper one_hot : index i → cell = i // 11, ch = i % 11  (N_SPATIAL_CHANNELS=11)
  Minesweeper spatial  : index i → ch = i // (n*n), cell = i % (n*n)
"""
from __future__ import annotations
from typing import Any
from .atom import Atom


def _parse_index(feature_name: str) -> int | None:
    """Extract integer index from "f_{i}" style names. Returns None if unrecognised."""
    if not feature_name.startswith("f_"):
        return None
    try:
        return int(feature_name[2:])
    except ValueError:
        return None


def decode(feature_name: str, polarity: bool, game: Any) -> Atom | None:
    """
    Map a sympy Symbol name (e.g. "f_38") plus polarity to an Atom.
    Returns None when the game type is unsupported or name is unrecognised.
    """
    from ...games.sudoku import SudokuGame
    from ...games.minesweeper import MinesweeperGame, N_SPATIAL_CHANNELS

    i = _parse_index(feature_name)
    if i is None or i < 0:
        return None

    if isinstance(game, SudokuGame):
        n = game.size
        total = n * n * n
        if i >= total:
            return None
        cell = i // n
        row, col = divmod(cell, n)
        digit = (i % n) + 1
        return Atom("cell_digit", {"row": row, "col": col, "digit": digit}, polarity)

    if isinstance(game, MinesweeperGame):
        n = game.size
        k = N_SPATIAL_CHANNELS   # 11
        total = n * n * k
        if i >= total:
            return None
        # Determine layout: one_hot (cell-major) vs spatial (channel-major)
        # Default to one_hot (used by MLP rule extraction).
        # Spatial layout is used when CNN features have been extracted.
        cell = i // k
        ch = i % k
        row, col = divmod(cell, n)
        return Atom("cell_state", {"row": row, "col": col, "channel": ch}, polarity)

    return None


def decode_spatial(feature_name: str, polarity: bool, game: Any) -> Atom | None:
    """Variant of decode() for spatially-encoded Minesweeper (channel-first layout)."""
    from ...games.minesweeper import MinesweeperGame, N_SPATIAL_CHANNELS

    i = _parse_index(feature_name)
    if i is None or i < 0:
        return None

    if isinstance(game, MinesweeperGame):
        n = game.size
        k = N_SPATIAL_CHANNELS
        if i >= n * n * k:
            return None
        ch = i // (n * n)
        cell = i % (n * n)
        row, col = divmod(cell, n)
        return Atom("cell_state", {"row": row, "col": col, "channel": ch}, polarity)

    return decode(feature_name, polarity, game)
