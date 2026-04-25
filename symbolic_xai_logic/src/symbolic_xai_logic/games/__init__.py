from .base import Game
from .sudoku import SudokuGame
from .nqueens import NQueensGame
from .knights_knaves import KnightsKnavesGame
from .sat3 import SAT3Game
from .minesweeper import MinesweeperGame

GAME_REGISTRY: dict[str, type] = {
    "sudoku": SudokuGame,
    "nqueens": NQueensGame,
    "knights_knaves": KnightsKnavesGame,
    "sat3": SAT3Game,
    "minesweeper": MinesweeperGame,
}


def get_game(name: str, **kwargs) -> "Game":
    if name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {name}. Available: {list(GAME_REGISTRY)}")
    return GAME_REGISTRY[name](**kwargs)
