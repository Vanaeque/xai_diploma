from .base import Game
from .sudoku import SudokuGame
from .minesweeper import MinesweeperGame

GAME_REGISTRY: dict[str, type] = {
    "sudoku": SudokuGame,
    "minesweeper": MinesweeperGame,
}


def get_game(name: str, **kwargs) -> "Game":
    if name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {name}. Available: {list(GAME_REGISTRY)}")
    return GAME_REGISTRY[name](**kwargs)
