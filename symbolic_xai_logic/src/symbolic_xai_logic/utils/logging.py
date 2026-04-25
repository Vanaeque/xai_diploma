from __future__ import annotations
"""Rich-based logging helpers."""
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)
