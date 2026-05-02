"""Validate that every model config in scripts/run_extended.py matches its model's __init__ signature.

Each test instantiates the model with the exact kwargs from the config dict (the same
way ExperimentRunner does in runner.py) and runs a forward pass. Any param-name mismatch
raises TypeError immediately.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
# Make the package importable (mirrors run_extended.py's own sys.path setup)
sys.path.insert(0, str(REPO / "src"))
# Make scripts/ importable so we can import run_extended
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "run_extended", REPO / "scripts" / "run_extended.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
CONFIGS: list[dict] = _mod.CONFIGS

from symbolic_xai_logic.models.registry import get_model  # noqa: E402


def _build_instantiation_args(cfg: dict) -> tuple[str, int, int, dict]:
    """Return (model_name, input_dim, output_dim, kwargs) mirroring runner.py logic."""
    model_cfg = cfg["model"]
    model_name: str = model_cfg["name"]
    kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "size")}

    if model_name == "cnn":
        game = cfg["game"]
        grid_size: int = game.get("size", 8)
        encoding: str = cfg.get("data", {}).get("encoding", "one_hot")
        # Minesweeper spatial encoding uses 4 channels; everything else is 1-channel one-hot
        n_channels = 4 if (game["name"] == "minesweeper" and encoding == "spatial") else 1
        kwargs.setdefault("grid_size", grid_size)
        kwargs.setdefault("n_channels", n_channels)
        input_dim = n_channels * grid_size * grid_size
        output_dim = grid_size * grid_size
    else:
        # 128 is divisible by typical token / node-feature sizes used in the configs
        input_dim = 128
        output_dim = 16

    return model_name, input_dim, output_dim, kwargs


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c["label"] for c in CONFIGS])
def test_model_instantiation(cfg: dict) -> None:
    """Model instantiation must not raise TypeError (catches param-name mismatches)."""
    model_name, input_dim, output_dim, kwargs = _build_instantiation_args(cfg)
    model = get_model(model_name, input_dim=input_dim, output_dim=output_dim, **kwargs)
    assert model is not None


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c["label"] for c in CONFIGS])
def test_model_forward(cfg: dict) -> None:
    """Model forward pass must produce output of shape (batch, output_dim)."""
    model_name, input_dim, output_dim, kwargs = _build_instantiation_args(cfg)
    model = get_model(model_name, input_dim=input_dim, output_dim=output_dim, **kwargs)
    model.eval()
    x = torch.randn(2, input_dim)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, output_dim), (
        f"{cfg['label']}: expected output shape (2, {output_dim}), got {tuple(out.shape)}"
    )
