"""Run XAI on a trained checkpoint.

Usage:
    python scripts/explain.py --checkpoint results/checkpoints/sudoku4_best.pt --xai rule_extraction
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from symbolic_xai_logic.utils.io import load_checkpoint
from symbolic_xai_logic.utils.logging import get_logger
from symbolic_xai_logic.games import get_game
from symbolic_xai_logic.models import get_model
from symbolic_xai_logic.xai import get_explainer
from symbolic_xai_logic.data import generate_dataset
from symbolic_xai_logic.viz.rule_render import render_explanation

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XAI on a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--xai", default="rule_extraction",
                   choices=["lime", "shap", "lrp", "rule_extraction",
                            "concept_probe", "symbolic_regression"])
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt.get("config", {})

    game_cfg = config.get("game", {"name": "sudoku", "size": 4})
    game_name = game_cfg.get("name", "sudoku")
    game_kwargs = {k: v for k, v in game_cfg.items() if k != "name"}
    game = get_game(game_name, **game_kwargs)

    model_cfg = config.get("model", {"name": "mlp"})
    model_name = model_cfg.get("name", "mlp")
    model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "size")}

    # CNN needs grid_size and n_channels
    if model_name == "cnn":
        from symbolic_xai_logic.games.minesweeper import MinesweeperGame, N_SPATIAL_CHANNELS
        if isinstance(game, MinesweeperGame):
            input_dim = game.spatial_input_dim
            model_kwargs.setdefault("grid_size", game.size)
            model_kwargs.setdefault("n_channels", N_SPATIAL_CHANNELS)
        else:
            input_dim = game.input_dim
    else:
        input_dim = game.input_dim

    model = get_model(model_name, input_dim=input_dim, output_dim=game.output_dim, **model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Generate test data with the encoding stored in the checkpoint config
    data_cfg = config.get("data", {})
    encoding = data_cfg.get("encoding", "one_hot")
    splits = generate_dataset(game, n_train=10, n_val=10, n_test=args.n_samples,
                               encoding=encoding, seed=42)
    X_test = splits["test"]["X"]
    solutions_test = splits["test"].get("solutions")

    explainer = get_explainer(args.xai, model=model, game=game)
    explain_kwargs = {}
    if solutions_test:
        explain_kwargs["solutions"] = solutions_test

    explanation = explainer.explain(X_test, **explain_kwargs)
    # Pass game= so the NL rules section is rendered
    rendered = render_explanation(explanation, game_name=game.name, xai_name=args.xai, game=game)
    print(rendered)

    # Save
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"explanation_{game.name}_{args.xai}.txt"
    out_path.write_text(rendered)
    logger.info(f"Explanation saved to {out_path}")


if __name__ == "__main__":
    main()
