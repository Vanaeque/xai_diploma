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
    p.add_argument(
        "--all-targets", action="store_true",
        help="(rule_extraction only) Fit a tree per blank cell and save all canonical rules",
    )
    return p.parse_args()


def _save_canonical_rules(rules: list[dict], path: Path, game_name: str, xai_name: str) -> None:
    from collections import defaultdict
    by_template: dict[str, list[str]] = defaultdict(list)
    for r in rules:
        by_template[r["template"]].append(r["text"])

    lines = [
        f"=== All Canonical Rules: {game_name} / {xai_name} ===",
        f"Total unique canonical rules: {len(rules)}",
        "",
    ]
    for template in sorted(by_template):
        entries = by_template[template]
        lines.append(f"[{template}]  ({len(entries)} rules)")
        for i, text in enumerate(entries, 1):
            lines.append(f"  {i:3}. {text}")
        lines.append("")

    path.write_text("\n".join(lines))


def _print_canonical_summary(rules: list[dict]) -> None:
    from collections import Counter
    counts = Counter(r["template"] for r in rules)
    print(f"\n=== Canonical Rules Summary ({len(rules)} unique) ===")
    for template, n in sorted(counts.items()):
        print(f"  {template:<24} {n}")


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
            # For other games (like Sudoku), use grid_size = size and n_channels = size
            model_kwargs.setdefault("grid_size", game.size)
            model_kwargs.setdefault("n_channels", game.size)
            # CNN input for spatial encoding: n_channels * size * size
            input_dim = game.size * game.size * game.size
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

    # Save primary explanation
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"explanation_{game.name}_{args.xai}.txt"
    out_path.write_text(rendered)
    logger.info(f"Explanation saved to {out_path}")

    # All-targets canonical rules
    if args.all_targets and hasattr(explainer, "extract_all_canonical_rules"):
        logger.info("Extracting canonical rules for all blank cells …")
        all_rules = explainer.extract_all_canonical_rules(X_test, game)
        canonical_path = results_dir / f"canonical_rules_{game.name}_{args.xai}.txt"
        _save_canonical_rules(all_rules, canonical_path, game.name, args.xai)
        print(f"\nCanonical rules saved to {canonical_path}")
        _print_canonical_summary(all_rules)


if __name__ == "__main__":
    main()
