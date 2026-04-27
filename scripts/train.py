"""Train a model on a logic game.

Usage:
    python scripts/train.py --game sudoku --model mlp
    python scripts/train.py --game minesweeper --model cnn --no-xai  # checkpoint only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symbolic_xai_logic.experiments.runner import ExperimentRunner
from symbolic_xai_logic.utils.seeding import set_global_seed

GAME_CHOICES = ["sudoku", "nqueens", "knights_knaves", "sat3", "minesweeper"]
MODEL_CHOICES = ["mlp", "gnn", "transformer", "cnn"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model on a logic game")
    p.add_argument("--game", default="sudoku", choices=GAME_CHOICES)
    p.add_argument("--model", default="mlp", choices=MODEL_CHOICES)
    p.add_argument("--size", type=int, default=4,
                   help="Game size (e.g. 4 for 4×4 Sudoku, 8 for Minesweeper)")
    p.add_argument("--difficulty", default="medium",
                   choices=["easy", "medium", "hard"],
                   help="Puzzle difficulty: medium = 8/16 given cells, hard = 6/16")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--xai", default="rule_extraction")
    p.add_argument("--no-xai", action="store_true",
                   help="Train only — save checkpoint without running any explainer")
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="Stop after N eval rounds with no val_loss improvement (0 = disabled)")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                   help="Minimum val_loss decrease counted as improvement")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    encoding = "spatial" if (args.model == "cnn" and args.game == "minesweeper") else "one_hot"
    n_test = max(200, args.n_val)

    config = {
        "seed": args.seed,
        "device": args.device,
        "game": {"name": args.game, "size": args.size, "difficulty": args.difficulty},
        "model": {"name": args.model},
        "xai": {"name": args.xai},
        "data": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": n_test,
            "encoding": encoding,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": 1e-4,
            "eval_interval": 2,
            "checkpoint_dir": f"{args.results_dir}/checkpoints",
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
    }

    runner = ExperimentRunner(config, results_dir=args.results_dir)

    if args.no_xai:
        ckpt_path = runner.train_only()
        print(f"\n=== Training Complete (no XAI) ===")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Run explain.py --checkpoint {ckpt_path} to explain later.")
    else:
        report = runner.run()
        print("\n=== Training Complete ===")
        for k, v in report.to_dict().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
