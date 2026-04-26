"""Run the full pipeline: data → train → explain → report.

Usage:
    python scripts/run_experiment.py --game sudoku --model mlp --xai rule_extraction
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symbolic_xai_logic.experiments.runner import ExperimentRunner
from symbolic_xai_logic.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full experiment pipeline")
    p.add_argument("--game", default="sudoku")
    p.add_argument("--model", default="mlp")
    p.add_argument("--xai", default="rule_extraction")
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--difficulty", default="medium",
                   choices=["easy", "medium", "hard"],
                   help="Puzzle difficulty: medium = 8/16 given cells, hard = 6/16")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="Stop after N eval rounds with no val_loss improvement (0 = disabled)")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    n_val = max(200, args.n_train // 10)
    n_test = max(200, args.n_train // 10)

    config = {
        "seed": args.seed,
        "device": args.device,
        "game": {"name": args.game, "size": args.size, "difficulty": args.difficulty},
        "model": {"name": args.model},
        "xai": {"name": args.xai},
        "data": {"n_train": args.n_train, "n_val": n_val, "n_test": n_test, "encoding": "one_hot"},
        "training": {
            "epochs": args.epochs,
            "lr": 1e-3,
            "batch_size": 128,
            "weight_decay": 1e-4,
            "eval_interval": 2,
            "checkpoint_dir": f"{args.results_dir}/checkpoints",
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
    }

    runner = ExperimentRunner(config, results_dir=args.results_dir)
    report = runner.run()

    print("\n=== Experiment Results ===")
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
