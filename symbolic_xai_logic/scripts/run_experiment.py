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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    config = {
        "seed": args.seed,
        "device": args.device,
        "game": {"name": args.game, "size": args.size, "difficulty": "easy"},
        "model": {"name": args.model},
        "xai": {"name": args.xai},
        "data": {"n_train": args.n_train, "n_val": 400, "n_test": 400, "encoding": "one_hot"},
        "training": {
            "epochs": args.epochs,
            "lr": 1e-3,
            "batch_size": 64,
            "weight_decay": 1e-4,
            "eval_interval": max(1, args.epochs // 5),
            "checkpoint_dir": f"{args.results_dir}/checkpoints",
        },
    }

    runner = ExperimentRunner(config, results_dir=args.results_dir)
    report = runner.run()

    print("\n=== Experiment Results ===")
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
