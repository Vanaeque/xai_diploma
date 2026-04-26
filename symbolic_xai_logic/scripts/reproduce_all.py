"""Reproduce all diploma results.

Two modes:
    python scripts/reproduce_all.py           # full  — 3 configs × 5 seeds × 4 XAI = 60 runs
    python scripts/reproduce_all.py --quick   # smoke — 1 config, 1 seed, rule_extraction only

Pipeline
--------
  Phase 1  Train one checkpoint per (config, seed)      — skips if checkpoint already exists
  Phase 2  Run all four XAI methods on each checkpoint  — writes explanation_*.txt,
                                                           canonical_rules_*.txt, report_*.json
  Phase 3  Aggregate canonical rules across seeds       — canonical_summary.csv / .md
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from symbolic_xai_logic.experiments.runner import ExperimentRunner
from symbolic_xai_logic.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
FULL_XAI = ["rule_extraction", "symbolic_regression", "concept_probe", "lrp"]

CONFIGS: list[dict] = [
    {
        "label": "sudoku4_medium_mlp_large",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
        "model": {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 80, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
    },
    {
        "label": "sudoku4_hard_mlp_large",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "hard"},
        "model": {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 100, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 8, "early_stop_min_delta": 1e-4,
        },
    },
    {
        "label": "minesweeper8_medium_mlp",
        "game":  {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
        "model": {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1},
        "data":  {"n_train": 30000, "n_val": 3000, "n_test": 3000, "encoding": "one_hot"},
        "training": {
            "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
    },
]

# Quick-mode overrides on the first config: tiny data, minimal epochs
_QUICK_DATA = {"n_train": 1000, "n_val": 200, "n_test": 200, "encoding": "one_hot"}
_QUICK_TRAINING = {
    "epochs": 3, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-4,
    "eval_interval": 1, "early_stop_patience": 0, "early_stop_min_delta": 0.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runtime_cfg(cfg: dict, seed: int, device: str, run_dir: Path) -> dict:
    return {
        "seed": seed,
        "device": device,
        "game": dict(cfg["game"]),
        "model": dict(cfg["model"]),
        "data": dict(cfg["data"]),
        "training": {**cfg["training"], "checkpoint_dir": str(run_dir / "checkpoints")},
        "xai": {"name": "rule_extraction"},  # train_only ignores this
    }


def _ckpt_path(cfg: dict, run_dir: Path) -> Path:
    game = cfg["game"]
    return run_dir / "checkpoints" / f"{game['name']}{game['size']}_best.pt"


def train_one(
    cfg: dict, seed: int, device: str, results_dir: Path
) -> tuple[str, int, Path, Path] | None:
    """Train one (config, seed). Returns (label, seed, ckpt, run_dir) or None on failure."""
    label = cfg["label"]
    run_dir = results_dir / f"{label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _ckpt_path(cfg, run_dir)

    if ckpt.exists():
        logger.info(f"[skip train] {label} seed{seed} — checkpoint exists")
        return (label, seed, ckpt, run_dir)

    logger.info(f"[train] {label} seed{seed}")
    runtime = _runtime_cfg(cfg, seed, device, run_dir)
    try:
        runner = ExperimentRunner(runtime, results_dir=str(run_dir))
        runner.train_only()
        return (label, seed, ckpt, run_dir)
    except Exception as exc:
        logger.error(f"[train fail] {label} seed{seed}: {exc}")
        import traceback; traceback.print_exc()
        return None


def explain_one(label: str, seed: int, ckpt: Path, run_dir: Path, xai: str) -> dict:
    """Run explain.py for one (checkpoint, xai). Returns a status dict."""
    extra = ["--all-targets"] if xai == "rule_extraction" else []
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "explain.py"),
        "--checkpoint", str(ckpt),
        "--xai", xai,
        "--results-dir", str(run_dir),
        *extra,
    ]
    logger.info(f"[explain] {label} seed{seed} {xai}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.error(f"[explain fail] {label} seed{seed} {xai}\n{proc.stderr[-2000:]}")
    return {
        "label": label, "seed": seed, "xai": xai,
        "ok": proc.returncode == 0,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Reproduce all diploma results (full or quick smoke test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/reproduce_all.py                         # full run\n"
            "  python scripts/reproduce_all.py --quick                 # smoke test\n"
            "  python scripts/reproduce_all.py --device cuda           # GPU\n"
            "  python scripts/reproduce_all.py --results-dir my_run    # custom output dir\n"
        ),
    )
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 1 config, 1 seed, rule_extraction only, tiny data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-dir", default="results/full")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seeds to use (full mode only; quick always uses seed 0)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        quick_cfg = {
            **CONFIGS[0],
            "label": CONFIGS[0]["label"] + "_quick",
            "data": _QUICK_DATA,
            "training": _QUICK_TRAINING,
        }
        configs = [quick_cfg]
        seeds = [0]
        xai_methods = ["rule_extraction"]
    else:
        configs = CONFIGS
        seeds = args.seeds
        xai_methods = FULL_XAI

    n_train_jobs = len(configs) * len(seeds)
    n_explain_jobs = n_train_jobs * len(xai_methods)
    mode = "QUICK" if args.quick else "FULL"
    logger.info(
        f"[{mode}] {len(configs)} config(s) × {len(seeds)} seed(s) × {len(xai_methods)} XAI "
        f"= {n_explain_jobs} explanation runs on {n_train_jobs} checkpoints"
    )
    logger.info(f"Results dir: {results_dir}")

    t0 = time.time()

    # --- Phase 1: Training ------------------------------------------------
    logger.info("=== Phase 1: Training ===")
    trained: list[tuple[str, int, Path, Path]] = []
    for cfg in configs:
        for seed in seeds:
            result = train_one(cfg, seed, args.device, results_dir)
            if result is not None:
                trained.append(result)

    logger.info(f"Phase 1 complete: {len(trained)}/{n_train_jobs} checkpoints ready")

    # --- Phase 2: Explanation ---------------------------------------------
    logger.info("=== Phase 2: Explanation ===")
    statuses: list[dict] = []
    for label, seed, ckpt, run_dir in trained:
        for xai in xai_methods:
            status = explain_one(label, seed, ckpt, run_dir, xai)
            statuses.append(status)

    n_ok = sum(1 for s in statuses if s["ok"])
    logger.info(f"Phase 2 complete: {n_ok}/{len(statuses)} explanations succeeded")

    # Save run manifest
    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": mode,
        "configs": [c["label"] for c in configs],
        "seeds": seeds,
        "xai_methods": xai_methods,
        "trained": [{"label": t[0], "seed": t[1], "ckpt": str(t[2])} for t in trained],
        "explanations": statuses,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    logger.info(f"Manifest → {manifest_path}")

    # --- Phase 3: Aggregate canonical rules -------------------------------
    if not args.quick:
        logger.info("=== Phase 3: Aggregating canonical rules ===")
        agg_cmd = [
            sys.executable,
            str(REPO / "scripts" / "aggregate_canonical_rules.py"),
            str(results_dir),
        ]
        proc = subprocess.run(agg_cmd, cwd=str(REPO))
        if proc.returncode != 0:
            logger.error("Aggregation failed — canonical_rules_*.txt files are intact but not summarised")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print(f"\n{'='*60}")
    print(f"{'QUICK' if args.quick else 'FULL'} run complete in {elapsed/60:.1f} min")
    print(f"Results: {results_dir}/")
    print(f"  <label>_seed<N>/checkpoints/         — model checkpoints")
    print(f"  <label>_seed<N>/explanation_*.txt    — XAI output with NL rules")
    print(f"  <label>_seed<N>/canonical_rules_*.txt — all-targets canonical rules")
    print(f"  <label>_seed<N>/report_*.json        — fidelity / canonical_match_rate")
    if not args.quick:
        print(f"  canonical_summary.csv / .md          — cross-seed aggregation")
    if n_ok < len(statuses):
        print(f"\nWARNING: {len(statuses) - n_ok} explanation(s) failed. Check logs above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
