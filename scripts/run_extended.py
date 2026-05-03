#!/usr/bin/env python3
"""
Extended diploma experiment matrix for canonical-rule recovery.

Premise:  when canonical_match_rate is low, the bottleneck is rarely "XAI
complexity".  Deeper trees produce longer DNF clauses, which match canonical
2-atom templates *less* often, not more.  The high-leverage knobs are:
  1. Puzzle difficulty       — easy puzzles let the network copy, not infer
  2. Training data volume    — more puzzles → cleaner rule signals
  3. Model capacity          — wide enough to encode the constraint graph
  4. XAI breadth, not depth  — explain ALL blank-cell targets, keep depth ≤ 4
  5. Cross-seed aggregation  — a "real" rule appears across multiple seeds

Experiments:
  • Sudoku 4×4:   MLP, CNN, GNN, Transformer (medium difficulty)
  • Sudoku 9×9:   MLP, CNN, GNN, Transformer (medium difficulty)
  • Minesweeper 8×8: MLP, CNN, GNN, Transformer (local rule patterns)

This script orchestrates the full extended matrix:
  Phase 1:  train one checkpoint per (config, seed) — runs train_only
  Phase 2:  for every checkpoint, run all four XAI methods; rule_extraction
            uses --all-targets so we get per-(cell,digit) canonical rules
  Phase 3:  invoke aggregate_canonical_rules.py to produce the cross-seed
            summary table (templates × frequency × seed agreement)

Run with:
    python scripts/run_extended.py --device cuda --seeds 0 1 2 3 4
    python scripts/run_extended.py --device cuda --configs sudoku4_medium_mlp
    python scripts/run_extended.py --device cuda --parallel 2     # 2 trainings at once
    python scripts/run_extended.py --device cuda --configs sudoku9_medium_mlp sudoku4_medium_cnn

Total wall time on a single RTX 3090, full matrix, 5 seeds: ~28–36 hours.
"""
from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
import traceback as _traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make the package importable when running from the repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from symbolic_xai_logic.experiments.runner import ExperimentRunner
from symbolic_xai_logic.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
EXTENDED_XAI = ["rule_extraction", "symbolic_regression", "concept_probe", "lrp"]


# ---------------------------------------------------------------------------
# Extended configs — none of these are picked up from configs/*.yaml; we build
# them in Python so the orchestration is self-contained and reproducible.
#
# Per-config dict keys:
#   label                     : unique short name, used for results dirs
#   game / model / data       : passed verbatim to ExperimentRunner
#   training                  : passed verbatim
#   rule_extraction_kwargs    : injected into the rule_extraction explainer
#                               via the run-experiment XAI config
# ---------------------------------------------------------------------------
CONFIGS: list[dict] = [
    # ── Headline: medium-difficulty Sudoku 4×4 with a large MLP and lots of data
    {
        "label": "sudoku4_medium_mlp_large",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
        "model": {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 80, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Minesweeper 8×8 — local rule recovery (count + exhaustion)
    {
        "label": "minesweeper8_medium_mlp",
        "game":  {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
        "model": {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1},
        "data":  {"n_train": 30000, "n_val": 3000, "n_test": 3000, "encoding": "one_hot"},
        "training": {
            "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Minesweeper 8×8 with CNN — spatial locality for mine detection
    {
        "label": "minesweeper8_medium_cnn",
        "game":  {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
        "model": {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
        "data":  {"n_train": 30000, "n_val": 3000, "n_test": 3000, "encoding": "spatial"},
        "training": {
            "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Minesweeper 8×8 with GNN — neighbor relationships
    {
        "label": "minesweeper8_medium_gnn",
        "game":  {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
        "model": {"name": "gnn", "hidden_dim": 256, "num_layers": 3, "dropout": 0.1},
        "data":  {"n_train": 30000, "n_val": 3000, "n_test": 3000, "encoding": "one_hot"},
        "training": {
            "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Minesweeper 8×8 with Transformer — global attention for count propagation
    {
        "label": "minesweeper8_medium_transformer",
        "game":  {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
        "model": {"name": "transformer", "d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1},
        "data":  {"n_train": 30000, "n_val": 3000, "n_test": 3000, "encoding": "one_hot"},
        "training": {
            "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 4×4 with CNN architecture
    {
        "label": "sudoku4_medium_cnn",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
        "model": {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "spatial"},
        "training": {
            "epochs": 80, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 4×4 with GNN architecture
    {
        "label": "sudoku4_medium_gnn",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
        "model": {"name": "gnn", "hidden_dim": 256, "num_layers": 3, "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 80, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 4×4 with Transformer architecture
    {
        "label": "sudoku4_medium_transformer",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
        "model": {"name": "transformer", "d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 80, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 4×4 with RL architecture
    {
        "label": "sudoku4_medium_rl",
        "game":  {"name": "sudoku", "size": 4, "difficulty": "medium"},
        "model": {"name": "rl", "hidden_dims": [256, 256, 128], "dropout": 0.1, "use_value_head": True},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 80, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 9×9 with MLP — larger board, more complex rules
    {
        "label": "sudoku9_medium_mlp_large",
        "game":  {"name": "sudoku", "size": 9, "difficulty": "medium"},
        "model": {"name": "mlp", "hidden_dims": [1024, 512, 256], "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 100, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 8, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 9×9 with CNN — spatial locality exploited
    {
        "label": "sudoku9_medium_cnn",
        "game":  {"name": "sudoku", "size": 9, "difficulty": "medium"},
        "model": {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "spatial"},
        "training": {
            "epochs": 100, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 8, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 9×9 with GNN — constraint graph encoded
    {
        "label": "sudoku9_medium_gnn",
        "game":  {"name": "sudoku", "size": 9, "difficulty": "medium"},
        "model": {"name": "gnn", "hidden_dim": 256, "num_layers": 4, "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 100, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 8, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 9×9 with Transformer — self-attention over all cells
    {
        "label": "sudoku9_medium_transformer",
        "game":  {"name": "sudoku", "size": 9, "difficulty": "medium"},
        "model": {"name": "transformer", "d_model": 256, "nhead": 8, "num_layers": 3, "dropout": 0.1},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 100, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 8, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Sudoku 9×9 with RL architecture
    {
        "label": "sudoku9_medium_rl",
        "game":  {"name": "sudoku", "size": 9, "difficulty": "medium"},
        "model": {"name": "rl", "hidden_dims": [512, 256], "dropout": 0.1, "use_value_head": True},
        "data":  {"n_train": 20000, "n_val": 2000, "n_test": 2000, "encoding": "one_hot"},
        "training": {
            "epochs": 100, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 8, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
    # ── Minesweeper 8×8 with RL architecture
    {
        "label": "minesweeper8_medium_rl",
        "game":  {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
        "model": {"name": "rl", "hidden_dims": [512, 512, 256], "dropout": 0.1, "use_value_head": True},
        "data":  {"n_train": 30000, "n_val": 3000, "n_test": 3000, "encoding": "one_hot"},
        "training": {
            "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
            "eval_interval": 2, "early_stop_patience": 5, "early_stop_min_delta": 1e-4,
        },
        "rule_extraction_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 5000},
    },
]


def build_train_config(cfg: dict, seed: int, device: str, results_dir: Path) -> dict:
    """Compose the runtime config dict for one (cfg, seed) train_only call."""
    rdir = results_dir / f"{cfg['label']}_seed{seed}"
    rdir.mkdir(parents=True, exist_ok=True)
    return {
        "seed": seed,
        "device": device,
        "game": dict(cfg["game"]),
        "model": dict(cfg["model"]),
        "data": dict(cfg["data"]),
        "training": {**cfg["training"], "checkpoint_dir": str(rdir / "checkpoints")},
        # train_only ignores xai but the field is required by ExperimentRunner.
        "xai": {"name": "rule_extraction"},
    }


def _append_error(path: Path, header: str, detail: str) -> None:
    """Append one error block to a text file (safe for concurrent writes via line-buffering)."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = f"\n{'='*72}\n[{ts}] {header}\n{'-'*72}\n{detail}\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(block)


def train_one(cfg: dict, seed: int, device: str, results_dir: Path,
              skip_existing: bool = True) -> tuple[str, int, str, str, str | None]:
    """Train one (config, seed).

    Returns (label, seed, ckpt_path, run_dir, error_str).
    On success error_str is None; on failure ckpt_path and run_dir are empty strings.
    """
    label = cfg["label"]
    rdir = results_dir / f"{label}_seed{seed}"
    expected_ckpt = rdir / "checkpoints" / f"{cfg['game']['name']}{cfg['game']['size']}_best.pt"

    if skip_existing and expected_ckpt.exists():
        logger.info(f"[skip] {label} seed{seed} — checkpoint exists at {expected_ckpt}")
        return (label, seed, str(expected_ckpt), str(rdir), None)

    logger.info(f"[train] {label} seed{seed}")
    runtime_cfg = build_train_config(cfg, seed, device, results_dir)
    try:
        runner = ExperimentRunner(runtime_cfg, results_dir=str(rdir))
        ckpt_path = runner.train_only()
        return (label, seed, str(ckpt_path), str(rdir), None)
    except Exception as exc:
        tb = _traceback.format_exc()
        logger.error(f"[train fail] {label} seed{seed}: {exc}")
        # Write to per-seed dir so subprocess failures are always captured on disk
        rdir.mkdir(parents=True, exist_ok=True)
        _append_error(
            rdir / "errors.txt",
            f"TRAIN FAIL  label={label}  seed={seed}",
            tb,
        )
        return (label, seed, "", "", tb)


def _explain_already_done(rdir: str, label: str, xai: str) -> Path | None:
    """Return the existing report path if explanation has already produced one.

    Looks for any ``report_*_<xai>.json`` in ``rdir`` — the game name in the
    filename varies with config so we glob.  This lets `run_extended.py` skip
    (config, seed, xai) triples that completed in a prior run, which is the
    P0-5 fix: many directories under results/extended/ have checkpoints but
    no report files because the explain phase was interrupted.
    """
    rd = Path(rdir)
    matches = list(rd.glob(f"report_*_{xai}.json"))
    return matches[0] if matches else None


def explain_one(label: str, seed: int, ckpt: str, rdir: str, xai: str,
                skip_existing: bool = True) -> dict:
    """Run scripts/explain.py via subprocess. Returns status dict.

    When ``skip_existing=True`` (default) and the corresponding report_*.json
    already exists, the explanation is skipped — see Task P0-5.
    """
    if skip_existing:
        existing = _explain_already_done(rdir, label, xai)
        if existing is not None:
            logger.info(f"[skip explain] {label} seed{seed} {xai} — {existing.name} exists")
            return {
                "label": label, "seed": seed, "xai": xai,
                "ok": True, "elapsed_s": 0.0, "stderr": "",
                "skipped": True,
            }

    extra = ["--all-targets"] if xai == "rule_extraction" else []
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "explain.py"),
        "--checkpoint", ckpt,
        "--xai", xai,
        "--results-dir", rdir,
        *extra,
    ]
    logger.info(f"[explain] {label} seed{seed} {xai}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    elapsed = time.time() - t0
    stderr_tail = proc.stderr[-4000:] if proc.stderr else ""
    ok = proc.returncode == 0
    if not ok:
        logger.error(f"[explain fail] {label} seed{seed} {xai}\n{stderr_tail}")
        # Write to per-seed dir so subprocess failures are always captured on disk
        _append_error(
            Path(rdir) / "errors.txt",
            f"EXPLAIN FAIL  label={label}  seed={seed}  xai={xai}",
            stderr_tail,
        )
    return {
        "label": label, "seed": seed, "xai": xai,
        "ok": ok,
        "elapsed_s": round(elapsed, 1),
        "stderr": stderr_tail if not ok else "",
        "skipped": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Extended diploma experiment matrix")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--results-dir", default="results/extended")
    p.add_argument("--parallel", type=int, default=1,
                   help="Concurrent training jobs (1=sequential, recommended on a single GPU)")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Subset of config labels to run; default = all")
    p.add_argument("--no-skip", action="store_true",
                   help="Re-train even if a checkpoint already exists")
    p.add_argument("--no-skip-explain", action="store_true",
                   help="Re-run explain even if report_*_<xai>.json already exists "
                        "(default: skip, see Task P0-5)")
    p.add_argument("--explain-only", action="store_true",
                   help="Skip training entirely; assume checkpoints already exist on disk "
                        "and only fill in missing explanations.  Useful for back-filling "
                        "the gnn/transformer/rl configs whose explain phase was never run.")
    p.add_argument("--xai", nargs="+", default=EXTENDED_XAI,
                   help=f"XAI methods to run; default = {EXTENDED_XAI}")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Skip the final aggregation step")
    p.add_argument("--baseline-untrained", action="store_true",
                   help="Also run an untrained-NN sanity baseline for every config "
                        "(Task P1-6).  Saves under <label>_untrained_seed<N>/ so the "
                        "aggregator groups it as a separate row.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Attach a file handler so all ERROR+ messages from the parent process land in errors.txt
    errors_path = results_dir / "errors.txt"
    _fh = logging.FileHandler(errors_path, mode="a", encoding="utf-8")
    _fh.setLevel(logging.ERROR)
    _fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(_fh)

    configs_to_run = (
        [c for c in CONFIGS if c["label"] in args.configs]
        if args.configs else CONFIGS
    )
    if not configs_to_run:
        logger.error(f"No matching configs. Available: {[c['label'] for c in CONFIGS]}")
        sys.exit(1)

    logger.info(f"Configs: {[c['label'] for c in configs_to_run]}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"XAI methods: {args.xai}")
    logger.info(f"Results dir: {results_dir}")

    t0 = time.time()

    # --- Phase 1: Training -----------------------------------------------------
    train_tasks = [(cfg, seed) for cfg in configs_to_run for seed in args.seeds]
    trained: list = []   # successful: (label, seed, ckpt, rdir)
    train_errors: list[dict] = []

    def _collect_train(res: tuple) -> None:
        label, seed, ckpt, rdir, err = res
        if err is None:
            trained.append((label, seed, ckpt, rdir))
        else:
            train_errors.append({"label": label, "seed": seed, "traceback": err})
            _append_error(errors_path, f"TRAIN FAIL  label={label}  seed={seed}", err)

    if args.explain_only:
        # Don't train; just discover existing checkpoints (Task P0-5).
        for cfg, seed in train_tasks:
            label = cfg["label"]
            rdir = results_dir / f"{label}_seed{seed}"
            ckpt = rdir / "checkpoints" / f"{cfg['game']['name']}{cfg['game']['size']}_best.pt"
            if ckpt.exists():
                trained.append((label, seed, str(ckpt), str(rdir)))
            else:
                logger.warning(f"[explain-only] no checkpoint for {label} seed{seed} — skipping")
    elif args.parallel <= 1:
        for cfg, seed in train_tasks:
            _collect_train(train_one(cfg, seed, args.device, results_dir, skip_existing=not args.no_skip))
    else:
        # ProcessPoolExecutor with CUDA: each child re-imports torch and grabs the GPU.
        # Multiple small models share VRAM fine on a 3090; cap concurrency at 2-3 to be safe.
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(train_one, cfg, seed, args.device, results_dir, not args.no_skip):
                    (cfg["label"], seed)
                for cfg, seed in train_tasks
            }
            for fut in as_completed(futures):
                _collect_train(fut.result())

    logger.info(f"Phase 1 done: {len(trained)}/{len(train_tasks)} trainings succeeded")

    # --- Phase 2: Explanation -------------------------------------------------
    explain_tasks = [
        (label, seed, ckpt, rdir, xai)
        for (label, seed, ckpt, rdir) in trained
        for xai in args.xai
    ]
    statuses: list[dict] = []

    # Explanation parallelism is safer than training: most XAI methods are CPU-bound
    # for our model sizes. Run a few in parallel even on a single GPU.
    explain_parallel = max(args.parallel * 2, 4)
    skip_existing_explain = not args.no_skip_explain
    with ProcessPoolExecutor(max_workers=explain_parallel) as ex:
        futures = [
            ex.submit(explain_one, *t, skip_existing_explain) for t in explain_tasks
        ]
        for fut in as_completed(futures):
            statuses.append(fut.result())

    n_ok = sum(1 for s in statuses if s["ok"])
    explain_errors = [s for s in statuses if not s["ok"]]
    logger.info(f"Phase 2 done: {n_ok}/{len(statuses)} explanations succeeded")

    # --- Phase 2.5: Untrained-NN baseline (Task P1-6) -------------------------
    if args.baseline_untrained:
        logger.info("Running untrained-NN baseline …")
        baseline_statuses: list[dict] = []
        for cfg in configs_to_run:
            for seed in args.seeds:
                label = cfg["label"]
                rdir = results_dir / f"{label}_untrained_seed{seed}"
                rdir.mkdir(parents=True, exist_ok=True)
                runtime_cfg = build_train_config(cfg, seed, args.device, results_dir)
                # Send each XAI through the baseline runner
                for xai in args.xai:
                    runtime_cfg = dict(runtime_cfg)
                    runtime_cfg["xai"] = {"name": xai}
                    try:
                        runner = ExperimentRunner(runtime_cfg, results_dir=str(rdir))
                        runner.run_baseline_untrained()
                        baseline_statuses.append({"label": label + "_untrained",
                                                  "seed": seed, "xai": xai, "ok": True})
                    except Exception as exc:
                        tb = _traceback.format_exc()
                        logger.error(f"[baseline fail] {label} seed{seed} {xai}: {exc}")
                        _append_error(
                            rdir / "errors.txt",
                            f"BASELINE FAIL  label={label}_untrained  seed={seed}  xai={xai}",
                            tb,
                        )
                        baseline_statuses.append({"label": label + "_untrained",
                                                  "seed": seed, "xai": xai, "ok": False})
        n_baseline_ok = sum(1 for s in baseline_statuses if s["ok"])
        logger.info(f"Phase 2.5 done: {n_baseline_ok}/{len(baseline_statuses)} baselines succeeded")
        statuses.extend(baseline_statuses)

    # Append explain failures to the consolidated errors file
    for s in explain_errors:
        _append_error(
            errors_path,
            f"EXPLAIN FAIL  label={s['label']}  seed={s['seed']}  xai={s['xai']}",
            s.get("stderr", "(no stderr captured)"),
        )

    # Save phase-2 status log
    status_path = results_dir / "extended_run_status.json"
    status_path.write_text(json.dumps({
        "trained": [{"label": t[0], "seed": t[1], "ckpt": t[2]} for t in trained],
        "train_errors": [{"label": e["label"], "seed": e["seed"]} for e in train_errors],
        "explained": statuses,
        "explain_errors": [{"label": e["label"], "seed": e["seed"], "xai": e["xai"]} for e in explain_errors],
        "elapsed_s": round(time.time() - t0, 1),
        "configs": [c["label"] for c in configs_to_run],
        "seeds": args.seeds,
    }, indent=2))
    logger.info(f"Status log → {status_path}")
    if train_errors or explain_errors:
        logger.info(f"Error log   → {errors_path}  ({len(train_errors)} train, {len(explain_errors)} explain failures)")

    # --- Phase 3: Aggregate ---------------------------------------------------
    if not args.no_aggregate:
        agg_cmd = [
            sys.executable,
            str(REPO / "scripts" / "aggregate_canonical_rules.py"),
            str(results_dir),
        ]
        logger.info(f"[aggregate] {' '.join(agg_cmd)}")
        proc = subprocess.run(agg_cmd, cwd=str(REPO))
        if proc.returncode != 0:
            logger.error("Aggregation step failed; rules text files were produced but not summarised.")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nResults under: {results_dir}/")
    print(f"  - <label>_seed<N>/checkpoints/        — model checkpoints")
    print(f"  - <label>_seed<N>/explanation_*.txt   — full XAI output incl. NL rules")
    print(f"  - <label>_seed<N>/canonical_rules_*.txt — all-targets canonical rules per seed")
    print(f"  - <label>_seed<N>/report_*_*_*.json   — fidelity / canonical_match_rate JSON")
    print(f"  - <label>_seed<N>/errors.txt          — per-run errors (if any)")
    print(f"  - errors.txt                          — consolidated error log for the whole run")
    print(f"  - canonical_summary.csv               — cross-seed aggregated table (after Phase 3)")
    print(f"  - canonical_summary.md                — human-readable cross-seed report")


if __name__ == "__main__":
    main()
