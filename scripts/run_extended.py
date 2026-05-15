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

# tqdm is a soft dependency — degrade gracefully if it's not installed so the
# pipeline never blocks on a missing progress bar.
try:
    from tqdm import tqdm as _tqdm  # type: ignore[import-not-found]
    _HAVE_TQDM = True
except ImportError:
    _tqdm = None  # type: ignore[assignment]
    _HAVE_TQDM = False

# Make the package importable when running from the repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from symbolic_xai_logic.experiments.runner import ExperimentRunner
from symbolic_xai_logic.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
EXTENDED_XAI = ["rule_extraction", "symbolic_regression", "lrp"]


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
# ---------------------------------------------------------------------------
# Per-game training budget at base scale=1.0.  Configs reference these via
# the helper builders below so all five model architectures for one game
# share identical data/training settings (only model hyperparams differ).
#
# The numbers are sized to push past the saturation point we saw at the
# previous (20k × 80–100 epoch) configuration:
#
#   Sudoku 4×4   :  50k train,  5k val/test, 200 epochs, patience 12
#   Sudoku 9×9   : 100k train, 10k val/test, 300 epochs, patience 15
#   Minesweeper  :  60k train,  6k val/test, 150 epochs, patience 10
#
# 4× more sudoku4 data (20k → 50k), 5× more sudoku9 data (20k → 100k),
# 2× more minesweeper data (30k → 60k).  Epoch caps roughly 2× across the
# board, paired with looser early-stop (lower min_delta, higher patience)
# so cosine-annealed runs actually reach the LR floor before stopping.
#
# Symbolic-XAI per-(cell,digit) pass: bump n_explain so each per-cell
# decision tree has 10k–20k samples instead of 5k, and raise n_samples in
# rule_extraction_kwargs to the same ceiling.
#
# Override at the CLI with --scale-data and --scale-epochs (multipliers).
# ---------------------------------------------------------------------------

_BUDGET = {
    "sudoku4": {
        "data":     {"n_train": 50000, "n_val": 5000, "n_test": 5000, "n_explain": 10000},
        "training": {"epochs": 200, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
                     "eval_interval": 2, "early_stop_patience": 12, "early_stop_min_delta": 1e-5},
        "rule_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 10000},
    },
    "sudoku9": {
        "data":     {"n_train": 100000, "n_val": 10000, "n_test": 10000, "n_explain": 20000},
        # NOTE: the previous (lr=1e-3, patience=15, min_delta=1e-5) config
        # plateaued at the prior — val_acc stuck at 0.111 (= 1/9 = random
        # chance) from epoch 1 onward, then early-stopped around epoch 50.
        # That meant every sudoku9 checkpoint trained so far is essentially
        # random and every downstream explanation metric on those checkpoints
        # is explaining noise.  This revised config:
        #   * Halves the initial LR (1e-3 → 5e-4) — large output dim (729)
        #     with sigmoid+BCE is hyper-sensitive to LR; 1e-3 was overshooting.
        #   * Doubles patience and zeros min_delta — let cosine annealing
        #     actually reach the LR floor before early-stop kicks in.
        #   * Doubles max epochs.
        # Retraining required to take effect; old sudoku9 checkpoints stay
        # in /results/extended but should be regenerated.
        "training": {"epochs": 600, "lr": 5e-4, "batch_size": 128, "weight_decay": 1e-4,
                     "eval_interval": 2, "early_stop_patience": 30, "early_stop_min_delta": 0.0},
        "rule_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 20000},
    },
    "minesweeper8": {
        "data":     {"n_train": 60000, "n_val": 6000, "n_test": 6000, "n_explain": 10000},
        "training": {"epochs": 150, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
                     "eval_interval": 2, "early_stop_patience": 10, "early_stop_min_delta": 1e-5},
        "rule_kwargs": {"max_depth": 4, "min_samples_leaf": 30, "n_samples": 10000},
    },
}


def _build_cfg(label: str, game_key: str, game: dict, model: dict,
               encoding: str = "one_hot") -> dict:
    """Compose one CONFIGS entry from the per-game budget + per-arch overrides."""
    b = _BUDGET[game_key]
    return {
        "label": label,
        "game": dict(game),
        "model": dict(model),
        "data": {**b["data"], "encoding": encoding},
        "training": dict(b["training"]),
        "rule_extraction_kwargs": dict(b["rule_kwargs"]),
    }


CONFIGS: list[dict] = [
    # ── Headline: medium-difficulty Sudoku 4×4 with a large MLP and lots of data
    _build_cfg("sudoku4_medium_mlp_large", "sudoku4",
               {"name": "sudoku", "size": 4, "difficulty": "medium"},
               {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1}),
    # ── Minesweeper 8×8 — local rule recovery (count + exhaustion)
    _build_cfg("minesweeper8_medium_mlp", "minesweeper8",
               {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
               {"name": "mlp", "hidden_dims": [512, 512, 256], "dropout": 0.1}),
    # ── Minesweeper 8×8 with CNN — spatial locality for mine detection
    _build_cfg("minesweeper8_medium_cnn", "minesweeper8",
               {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
               {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
               encoding="spatial"),
    # ── Minesweeper 8×8 with GNN — neighbor relationships
    # _build_cfg("minesweeper8_medium_gnn", "minesweeper8",
    #            {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
    #            {"name": "gnn", "hidden_dim": 256, "num_layers": 3, "dropout": 0.1}),
    # ── Minesweeper 8×8 with Transformer — global attention for count propagation
    _build_cfg("minesweeper8_medium_transformer", "minesweeper8",
               {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
               {"name": "transformer", "d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1}),
    # ── Sudoku 4×4 with CNN architecture
    _build_cfg("sudoku4_medium_cnn", "sudoku4",
               {"name": "sudoku", "size": 4, "difficulty": "medium"},
               {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
               encoding="spatial"),
    # ── Sudoku 4×4 with GNN architecture
    # _build_cfg("sudoku4_medium_gnn", "sudoku4",
    #            {"name": "sudoku", "size": 4, "difficulty": "medium"},
    #            {"name": "gnn", "hidden_dim": 256, "num_layers": 3, "dropout": 0.1}),
    # ── Sudoku 4×4 with Transformer architecture
    _build_cfg("sudoku4_medium_transformer", "sudoku4",
               {"name": "sudoku", "size": 4, "difficulty": "medium"},
               {"name": "transformer", "d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1}),
    # ── Sudoku 4×4 with RL architecture
    _build_cfg("sudoku4_medium_rl", "sudoku4",
               {"name": "sudoku", "size": 4, "difficulty": "medium"},
               {"name": "rl", "hidden_dims": [256, 256, 128], "dropout": 0.1, "use_value_head": True}),
    # ── Sudoku 9×9 with MLP — larger board, more complex rules
    _build_cfg("sudoku9_medium_mlp_large", "sudoku9",
               {"name": "sudoku", "size": 9, "difficulty": "medium"},
               {"name": "mlp", "hidden_dims": [1024, 512, 256], "dropout": 0.1}),
    # ── Sudoku 9×9 with CNN — spatial locality exploited
    _build_cfg("sudoku9_medium_cnn", "sudoku9",
               {"name": "sudoku", "size": 9, "difficulty": "medium"},
               {"name": "cnn", "kernel_size": 3, "dropout": 0.1},
               encoding="spatial"),
    # ── Sudoku 9×9 with GNN — constraint graph encoded
    # _build_cfg("sudoku9_medium_gnn", "sudoku9",
    #            {"name": "sudoku", "size": 9, "difficulty": "medium"},
    #            {"name": "gnn", "hidden_dim": 256, "num_layers": 4, "dropout": 0.1}),
    # ── Sudoku 9×9 with Transformer — self-attention over all cells
    # NOTE: smaller than original (d_model 256→128, layers 3→2) to fit
    # 24GB VRAM after the 5× n_train bump; sudoku9_transformer OOM'd on the
    # previous 256/3 config.  Scale back up if you have an A100.
    _build_cfg("sudoku9_medium_transformer", "sudoku9",
               {"name": "sudoku", "size": 9, "difficulty": "medium"},
               {"name": "transformer", "d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.1}),
    # ── Sudoku 9×9 with RL architecture
    _build_cfg("sudoku9_medium_rl", "sudoku9",
               {"name": "sudoku", "size": 9, "difficulty": "medium"},
               {"name": "rl", "hidden_dims": [512, 256], "dropout": 0.1, "use_value_head": True}),
    # ── Minesweeper 8×8 with RL architecture
    _build_cfg("minesweeper8_medium_rl", "minesweeper8",
               {"name": "minesweeper", "size": 8, "n_mines": 10, "difficulty": "medium"},
               {"name": "rl", "hidden_dims": [512, 512, 256], "dropout": 0.1, "use_value_head": True}),
]


def _apply_scale(cfg: dict, scale_data: float, scale_epochs: float) -> dict:
    """Multiply data volumes and epoch budgets by the given scale factors.

    Used by the CLI flags --scale-data and --scale-epochs.  Patience and
    eval_interval scale with epochs (rounded to int, min 1) so the
    early-stop semantics stay roughly proportional.  scale=1.0 is a no-op.
    """
    out = {**cfg, "data": dict(cfg["data"]), "training": dict(cfg["training"])}
    if scale_data != 1.0:
        for k in ("n_train", "n_val", "n_test", "n_explain"):
            if k in out["data"]:
                out["data"][k] = max(1, int(out["data"][k] * scale_data))
    if scale_epochs != 1.0:
        for k in ("epochs",):
            if k in out["training"]:
                out["training"][k] = max(1, int(out["training"][k] * scale_epochs))
        for k in ("early_stop_patience",):
            if k in out["training"]:
                out["training"][k] = max(1, int(out["training"][k] * scale_epochs))
    return out


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
                skip_existing: bool = True,
                n_explain: int | None = None,
                n_samples: int | None = None,
                device: str = "cpu") -> dict:
    """Run scripts/explain.py via subprocess. Returns status dict.

    When ``skip_existing=True`` (default) and the corresponding report_*.json
    already exists, the explanation is skipped — see Task P0-5.

    ``n_explain`` controls the symbolic-method explain-split size (only
    used by rule_extraction + symbolic_regression).  ``n_samples`` controls
    the fidelity test-split size.  Both default to ``explain.py``'s own
    defaults (5000 / 200) when ``None``.  Lowering ``n_explain`` is the
    single biggest wall-time lever: each subprocess regenerates puzzles
    from scratch and that dominates wall time on sudoku9.

    ``device`` forwards to explain.py's --device flag. Previously --device
    only affected training; the explain phase was always CPU regardless,
    which made sudoku9_transformer rule_extraction take 15+ hours per run.
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

    extra: list[str] = []
    if xai == "rule_extraction":
        extra.append("--all-targets")
    # Pass smaller explain-split for symbolic methods to slash data-gen time
    if n_explain is not None and xai in ("rule_extraction", "symbolic_regression"):
        extra += ["--n-explain", str(int(n_explain))]
    if n_samples is not None:
        extra += ["--n-samples", str(int(n_samples))]
    extra += ["--device", device]

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
    p.add_argument("--n-explain", type=int, default=1500,
                   help="Override explain.py's --n-explain (symbolic-method explain "
                        "split size).  Lower = faster.  Default 1500 (vs explain.py's "
                        "own default 5000) — biggest single wall-time lever for "
                        "sudoku9 where puzzle generation dominates per-run cost.")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Override explain.py's --n-samples (fidelity test-split size). "
                        "Default leaves explain.py's own default (200) in place.")
    p.add_argument("--explain-parallel", type=int, default=None,
                   help="Number of concurrent explain.py subprocesses.  Defaults to "
                        "max(--parallel * 2, 4).  Bump higher on multi-core VMs.")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Skip the final aggregation step")
    p.add_argument("--baseline-untrained", action="store_true",
                   help="Also run an untrained-NN sanity baseline for every config "
                        "(Task P1-6).  Saves under <label>_untrained_seed<N>/ so the "
                        "aggregator groups it as a separate row.")
    p.add_argument("--scale-data", type=float, default=1.0,
                   help="Multiplier on n_train / n_val / n_test / n_explain. "
                        "Use 0.1 for a quick smoke at 10%% of the configured volumes.")
    p.add_argument("--scale-epochs", type=float, default=1.0,
                   help="Multiplier on epochs and early_stop_patience.  Use 0.1 "
                        "for a 10%%-epoch smoke run; use 1.5 to give the optimizer "
                        "more room without re-editing CONFIGS.")
    p.add_argument("--resume", action="store_true",
                   help="Crash-resume mode: skip any (config, seed, xai) triple "
                        "whose checkpoint OR report already exists.  Equivalent "
                        "to running without --no-skip and without --no-skip-explain "
                        "after a partial run, but also covers the untrained baseline "
                        "phase (which lacks skip logic by default).")
    args = p.parse_args()

    # --resume forces every skip-existing flag on, regardless of other flags
    if args.resume:
        args.no_skip = False
        args.no_skip_explain = False

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

    # Apply CLI scale factors (default 1.0 = no change)
    if args.scale_data != 1.0 or args.scale_epochs != 1.0:
        logger.info(f"Scaling: data×{args.scale_data}  epochs×{args.scale_epochs}")
        configs_to_run = [
            _apply_scale(c, args.scale_data, args.scale_epochs) for c in configs_to_run
        ]

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
    explain_parallel = args.explain_parallel if args.explain_parallel is not None \
        else max(args.parallel * 2, 4)
    skip_existing_explain = not args.no_skip_explain
    logger.info(f"Explain: parallel={explain_parallel}  n_explain={args.n_explain}  "
                f"n_samples={args.n_samples or '(default)'}  device={args.device}")
    with ProcessPoolExecutor(max_workers=explain_parallel) as ex:
        futures = [
            ex.submit(explain_one, *t, skip_existing_explain,
                      args.n_explain, args.n_samples, args.device) for t in explain_tasks
        ]
        # Progress bar over completed futures — shows count, rate, ETA, and
        # rolling counters for OK / skipped / failed.  Falls back to the
        # plain iterator when tqdm isn't installed (e.g. minimal CI image).
        n_ok_running = n_skip_running = n_fail_running = 0
        iterator = as_completed(futures)
        pbar = (
            _tqdm(iterator, total=len(futures), desc="explain",
                  unit="run", dynamic_ncols=True, smoothing=0.05)
            if _HAVE_TQDM else iterator
        )
        for fut in pbar:
            result = fut.result()
            statuses.append(result)
            if result.get("skipped"):
                n_skip_running += 1
            elif result.get("ok"):
                n_ok_running += 1
            else:
                n_fail_running += 1
            if _HAVE_TQDM:
                # postfix shows the latest finished task plus running totals
                last = f"{result['label'][:24]}/s{result['seed']}/{result['xai'][:8]}"
                pbar.set_postfix_str(
                    f"ok={n_ok_running} skip={n_skip_running} "
                    f"fail={n_fail_running}  last={last}",
                    refresh=False,
                )
        if _HAVE_TQDM:
            pbar.close()

    n_ok = sum(1 for s in statuses if s["ok"])
    explain_errors = [s for s in statuses if not s["ok"]]
    logger.info(f"Phase 2 done: {n_ok}/{len(statuses)} explanations succeeded")

    # --- Phase 2.5: Untrained-NN baseline (Task P1-6) -------------------------
    if args.baseline_untrained:
        logger.info("Running untrained-NN baseline …")
        baseline_statuses: list[dict] = []
        # Same skip-existing semantics as phase-2 explain: a baseline report
        # already on disk means the previous run produced it, so skip.  This
        # is what makes the full pipeline safely re-runnable after a crash —
        # see Task #15 in the upgrade instructions.
        skip_existing_baseline = (
            args.resume or not args.no_skip_explain
        )
        for cfg in configs_to_run:
            for seed in args.seeds:
                label = cfg["label"]
                rdir = results_dir / f"{label}_untrained_seed{seed}"
                rdir.mkdir(parents=True, exist_ok=True)
                runtime_cfg = build_train_config(cfg, seed, args.device, results_dir)
                # Send each XAI through the baseline runner
                for xai in args.xai:
                    # Skip if the corresponding baseline report already exists
                    game_name = cfg["game"]["name"]
                    model_name = cfg["model"]["name"]
                    expected_report = (
                        rdir / f"report_{game_name}_{model_name}_untrained_{xai}.json"
                    )
                    if skip_existing_baseline and expected_report.exists():
                        logger.info(
                            f"[skip baseline] {label} seed{seed} {xai} — "
                            f"{expected_report.name} exists"
                        )
                        baseline_statuses.append({
                            "label": label + "_untrained",
                            "seed": seed, "xai": xai, "ok": True, "skipped": True,
                        })
                        continue

                    runtime_cfg = dict(runtime_cfg)
                    runtime_cfg["xai"] = {"name": xai}
                    try:
                        runner = ExperimentRunner(runtime_cfg, results_dir=str(rdir))
                        runner.run_baseline_untrained()
                        baseline_statuses.append({"label": label + "_untrained",
                                                  "seed": seed, "xai": xai, "ok": True,
                                                  "skipped": False})
                    except Exception as exc:
                        tb = _traceback.format_exc()
                        logger.error(f"[baseline fail] {label} seed{seed} {xai}: {exc}")
                        _append_error(
                            rdir / "errors.txt",
                            f"BASELINE FAIL  label={label}_untrained  seed={seed}  xai={xai}",
                            tb,
                        )
                        baseline_statuses.append({"label": label + "_untrained",
                                                  "seed": seed, "xai": xai, "ok": False,
                                                  "skipped": False})
        n_baseline_ok = sum(1 for s in baseline_statuses if s["ok"])
        n_baseline_skipped = sum(1 for s in baseline_statuses if s.get("skipped"))
        logger.info(
            f"Phase 2.5 done: {n_baseline_ok}/{len(baseline_statuses)} baselines succeeded "
            f"({n_baseline_skipped} skipped)"
        )
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
