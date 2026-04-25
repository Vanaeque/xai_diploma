"""Reproduce all results: train + explain all (game, model, xai) combinations.

Usage:
    python scripts/reproduce_all.py          # full 5-seed run
    python scripts/reproduce_all.py --quick  # fast smoke run (Minesweeper-8x8 + MLP)
    python scripts/reproduce_all.py --seeds 42 123 456  # custom seeds
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from symbolic_xai_logic.experiments.runner import ExperimentRunner, DEFAULT_XAI_METHODS
from symbolic_xai_logic.experiments.compare import compare_experiments, aggregate_seeds
from symbolic_xai_logic.symbolic.fidelity import FidelityReport
from symbolic_xai_logic.utils.io import save_json
from symbolic_xai_logic.utils.logging import get_logger
from symbolic_xai_logic.viz.plots import plot_training_curves, plot_fidelity_heatmap

logger = get_logger(__name__)

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce all results")
    p.add_argument("--quick", action="store_true", help="Quick smoke run (Minesweeper-8x8 + MLP, 1 seed)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# Diploma experiment matrix per COMPUTE_SETUP.md
# Four XAI methods are the default (DEFAULT_XAI_METHODS); LIME/SHAP excluded from
# default reproduction but remain available via --xai lime/shap individually.
_HEADLINE = [
    # 5-seed headline configs — all four methods
    {"game": "sudoku",      "size": 4,  "model": "mlp", "epochs": 30, "n_train": 5000},
    {"game": "minesweeper", "size": 8,  "n_mines": 10, "model": "mlp", "epochs": 30, "n_train": 5000},
]
_EXTENDED = [
    # 3-seed extended configs — all four methods
    {"game": "sudoku",      "size": 9,  "model": "gnn", "epochs": 50, "n_train": 20000},
    {"game": "minesweeper", "size": 8,  "n_mines": 10, "model": "cnn", "epochs": 40, "n_train": 10000},
    {"game": "minesweeper", "size": 16, "n_mines": 40, "model": "cnn", "epochs": 60, "n_train": 20000},
]

EXPERIMENTS_FULL = [
    {**base, "xai": xai}
    for base in _HEADLINE + _EXTENDED
    for xai in DEFAULT_XAI_METHODS
]

EXPERIMENTS_QUICK = [
    {"game": "minesweeper", "size": 4, "n_mines": 5, "model": "mlp",
     "xai": "rule_extraction", "epochs": 30, "n_train": 2000},
]


def make_config(exp: dict, seed: int, device: str, results_dir: str) -> dict:
    game_cfg: dict = {"name": exp["game"], "size": exp["size"], "difficulty": "easy"}
    if exp["game"] == "sat3":
        game_cfg["n_vars"] = exp.get("size", 10)
        game_cfg["n_clauses"] = 40
    elif exp["game"] == "knights_knaves":
        game_cfg["n_agents"] = exp.get("size", 4)
    elif exp["game"] == "minesweeper" and "n_mines" in exp:
        game_cfg["n_mines"] = exp["n_mines"]

    model_name = exp["model"]
    encoding = "spatial" if (model_name == "cnn" and exp["game"] == "minesweeper") else "one_hot"

    return {
        "seed": seed,
        "device": device,
        "game": game_cfg,
        "model": {"name": model_name},
        "xai": {"name": exp["xai"]},
        "data": {
            "n_train": exp["n_train"],
            "n_val": max(200, exp["n_train"] // 10),
            "n_test": max(200, exp["n_train"] // 10),
            "encoding": encoding,
        },
        "training": {
            "epochs": exp["epochs"],
            "lr": 1e-3,
            "batch_size": 64,
            "weight_decay": 1e-4,
            "eval_interval": max(1, exp["epochs"] // 5),
            "checkpoint_dir": f"{results_dir}/checkpoints",
        },
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS_QUICK if args.quick else EXPERIMENTS_FULL
    if args.quick:
        seeds = [42]
    else:
        # Headline configs use all 5 seeds; extended configs use first 3
        seeds = args.seeds

    all_reports: list[FidelityReport] = []
    histories: dict = {}

    t0 = time.time()
    run_idx = 0
    n_total = sum(
        len(seeds if (args.quick or (e["game"], e["size"]) in _headline_keys) else seeds[:3])
        for e in experiments
    )

    _headline_keys = {(b["game"], b["size"]) for b in _HEADLINE} if not args.quick else set()

    for exp in experiments:
        exp_label = f"{exp['game']}{exp['size']}-{exp['model']}-{exp['xai']}"
        is_headline = (exp["game"], exp["size"]) in _headline_keys
        exp_seeds = seeds if (args.quick or is_headline) else seeds[:3]
        for seed in exp_seeds:
            run_idx += 1
            label = f"{exp_label}-seed{seed}"
            logger.info(f"[{run_idx}/{n_total}] {label}")

            cfg = make_config(exp, seed, args.device, args.results_dir)
            try:
                runner = ExperimentRunner(cfg, results_dir=args.results_dir)
                report = runner.run()
                all_reports.append(report)

                history_path = results_dir / f"history_{exp['game']}_{exp['model']}.json"
                if history_path.exists():
                    import json
                    with open(history_path) as f:
                        histories[label] = json.load(f)
            except Exception as e:
                logger.error(f"Experiment {label} failed: {e}")
                import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    logger.info(f"All experiments done in {elapsed:.1f}s")

    # Per-run summary
    df_all = compare_experiments(all_reports)
    df_all.to_csv(results_dir / "summary_all_seeds.csv", index=False)

    # Aggregated (mean ± std across seeds)
    df_agg = aggregate_seeds(all_reports)
    df_agg.to_csv(results_dir / "summary.csv", index=False)
    logger.info(f"Aggregated summary saved to {results_dir / 'summary.csv'}")

    # Plots
    for label, hist in histories.items():
        plot_training_curves(hist, save_path=str(results_dir / f"training_{label}.png"))

    if not df_agg.empty:
        plot_fidelity_heatmap(df_agg, save_path=str(results_dir / "fidelity_heatmap.png"))

    # Markdown report
    report_path = results_dir / "report.md"
    _write_report(report_path, df_agg, df_all, all_reports, elapsed, args.quick, seeds)
    logger.info(f"Report written to {report_path}")
    print(f"\nDone. Results in {results_dir}/")
    print("  report.md, summary.csv, summary_all_seeds.csv, plots generated.")


def _write_report(
    path: Path,
    df_agg: pd.DataFrame,
    df_all: pd.DataFrame,
    reports: list,
    elapsed: float,
    quick: bool,
    seeds: list[int],
) -> None:
    lines = [
        "# Symbolic XAI for Logic Games — Results",
        "",
        f"**Run type:** {'Quick (smoke)' if quick else 'Full diploma matrix'}  ",
        f"**Seeds:** {seeds}  ",
        f"**Wall time:** {elapsed:.1f}s  ",
        "",
        "## Aggregated Summary (mean ± std across seeds)",
        "",
    ]

    if not df_agg.empty:
        lines.append(df_agg.to_markdown(index=False))
    else:
        lines.append("*No results generated.*")

    lines += ["", "## Per-Seed Results", ""]
    if not df_all.empty:
        lines.append(df_all.to_markdown(index=False))

    lines += ["", "## Key Findings", ""]
    for r in reports:
        d = r.to_dict()
        seed_str = f" seed={d.get('seed', '?')}" if not quick else ""
        lines.append(
            f"- **{d['game']} / {d['model']} / {d['xai']}{seed_str}**: "
            f"NN acc={d['nn_accuracy']:.3f}, "
            f"fidelity={d['fidelity_to_nn']:.3f}, "
            f"GT agreement={d['agreement_gt']:.3f}"
        )

    lines += ["", "## Done Criteria Check", ""]
    sudoku_mlp_rule = [
        r for r in reports
        if "sudoku" in r.game_name and r.model_name == "mlp" and r.xai_name == "rule_extraction"
    ]
    if sudoku_mlp_rule:
        fidels = [r.explainer_fidelity_to_nn for r in sudoku_mlp_rule]
        mean_f = sum(fidels) / len(fidels)
        passed = mean_f >= 0.85
        lines.append(f"- Rule extraction fidelity ≥ 0.85: {'PASS' if passed else 'FAIL'} "
                     f"(mean={mean_f:.3f} over {len(fidels)} seeds)")

    msw_cnn_rule = [
        r for r in reports
        if "minesweeper" in r.game_name and r.model_name == "cnn" and r.xai_name == "rule_extraction"
    ]
    if msw_cnn_rule:
        fidels = [r.explainer_fidelity_to_nn for r in msw_cnn_rule]
        mean_f = sum(fidels) / len(fidels)
        passed = mean_f >= 0.85
        lines.append(f"- Minesweeper CNN rule fidelity ≥ 0.85: {'PASS' if passed else 'FAIL'} "
                     f"(mean={mean_f:.3f} over {len(fidels)} seeds)")

    lines += ["", "## Plots", "", "![Fidelity Heatmap](fidelity_heatmap.png)", ""]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
