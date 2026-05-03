"""Orchestrates the full pipeline from config."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.seeding import set_global_seed
from ..utils.io import save_json, config_hash
from ..games import get_game
from ..data import generate_dataset, build_dataloaders
from ..models import get_model
from ..training import Trainer
from ..xai import get_explainer
from ..symbolic.fidelity import compute_fidelity, FidelityReport

logger = get_logger(__name__)

DEFAULT_XAI_METHODS = ["rule_extraction", "symbolic_regression", "concept_probe", "lrp"]


class ExperimentRunner:
    """Run the full data→train→explain→report pipeline."""

    def __init__(self, config: dict, results_dir: str = "results"):
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._cfg_hash = config_hash(config)

    def run(self) -> FidelityReport:
        cfg = self.config
        seed = cfg.get("seed", 42)
        set_global_seed(seed)

        game_cfg = cfg.get("game", {})
        game_name = game_cfg.get("name", "sudoku")
        game_size = game_cfg.get("size", 4)
        difficulty = game_cfg.get("difficulty", "medium")

        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "mlp")

        xai_cfg = cfg.get("xai", {})
        xai_name = xai_cfg.get("name", "rule_extraction")

        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})

        logger.info(f"Starting experiment: game={game_name}, model={model_name}, xai={xai_name}")

        # Build game
        game_kwargs = {k: v for k, v in game_cfg.items() if k != "name"}
        game = get_game(game_name, **game_kwargs)

        # Generate data — CNN+Minesweeper uses spatial encoding
        default_encoding = "spatial" if (model_name == "cnn" and game_name == "minesweeper") else "one_hot"
        encoding = data_cfg.get("encoding", default_encoding)
        # Dedicated explanation set (Task P0-3): symbolic XAI methods need many
        # more samples than test (400-2000) to give per-cell trees enough
        # blanks to fit.  Default 5000 matches existing rule_extraction_kwargs.
        # Symbolic methods only — attribution methods just use X_test below.
        n_explain = data_cfg.get("n_explain", 5000) if xai_name in (
            "rule_extraction", "symbolic_regression"
        ) else 0
        splits = generate_dataset(
            game,
            n_train=data_cfg.get("n_train", 20000),
            n_val=data_cfg.get("n_val", 400),
            n_test=data_cfg.get("n_test", 400),
            n_explain=n_explain,
            difficulty=difficulty,
            encoding=encoding,
            seed=seed,
        )

        loaders = build_dataloaders(
            {k: v for k, v in splits.items() if k in ("train", "val", "test")},
            batch_size=training_cfg.get("batch_size", 64),
        )

        # Build model — CNN needs grid_size + n_channels from spatial encoding
        if model_name == "cnn" or encoding == "spatial":
            from ..games.minesweeper import MinesweeperGame, N_SPATIAL_CHANNELS
            if encoding == "spatial" and hasattr(game, "spatial_input_dim"):
                # Use spatial encoding if available
                input_dim = game.spatial_input_dim
                grid_size = game.size
                if isinstance(game, MinesweeperGame):
                    n_channels = N_SPATIAL_CHANNELS
                else:
                    # For other games, infer n_channels from spatial_input_dim
                    n_channels = input_dim // (grid_size * grid_size)
            elif isinstance(game, MinesweeperGame):
                input_dim = game.spatial_input_dim
                grid_size = game.size
                n_channels = N_SPATIAL_CHANNELS
            else:
                input_dim = game.input_dim
                grid_size = getattr(game, "size", 8)
                n_channels = input_dim // (grid_size * grid_size)
        else:
            input_dim = game.input_dim
            grid_size = None
            n_channels = None

        output_dim = game.output_dim
        model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "size")}
        if model_name == "cnn":
            if grid_size is not None:
                model_kwargs.setdefault("grid_size", grid_size)
            if n_channels is not None:
                model_kwargs.setdefault("n_channels", n_channels)
        model = get_model(model_name, input_dim=input_dim, output_dim=output_dim, **model_kwargs)
        logger.info(f"Model: {model_name}, params={sum(p.numel() for p in model.parameters())}")

        # Train
        trainer = Trainer(
            model=model,
            game=game,
            device=cfg.get("device", "cpu"),
            lr=training_cfg.get("lr", 1e-3),
            weight_decay=training_cfg.get("weight_decay", 1e-4),
            epochs=training_cfg.get("epochs", 80),
            eval_interval=training_cfg.get("eval_interval", 5),
            checkpoint_dir=training_cfg.get("checkpoint_dir", str(self.results_dir / "checkpoints")),
            config=cfg,
            seed=seed,
            early_stop_patience=training_cfg.get("early_stop_patience", 5),
            early_stop_min_delta=training_cfg.get("early_stop_min_delta", 0.0),
        )
        history = trainer.train(loaders["train"], loaders["val"])
        save_json(history, self.results_dir / f"history_{game_name}_{model_name}_seed{seed}.json")

        # Explain
        xai_kwargs = {k: v for k, v in xai_cfg.items() if k != "name"}
        explainer = get_explainer(xai_name, model=model, game=game, **xai_kwargs)

        # Compute fidelity
        X_test = splits["test"]["X"]
        y_test = splits["test"]["y"]
        solutions_test = splits["test"].get("solutions")
        puzzles_test = splits["test"].get("puzzles")

        # Optional explain split (Task P0-3): only symbolic XAI uses it
        explain_split = splits.get("explain")
        X_explain = explain_split["X"] if explain_split is not None else None
        sols_explain = explain_split.get("solutions") if explain_split is not None else None

        # Load metrics config (plain dict from cfg, or empty)
        metrics_cfg = cfg.get("metrics", {})

        report = compute_fidelity(
            model=model,
            game=game,
            explainer=explainer,
            X_test=X_test,
            y_test=y_test,
            solutions_test=solutions_test,
            puzzles_test=puzzles_test,
            model_name=model_name,
            xai_name=xai_name,
            config_hash=self._cfg_hash,
            seed=seed,
            metrics_cfg=metrics_cfg,
            X_explain=X_explain,
            solutions_explain=sols_explain,
        )

        # All canonical-rule fields (canonical_match_rate, by_template,
        # templates_fired, template_coverage, matched, total) are merged into
        # FidelityReport.extra by compute_fidelity, then flattened by to_dict.
        # No additional NL pass needed here.
        report_dict = report.to_dict()

        # Promote canonical_stats nested dict to top level for backward-compat
        # readers that look for top-level n_matched / n_total / canonical_match_rate.
        if "canonical_stats" in report_dict:
            cs = report_dict.pop("canonical_stats")
            for k, v in cs.items():
                report_dict.setdefault(k, v)

        logger.info(f"Results: {report_dict}")
        save_json(report_dict, self.results_dir / f"report_{game_name}_{model_name}_{xai_name}.json")

        return report

    def train_only(self) -> str:
        """
        Run data generation + training only; return the path to the best checkpoint.
        No explainer is invoked.  Use scripts/explain.py to explain the checkpoint later.
        """
        cfg = self.config
        seed = cfg.get("seed", 42)
        set_global_seed(seed)

        game_cfg = cfg.get("game", {})
        game_name = game_cfg.get("name", "sudoku")
        difficulty = game_cfg.get("difficulty", "medium")
        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "mlp")
        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})

        game_kwargs = {k: v for k, v in game_cfg.items() if k != "name"}
        game = get_game(game_name, **game_kwargs)

        default_encoding = "spatial" if (model_name == "cnn" and game_name == "minesweeper") else "one_hot"
        encoding = data_cfg.get("encoding", default_encoding)
        splits = generate_dataset(
            game,
            n_train=data_cfg.get("n_train", 20000),
            n_val=data_cfg.get("n_val", 400),
            n_test=data_cfg.get("n_test", 400),
            difficulty=difficulty,
            encoding=encoding,
            seed=seed,
        )
        loaders = build_dataloaders(splits, batch_size=training_cfg.get("batch_size", 64))

        if model_name == "cnn" or encoding == "spatial":
            from ..games.minesweeper import MinesweeperGame, N_SPATIAL_CHANNELS
            if encoding == "spatial" and hasattr(game, "spatial_input_dim"):
                # Use spatial encoding if available
                input_dim = game.spatial_input_dim
                grid_size = game.size
                if isinstance(game, MinesweeperGame):
                    n_channels = N_SPATIAL_CHANNELS
                else:
                    # For other games, infer n_channels from spatial_input_dim
                    n_channels = input_dim // (grid_size * grid_size)
            elif isinstance(game, MinesweeperGame):
                input_dim = game.spatial_input_dim
                grid_size, n_channels = game.size, N_SPATIAL_CHANNELS
            else:
                input_dim = game.input_dim
                grid_size = getattr(game, "size", 8)
                n_channels = input_dim // (grid_size * grid_size)
        else:
            input_dim, grid_size, n_channels = game.input_dim, None, None

        output_dim = game.output_dim
        model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "size")}
        if model_name == "cnn":
            if grid_size is not None:
                model_kwargs.setdefault("grid_size", grid_size)
            if n_channels is not None:
                model_kwargs.setdefault("n_channels", n_channels)
        model = get_model(model_name, input_dim=input_dim, output_dim=output_dim, **model_kwargs)
        
        # Validate CNN dimensions before training
        if model_name == "cnn":
            import torch
            # Test reshape: input_dim should equal n_channels * grid_size * grid_size
            expected_elements = n_channels * grid_size * grid_size
            if input_dim != expected_elements:
                raise ValueError(
                    f"[{game_name}] CNN input dimension mismatch:\n"
                    f"  input_dim={input_dim} (from data)\n"
                    f"  n_channels={n_channels} × grid_size={grid_size} × grid_size={grid_size}"
                    f" = {expected_elements}\n"
                    f"  encoding={encoding}"
                )
            # Test forward pass with dummy batch (CPU only — model moves to device later in Trainer)
            try:
                dummy = torch.randn(2, input_dim)
                _ = model(dummy)
            except Exception as e:
                raise RuntimeError(
                    f"[{game_name}] CNN forward pass failed with input_dim={input_dim}:\n{e}"
                ) from e

        checkpoint_dir = training_cfg.get("checkpoint_dir", str(self.results_dir / "checkpoints"))
        trainer = Trainer(
            model=model,
            game=game,
            device=cfg.get("device", "cpu"),
            lr=training_cfg.get("lr", 1e-3),
            weight_decay=training_cfg.get("weight_decay", 1e-4),
            epochs=training_cfg.get("epochs", 80),
            eval_interval=training_cfg.get("eval_interval", 5),
            checkpoint_dir=checkpoint_dir,
            config=cfg,
            seed=seed,
            early_stop_patience=training_cfg.get("early_stop_patience", 5),
            early_stop_min_delta=training_cfg.get("early_stop_min_delta", 0.0),
        )
        history = trainer.train(loaders["train"], loaders["val"])
        save_json(history, self.results_dir / f"history_{game_name}_{model_name}_seed{seed}.json")

        ckpt_path = str(Path(checkpoint_dir) / f"{game_name}{getattr(game, 'size', '')}_best.pt")
        logger.info(f"Training complete. Checkpoint: {ckpt_path}")
        return ckpt_path

    def run_baseline_untrained(self) -> FidelityReport:
        """Run the explain+report pipeline with an UNTRAINED random-init model.

        This is the sanity control for the diploma's central hypothesis: a
        trained network internally encodes rules.  If an untrained network
        produces a similar canonical_match_rate, the trained number is
        meaningless.  See Task P1-6 in copilot_upgrade_instructions.md.

        Identical to ``run()`` minus the training step — we build a fresh
        random model with the same architecture and immediately invoke the
        explainer.  The resulting report is saved with the same filename
        scheme so the aggregator picks it up automatically; the directory
        passed in via ``results_dir`` should be `<label>_untrained_seed<N>`
        so the aggregator groups it as a separate config.
        """
        cfg = self.config
        seed = cfg.get("seed", 42)
        set_global_seed(seed)

        game_cfg = cfg.get("game", {})
        game_name = game_cfg.get("name", "sudoku")
        difficulty = game_cfg.get("difficulty", "medium")
        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "mlp")
        xai_cfg = cfg.get("xai", {})
        xai_name = xai_cfg.get("name", "rule_extraction")
        data_cfg = cfg.get("data", {})

        game_kwargs = {k: v for k, v in game_cfg.items() if k != "name"}
        game = get_game(game_name, **game_kwargs)

        default_encoding = "spatial" if (model_name == "cnn" and game_name == "minesweeper") else "one_hot"
        encoding = data_cfg.get("encoding", default_encoding)
        n_explain = data_cfg.get("n_explain", 5000) if xai_name in (
            "rule_extraction", "symbolic_regression"
        ) else 0
        splits = generate_dataset(
            game,
            n_train=data_cfg.get("n_train", 200),   # untrained: small splits OK
            n_val=data_cfg.get("n_val", 100),
            n_test=data_cfg.get("n_test", 400),
            n_explain=n_explain,
            difficulty=difficulty,
            encoding=encoding,
            seed=seed,
        )

        # Same model-building branch as run(), but no Trainer call
        if model_name == "cnn" or encoding == "spatial":
            from ..games.minesweeper import MinesweeperGame, N_SPATIAL_CHANNELS
            if encoding == "spatial" and hasattr(game, "spatial_input_dim"):
                input_dim = game.spatial_input_dim
                grid_size = game.size
                if isinstance(game, MinesweeperGame):
                    n_channels = N_SPATIAL_CHANNELS
                else:
                    n_channels = input_dim // (grid_size * grid_size)
            elif isinstance(game, MinesweeperGame):
                input_dim = game.spatial_input_dim
                grid_size, n_channels = game.size, N_SPATIAL_CHANNELS
            else:
                input_dim = game.input_dim
                grid_size = getattr(game, "size", 8)
                n_channels = input_dim // (grid_size * grid_size)
        else:
            input_dim, grid_size, n_channels = game.input_dim, None, None

        output_dim = game.output_dim
        model_kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "size")}
        if model_name == "cnn":
            if grid_size is not None:
                model_kwargs.setdefault("grid_size", grid_size)
            if n_channels is not None:
                model_kwargs.setdefault("n_channels", n_channels)
        model = get_model(model_name, input_dim=input_dim, output_dim=output_dim, **model_kwargs)
        # NOTE: deliberately NO trainer.train() — model stays at random init.
        model.eval()
        logger.info(f"[baseline-untrained] {model_name} {game.name} seed={seed} — random init")

        xai_kwargs = {k: v for k, v in xai_cfg.items() if k != "name"}
        explainer = get_explainer(xai_name, model=model, game=game, **xai_kwargs)

        X_test = splits["test"]["X"]
        y_test = splits["test"]["y"]
        solutions_test = splits["test"].get("solutions")
        puzzles_test = splits["test"].get("puzzles")
        explain_split = splits.get("explain")
        X_explain = explain_split["X"] if explain_split is not None else None
        sols_explain = explain_split.get("solutions") if explain_split is not None else None

        metrics_cfg = cfg.get("metrics", {})
        report = compute_fidelity(
            model=model,
            game=game,
            explainer=explainer,
            X_test=X_test,
            y_test=y_test,
            solutions_test=solutions_test,
            puzzles_test=puzzles_test,
            model_name=f"{model_name}_untrained",
            xai_name=xai_name,
            config_hash=self._cfg_hash,
            seed=seed,
            metrics_cfg=metrics_cfg,
            X_explain=X_explain,
            solutions_explain=sols_explain,
        )

        report_dict = report.to_dict()
        if "canonical_stats" in report_dict:
            cs = report_dict.pop("canonical_stats")
            for k, v in cs.items():
                report_dict.setdefault(k, v)

        save_json(report_dict, self.results_dir / f"report_{game_name}_{model_name}_untrained_{xai_name}.json")
        return report

    def run_all_xai(self, xai_names: list[str] | None = None) -> list[FidelityReport]:
        """Run the same experiment with multiple XAI methods."""
        if xai_names is None:
            xai_names = ["rule_extraction", "lime", "lrp", "concept_probe", "symbolic_regression"]
        reports = []
        for xai in xai_names:
            cfg = {**self.config, "xai": {"name": xai}}
            runner = ExperimentRunner(cfg, str(self.results_dir))
            try:
                report = runner.run()
                reports.append(report)
            except Exception as e:
                logger.error(f"XAI {xai} failed: {e}")
        return reports
