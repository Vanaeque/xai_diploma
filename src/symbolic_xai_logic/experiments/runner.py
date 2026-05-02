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
        )

        # Retrieve NL breakdown for the JSON report (canonical_match_rate already in report).
        # RuleExtractor caches the fitted tree; to_sympy() reuses it without refitting.
        nl_result: dict = {}
        if hasattr(explainer, "to_sympy"):
            formulas = explainer.to_sympy()
            if formulas:
                try:
                    from ..viz.templates import render_clauses_as_nl
                    nl_result = render_clauses_as_nl(formulas, game)
                except Exception as exc:
                    logger.warning(f"NL template breakdown failed: {exc}")

        report_dict = report.to_dict()
        if nl_result:
            report_dict["matched"] = nl_result.get("matched", 0)
            report_dict["total"] = nl_result.get("total", 0)
            report_dict["by_template"] = nl_result.get("by_template", {})

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
