"""Ablation experiments: vary model size, data size, XAI method."""
from __future__ import annotations
from typing import Any
from .runner import ExperimentRunner
from ..utils.logging import get_logger

logger = get_logger(__name__)


def run_data_size_ablation(base_config: dict, sizes: list[int], results_dir: str = "results") -> list[dict]:
    """Ablation: vary training set size."""
    reports = []
    for n in sizes:
        cfg = {**base_config, "data": {**base_config.get("data", {}), "n_train": n}}
        runner = ExperimentRunner(cfg, results_dir)
        try:
            r = runner.run()
            reports.append({"n_train": n, **r.to_dict()})
        except Exception as e:
            logger.error(f"Ablation n_train={n} failed: {e}")
    return reports


def run_model_size_ablation(base_config: dict, sizes: list[str], results_dir: str = "results") -> list[dict]:
    """Ablation: vary model size (small/medium/large)."""
    size_configs = {
        "small": {"hidden_dims": [64, 64]},
        "medium": {"hidden_dims": [256, 256, 128]},
        "large": {"hidden_dims": [512, 512, 256, 128]},
    }
    reports = []
    for size in sizes:
        extra = size_configs.get(size, {})
        cfg = {**base_config, "model": {**base_config.get("model", {}), **extra, "size": size}}
        runner = ExperimentRunner(cfg, results_dir)
        try:
            r = runner.run()
            reports.append({"model_size": size, **r.to_dict()})
        except Exception as e:
            logger.error(f"Ablation model_size={size} failed: {e}")
    return reports
