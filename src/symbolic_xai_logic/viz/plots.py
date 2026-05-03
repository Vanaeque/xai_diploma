"""Accuracy curves, fidelity heatmaps, attribution bar charts."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_training_curves(history: dict, save_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(
            np.linspace(0, len(history.get("train_loss", [1])) - 1, len(history["val_loss"])),
            history["val_loss"],
            label="Val Loss",
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    if "val_acc" in history:
        axes[1].plot(
            np.linspace(0, len(history.get("train_loss", [1])) - 1, len(history["val_acc"])),
            history["val_acc"],
            label="Val Accuracy",
            color="orange",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_fidelity_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    if df.empty:
        return

    metrics = ["nn_accuracy", "fidelity_to_nn", "agreement_gt"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return

    pivot_data = df.pivot_table(index="xai", columns="model", values=available[0], aggfunc="mean")
    if pivot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Fidelity Heatmap: {available[0]}")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_morf_lerf(reports: list[dict], save_path: str | None = None,
                    title: str | None = None) -> None:
    """One MoRF + LeRF curve panel per (model, xai), with shaded ±1 std band
    across seeds.  Reports must each have ``morf_curve`` and/or ``lerf_curve``
    sub-dicts in the form ``{"k_grid": [...], "mean_drop": [...]}``.

    Reviewers expect Samek et al. 2017-style curves rather than single-point
    comprehensiveness/sufficiency numbers.  See P1-12 in
    copilot_upgrade_instructions.md.
    """
    if not reports:
        return
    grouped: dict[tuple[str, str], dict[str, list[list[float]]]] = {}
    for r in reports:
        key = (r.get("model", "?"), r.get("xai", "?"))
        bucket = grouped.setdefault(key, {"k": None, "morf": [], "lerf": []})
        morf = r.get("morf_curve")
        lerf = r.get("lerf_curve")
        if morf and "mean_drop" in morf:
            bucket["morf"].append(list(morf["mean_drop"]))
            bucket["k"] = morf.get("k_grid", bucket["k"])
        if lerf and "mean_drop" in lerf:
            bucket["lerf"].append(list(lerf["mean_drop"]))
            bucket["k"] = lerf.get("k_grid", bucket["k"])

    grouped = {k: v for k, v in grouped.items() if (v["morf"] or v["lerf"]) and v["k"]}
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=max(2, len(grouped)))
    for (key, bucket), color in zip(grouped.items(), palette):
        ks = bucket["k"]
        if bucket["morf"]:
            arr = np.array(bucket["morf"])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            label = f"{key[0]}/{key[1]} MoRF"
            ax.plot(ks, mean, color=color, linestyle="-", label=label)
            ax.fill_between(ks, mean - std, mean + std, color=color, alpha=0.15)
        if bucket["lerf"]:
            arr = np.array(bucket["lerf"])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            label = f"{key[0]}/{key[1]} LeRF"
            ax.plot(ks, mean, color=color, linestyle="--", label=label)
            ax.fill_between(ks, mean - std, mean + std, color=color, alpha=0.10)

    ax.set_xlabel("k (features removed)")
    ax.set_ylabel("Mean |Δ output|")
    ax.set_title(title or "MoRF / LeRF removal curves")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_canonical_cell_heatmap(per_cell: dict[str, int], game_size: int,
                                save_path: str | None = None,
                                title: str | None = None) -> None:
    """Per-blank-cell distribution of canonical rule hits (Task P1-8).

    ``per_cell`` maps the human-readable target_cell labels emitted by
    ``RuleExtractor._dim_to_label`` (e.g. ``"cell (2,3) = digit 1"``) to the
    number of canonical rules that target that cell.  We aggregate over digit
    so the heatmap is one cell per (row, col).
    """
    import re
    grid = np.zeros((game_size, game_size), dtype=int)
    cell_rx = re.compile(r"cell\s*\((\d+)\s*,\s*(\d+)\)")
    for label, count in per_cell.items():
        m = cell_rx.search(label)
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        if 0 <= r < game_size and 0 <= c < game_size:
            grid[r, c] += count

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(grid, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "Canonical rules per cell"})
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title or "Canonical-rule recovery per blank cell")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_attribution_bar(attributions: np.ndarray, feature_names: list[str] | None = None,
                          top_k: int = 20, save_path: str | None = None) -> None:
    mean_abs = np.abs(attributions).mean(axis=0) if attributions.ndim == 2 else np.abs(attributions)
    if len(mean_abs) == 0:
        return

    k = min(top_k, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-k:][::-1]
    top_vals = mean_abs[top_idx]
    names = [feature_names[i] if feature_names and i < len(feature_names) else f"f_{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(k), top_vals)
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |Attribution|")
    ax.set_title(f"Top-{k} Feature Attributions")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
