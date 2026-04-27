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
