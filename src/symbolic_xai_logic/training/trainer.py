"""Training loop with checkpointing and seed control."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import get_logger
from ..utils.io import save_checkpoint, load_checkpoint, get_git_sha, config_hash
from ..utils.seeding import set_global_seed
from .losses import get_loss
from .metrics import accuracy, cell_accuracy, blank_cell_accuracy

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        game: Any,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        eval_interval: int = 5,
        checkpoint_dir: str = "results/checkpoints",
        config: dict | None = None,
        seed: int = 42,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
    ):
        self.model = model.to(device)
        self.game = game
        self.device = device
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or {}
        self.seed = seed
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta

        self.criterion = get_loss(game.name)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_acc": [], "blank_cell_acc": [],
        }
        self._is_sudoku = "sudoku" in game.name

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list]:
        set_global_seed(self.seed)
        best_val_loss = float("inf")
        no_improve_rounds = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["train_loss"].append(avg_loss)

            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                val_metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                if val_metrics.get("blank_cell_accuracy") is not None:
                    self.history["blank_cell_acc"].append(val_metrics["blank_cell_accuracy"])
                logger.info(
                    f"Epoch {epoch}: train_loss={avg_loss:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )
                improved = (best_val_loss - val_metrics["loss"]) > self.early_stop_min_delta
                if improved:
                    best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, val_metrics)
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if self.early_stop_patience > 0 and no_improve_rounds >= self.early_stop_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch}: "
                            f"no val_loss improvement for {self.early_stop_patience} eval rounds "
                            f"(best={best_val_loss:.4f})"
                        )
                        break

        return self.history

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_X, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item()
                all_X.append(X_batch.cpu())
                all_preds.append(pred.cpu())
                all_targets.append(y_batch.cpu())

        all_X = torch.cat(all_X)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics: dict[str, float] = {"loss": total_loss / max(len(loader), 1)}

        if self._is_sudoku:
            n = self.game.size
            metrics["accuracy"] = cell_accuracy(all_preds, all_targets, n_classes=n)
            
            # Derive given_mask based on encoding format
            # For one-hot: X shape (B, n²×n), reshape to (B, n², n) and check if any digit is 1
            # For spatial: X shape (B, (n+1)×n²), reshape to (B, n+1, n, n), check if channel 0 is < 0.5
            if all_X.shape[1] == n * n * n:
                # One-hot encoding: (B, n²×n)
                X_r = all_X.view(all_X.shape[0], -1, n)
                given_mask = X_r.max(dim=-1).values > 0.9
            else:
                # Spatial encoding: (B, (n+1)×n²)
                X_spatial = all_X.view(all_X.shape[0], n+1, n, n)
                # Channel 0 is "unknown"; given cells have channel 0 < 0.5
                given_mask = (X_spatial[:, 0, :, :] < 0.5).view(all_X.shape[0], -1)
            
            metrics["blank_cell_accuracy"] = blank_cell_accuracy(
                all_preds, all_targets, given_mask, n_classes=n
            )
        else:
            metrics["accuracy"] = accuracy(all_preds, all_targets)

        return metrics

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg_hash = config_hash(self.config)
        path = self.checkpoint_dir / f"{self.game.name}_best.pt"
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
                "config_hash": cfg_hash,
                "git_sha": get_git_sha(),
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def load_best(self, checkpoint_dir: str | None = None) -> dict:
        ckpt_dir = Path(checkpoint_dir or self.checkpoint_dir)
        path = ckpt_dir / f"{self.game.name}_best.pt"
        ckpt = load_checkpoint(path)
        self.model.load_state_dict(ckpt["model_state"])
        return ckpt
