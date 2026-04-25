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
from .metrics import accuracy, cell_accuracy

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
    ):
        self.model = model.to(device)
        self.game = game
        self.device = device
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or {}
        self.seed = seed

        self.criterion = get_loss(game.name)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self._is_sudoku = "sudoku" in game.name

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list]:
        set_global_seed(self.seed)
        best_val_loss = float("inf")

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
                logger.info(
                    f"Epoch {epoch}: train_loss={avg_loss:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, val_metrics)

        return self.history

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item()
                all_preds.append(pred.cpu())
                all_targets.append(y_batch.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if self._is_sudoku:
            acc = cell_accuracy(all_preds, all_targets, n_classes=self.game.size)
        else:
            acc = accuracy(all_preds, all_targets)

        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": acc,
        }

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
