"""Tests for Trainer: early stopping and blank_cell_accuracy integration."""
from __future__ import annotations
import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class _FakeGame:
    name = "fake"


def _make_trainer(patience: int, epochs: int = 20, eval_interval: int = 1):
    from symbolic_xai_logic.training.trainer import Trainer

    model = nn.Linear(4, 1)
    trainer = Trainer(
        model=model,
        game=_FakeGame(),
        epochs=epochs,
        eval_interval=eval_interval,
        early_stop_patience=patience,
        checkpoint_dir="/tmp/test_ckpts_es",
        seed=0,
    )
    # Suppress disk writes during tests
    trainer._save_checkpoint = lambda epoch, metrics: None
    return trainer


def _dummy_loader(n: int = 10) -> DataLoader:
    X = torch.zeros(n, 4)
    y = torch.zeros(n, 1)
    return DataLoader(TensorDataset(X, y), batch_size=n)


class TestEarlyStopping:
    def test_no_early_stop_runs_all_epochs(self):
        trainer = _make_trainer(patience=0, epochs=6, eval_interval=1)
        call_count = [0]

        def _eval(loader):
            call_count[0] += 1
            return {"loss": 1.0 - call_count[0] * 0.01, "accuracy": 0.5}

        trainer.evaluate = _eval
        loader = _dummy_loader()
        history = trainer.train(loader, loader)
        # All 6 epochs ran
        assert len(history["val_loss"]) == 6

    def test_early_stopping_triggers(self):
        """Constant val_loss → patience=2 stops after 3 eval rounds (1 improvement + 2 no-improve)."""
        trainer = _make_trainer(patience=2, epochs=20, eval_interval=1)

        def _constant_eval(loader):
            return {"loss": 1.0, "accuracy": 0.5}

        trainer.evaluate = _constant_eval
        loader = _dummy_loader()
        history = trainer.train(loader, loader)

        # Round 1: loss=1.0 improves from inf → saves, no_improve=0
        # Round 2: loss=1.0 → no_improve=1
        # Round 3: loss=1.0 → no_improve=2 >= patience=2 → stop
        assert len(history["val_loss"]) == 3
        assert len(history["val_loss"]) < 20

    def test_early_stopping_resets_on_improvement(self):
        """Counter resets each time loss improves; stops only after consecutive non-improvements."""
        trainer = _make_trainer(patience=2, epochs=20, eval_interval=1)
        losses = [1.0, 0.9, 0.9, 0.9, 0.9]
        idx = [0]

        def _varying_eval(loader):
            loss = losses[min(idx[0], len(losses) - 1)]
            idx[0] += 1
            return {"loss": loss, "accuracy": 0.5}

        trainer.evaluate = _varying_eval
        loader = _dummy_loader()
        history = trainer.train(loader, loader)

        # Rounds: inf→1.0 (improve), 1.0→0.9 (improve, reset), 0.9→0.9 (no, 1), 0.9→0.9 (no, 2 → stop)
        assert len(history["val_loss"]) == 4

    def test_history_matches_actual_epochs(self):
        """history lengths should match the number of epochs actually run."""
        trainer = _make_trainer(patience=2, epochs=20, eval_interval=2)

        def _constant_eval(loader):
            return {"loss": 0.5, "accuracy": 0.5}

        trainer.evaluate = _constant_eval
        loader = _dummy_loader()
        history = trainer.train(loader, loader)

        # eval_interval=2 → eval at epochs 2,4,6,...; patience=2 stops at round 3 = epoch 6
        # train_loss gets one entry per epoch; val_loss one entry per eval round
        assert len(history["train_loss"]) == len(history["val_loss"]) * 2
