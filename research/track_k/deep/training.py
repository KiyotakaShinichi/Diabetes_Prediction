"""A small, deterministic training loop for the Track K challengers.

Scoped to this project on purpose. It is not a framework: no plugin system, no
callback registry, no abstract trainer hierarchy. It trains a binary classifier
on a tabular dataset that fits comfortably in memory, on CPU, reproducibly, and
stops when validation stops improving.

Determinism is the property that matters most here. A benchmark whose numbers
move between runs cannot support a promotion decision, so every source of
randomness is seeded and recorded: Python, NumPy, torch, and the DataLoader's
shuffling generator.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularClassifier(Protocol):
    """What the training loop needs from a model. Both challengers satisfy it."""

    def __call__(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor: ...

    def parameters(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Every knob that affects a training run, recorded into provenance."""

    max_epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    #: Stop after this many epochs without validation improvement. The best
    #: checkpoint is restored, so a late overfit cannot leak into the result.
    patience: int = 8
    #: Minimum improvement that counts as progress, so noise does not reset it.
    min_delta: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    #: Single-threaded by default: reproducible across machines and plenty for
    #: models this size. Multi-threaded BLAS can reorder float accumulation.
    torch_threads: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "seed": self.seed,
            "device": self.device,
            "torch_threads": self.torch_threads,
        }


@dataclass
class EpochRecord:
    """One epoch's learning-curve point."""

    epoch: int
    train_loss: float
    val_loss: float
    val_roc_auc: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_roc_auc": self.val_roc_auc,
        }


@dataclass
class TrainingResult:
    """Outcome of one training run, including the curve that produced it."""

    best_epoch: int
    best_val_loss: float
    best_val_roc_auc: float
    epochs_run: int
    early_stopped: bool
    duration_seconds: float
    history: list[EpochRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_roc_auc": self.best_val_roc_auc,
            "epochs_run": self.epochs_run,
            "early_stopped": self.early_stopped,
            "duration_seconds": self.duration_seconds,
            "history": [record.as_dict() for record in self.history],
        }


def seed_everything(seed: int) -> None:
    """Seed every generator this project's training touches.

    torch.use_deterministic_algorithms is deliberately NOT enabled: it raises on
    operations that have no deterministic CPU kernel, and the models here do not
    need it. Seeding plus single-threaded CPU execution already reproduces a run
    exactly, which the tests verify rather than assume.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_seeded(factory: Any, seed: int) -> Any:
    """Construct a model with reproducible initial weights.

    Seeding inside train_model is too late: a module randomises its parameters
    in __init__, so two runs that seed only at training time start from
    different weights and diverge for a reason the seed never controlled. Every
    Track K model is therefore built through this helper, and a test trains the
    same architecture twice to prove the weights match.
    """
    seed_everything(seed)
    return factory()


def make_loader(
    numeric: np.ndarray,
    levels: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """A DataLoader whose shuffling is seeded, so epoch order is reproducible."""
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(numeric, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(levels, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
        num_workers=0,
    )


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    numeric: np.ndarray,
    levels: np.ndarray,
    *,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Positive-class probabilities for a whole partition."""
    model.eval()
    outputs: list[np.ndarray] = []
    numeric_tensor = torch.from_numpy(np.ascontiguousarray(numeric, dtype=np.float32))
    level_tensor = torch.from_numpy(np.ascontiguousarray(levels, dtype=np.int64))

    for start in range(0, len(numeric_tensor), batch_size):
        stop = start + batch_size
        logits = model(
            numeric_tensor[start:stop].to(device), level_tensor[start:stop].to(device)
        )
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
    if not outputs:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(outputs).astype(np.float64)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the rank-sum identity.

    Ranking is done by scipy.stats.rankdata, whose "average" method resolves
    ties in C. An earlier version resolved them in a Python while-loop, which
    was correct but made the 2,000-replicate paired bootstrap take longer than
    ten minutes: 24,000 metric evaluations over 13,376 rows is hundreds of
    millions of interpreter steps. The identity below is the same computation
    at C speed, and the test comparing it against sklearn still passes.

    A single-class partition has no defined ROC-AUC. Degenerate resamples happen
    on tiny smoke datasets, so 0.5 is returned rather than raising and aborting
    a run.
    """
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = rankdata(y_score, method="average")
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train_model(
    model: nn.Module,
    *,
    train_numeric: np.ndarray,
    train_levels: np.ndarray,
    train_target: np.ndarray,
    val_numeric: np.ndarray,
    val_levels: np.ndarray,
    val_target: np.ndarray,
    config: TrainingConfig,
) -> tuple[nn.Module, TrainingResult]:
    """Train with early stopping on validation loss; restore the best weights.

    Selection uses validation only. The test partition is never touched here,
    which is what makes the final benchmark a single honest evaluation rather
    than the last of many peeks.
    """
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    loader = make_loader(
        train_numeric, train_levels, train_target,
        batch_size=config.batch_size, shuffle=True, seed=config.seed,
    )

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val_loss = float("inf")
    best_val_auc = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[EpochRecord] = []
    started = time.perf_counter()

    val_numeric_tensor = torch.from_numpy(
        np.ascontiguousarray(val_numeric, dtype=np.float32)
    ).to(device)
    val_level_tensor = torch.from_numpy(
        np.ascontiguousarray(val_levels, dtype=np.int64)
    ).to(device)
    val_target_tensor = torch.from_numpy(
        np.ascontiguousarray(val_target, dtype=np.float32)
    ).to(device)

    epoch = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for numeric_batch, level_batch, target_batch in loader:
            numeric_batch = numeric_batch.to(device)
            level_batch = level_batch.to(device)
            target_batch = target_batch.to(device)

            optimiser.zero_grad(set_to_none=True)
            logits = model(numeric_batch, level_batch).reshape(-1)
            loss = criterion(logits, target_batch)
            loss.backward()
            optimiser.step()

            running += float(loss.item()) * len(target_batch)
            seen += len(target_batch)

        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_numeric_tensor, val_level_tensor).reshape(-1)
            val_loss = float(criterion(val_logits, val_target_tensor).item())
            val_proba = torch.sigmoid(val_logits).detach().cpu().numpy()
        val_auc = roc_auc(val_target, val_proba)

        history.append(EpochRecord(epoch, train_loss, val_loss, val_auc))

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, TrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_roc_auc=best_val_auc,
        epochs_run=epoch,
        early_stopped=epoch < config.max_epochs,
        duration_seconds=time.perf_counter() - started,
        history=history,
    )
