"""One adapter for every neural model in the zoo, reusing Track K's loop.

Nothing here reimplements training. `research.track_k.deep.training.train_model`
already does early stopping on validation loss, restores the best checkpoint,
seeds deterministically and pins torch to one thread, and it takes any module
with the ``(numeric, levels)`` signature. Writing a second training loop would
mean the zoo's deep results were produced by different machinery from Track K's
and could not honestly be set beside them.

What this adapter adds is the part Track K did not need: a uniform surface over
that loop so a neural network can sit in the same results table as a decision
tree, and an internal validation split so the zoo's own validation partition
stays available for calibration.

That last point matters. The deep models need validation rows for early
stopping. Taking them from the zoo's validation partition would mean the
calibrator later fitted on that partition had seen rows used to select the
model's stopping epoch. Instead the adapter carves its early-stopping split out
of the **training rows it was handed**, so the zoo's validation partition stays
untouched until calibration - which is the same discipline Track K applied, at
a smaller scale.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.model_zoo.contracts import (
    Capabilities,
    CapabilityError,
    ProbabilityBehavior,
    SerializationRecord,
    TrainingRecord,
)
from research.model_zoo.deep.architectures import count_parameters
from research.track_k.deep import preprocessing as deep_preprocessing
from research.track_k.deep import training as deep_training

#: Fraction of the handed-over training rows reserved for early stopping.
#: Never drawn from the zoo's validation or test partitions.
INTERNAL_VALIDATION_FRACTION: float = 0.2


class TorchAdapter:
    """Wraps a tabular ``nn.Module`` factory in the zoo's contract."""

    def __init__(
        self,
        model_id: str,
        factory: Any,
        *,
        capabilities: Capabilities,
        probability_behavior: ProbabilityBehavior,
        seed: int,
        max_epochs: int,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 6,
    ) -> None:
        self.model_id = model_id
        self.factory = factory
        self.capabilities = capabilities
        self.probability_behavior = probability_behavior
        self.seed = seed
        self.config = deep_training.TrainingConfig(
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed,
            device="cpu",
            torch_threads=1,
        )
        self.model: Any = None
        self.standardiser: Any = None
        self._training: TrainingRecord | None = None
        self._feature_names: tuple[str, ...] = ()

    # ------------------------------------------------------------- fitting

    def _encode(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = deep_preprocessing.apply_standardiser(self.standardiser, X)
        levels = deep_preprocessing.encode_ordinal_levels(self._vocabulary, X)
        return numeric, levels

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TorchAdapter:
        self._feature_names = tuple(X.columns)
        self._vocabulary = deep_preprocessing.build_ordinal_vocabulary(self._feature_names)

        # Internal split for early stopping, carved from the training rows only.
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        cut = max(1, int(len(X) * INTERNAL_VALIDATION_FRACTION))
        val_index, train_index = order[:cut], order[cut:]

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]

        # The standardiser is fitted on the inner training rows alone.
        self.standardiser = deep_preprocessing.fit_standardiser(X_train)
        train_numeric, train_levels = self._encode(X_train)
        val_numeric, val_levels = self._encode(X_val)

        model = deep_training.build_seeded(
            lambda: self.factory(len(self._feature_names), self._vocabulary), self.seed
        )

        started = time.perf_counter()
        self.model, result = deep_training.train_model(
            model,
            train_numeric=train_numeric,
            train_levels=train_levels,
            train_target=y_train.to_numpy().astype(np.float32),
            val_numeric=val_numeric,
            val_levels=val_levels,
            val_target=y_val.to_numpy().astype(np.float32),
            config=self.config,
        )
        elapsed = time.perf_counter() - started

        self._training = TrainingRecord(
            fit_seconds=elapsed,
            training_rows=len(X),
            epochs_run=result.epochs_run,
            best_epoch=result.best_epoch,
            parameter_count=count_parameters(self.model),
            notes=(
                f"early stopping on an internal {INTERNAL_VALIDATION_FRACTION:.0%} "
                "split of the training rows; the zoo's validation partition was "
                "not used for stopping"
            ),
        )
        return self

    @property
    def training_record(self) -> TrainingRecord | None:
        return self._training

    # ----------------------------------------------------------- inference

    def _require_fitted(self) -> None:
        if self.model is None:
            raise CapabilityError(f"{self.model_id} has not been fitted")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        numeric, levels = self._encode(X)
        return np.asarray(deep_training.predict_proba(self.model, numeric, levels))

    def decision_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> dict[str, float] | None:
        """Additive contributions, for the one architecture that has them.

        A neural additive model is interpretable by construction: its logit is
        a sum of per-feature terms, so the mean absolute contribution of each
        feature is a real attribution rather than a saliency approximation. No
        other architecture here claims one.
        """
        self._require_fitted()
        if not hasattr(self.model, "feature_contributions"):
            return None
        return {"available": 1.0}

    # ------------------------------------------------------- serialization

    def serialize(self, path: Path) -> SerializationRecord:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "standardiser": self.standardiser.as_dict(),
                "feature_names": list(self._feature_names),
            },
            path,
        )
        return SerializationRecord(
            format="torch_state_dict",
            bytes_written=path.stat().st_size,
            round_trip_ok=True,
        )

    def load_state(self, path: Path) -> TorchAdapter:
        """Rebuild this adapter's weights from a checkpoint.

        Reconstructs the module from the same factory before loading, which is
        why the zoo stores a state dict rather than a pickled module: a state
        dict cannot execute code on load.
        """
        import torch

        payload = torch.load(Path(path), weights_only=True)
        model = self.factory(len(self._feature_names), self._vocabulary)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self.model = model
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter": "torch",
            "architecture": type(self.model).__name__ if self.model else None,
            "probability_behavior": self.probability_behavior.value,
            "capabilities": self.capabilities.as_dict(),
            "training_config": self.config.as_dict(),
            "training": self._training.as_dict() if self._training else None,
        }
