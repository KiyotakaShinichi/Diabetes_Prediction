"""Model-agnostic training primitives shared by the maintained pipelines.

Only genuinely reusable, variant-independent stages live here: dataset loading
and schema validation, the train/validation/test split, threshold-based
evaluation and atomic JSON writing.

Model construction, hyperparameter search spaces, calibration wiring, SHAP
explainers and drift baselines deliberately stay in their pipeline modules. The
two variants differ in real ways there, and forcing them through a shared
abstraction would obscure the differences rather than remove them.

Nothing here loads a dataset, fits a model or writes a file at import time.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_core.evaluation import evaluate_predictions

#: The only target encoding the served pipelines support.
EXPECTED_TARGET_VALUES = frozenset({0, 1})


class DatasetValidationError(ValueError):
    """The training dataset does not satisfy the served feature contract."""


class ProbabilityEstimator(Protocol):
    """Minimal estimator surface required by the shared scoring helpers."""

    def predict_proba(self, X: pd.DataFrame) -> npt.ArrayLike: ...


def validate_training_dataset(
    frame: pd.DataFrame, feature_names: list[str], target_column: str
) -> None:
    """Check a dataset against the canonical served contract.

    Closes the gap left open by the feature contract itself: the contract
    described what the models expect, but nothing checked the data actually
    provided it until training was already under way.

    Every check corresponds to an assumption the existing pipelines already
    make. No new data-quality thresholds are invented here.

    Raises:
        DatasetValidationError: naming the first problem found.
    """
    if frame.empty:
        raise DatasetValidationError("dataset is empty")

    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise DatasetValidationError(f"duplicated columns in dataset: {duplicated}")

    if target_column not in frame.columns:
        raise DatasetValidationError(f"missing target column {target_column!r}")

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise DatasetValidationError(f"missing feature columns: {missing}")

    for name in feature_names:
        column = frame[name]
        if column.isna().all():
            raise DatasetValidationError(f"feature column {name!r} is entirely null")
        if not pd.api.types.is_numeric_dtype(column):
            coerced = pd.to_numeric(column, errors="coerce")
            if coerced.isna().all():
                raise DatasetValidationError(
                    f"feature column {name!r} is not numeric and cannot be coerced"
                )

    target = frame[target_column]
    if target.isna().all():
        raise DatasetValidationError(f"target column {target_column!r} is entirely null")

    observed = set(pd.unique(target.dropna()))
    unexpected = {value for value in observed if int(value) not in EXPECTED_TARGET_VALUES}
    if unexpected:
        raise DatasetValidationError(
            f"target {target_column!r} must be binary 0/1; found {sorted(unexpected)}"
        )
    if len({int(value) for value in observed}) < 2:
        raise DatasetValidationError(
            f"target {target_column!r} holds a single class; both are required"
        )


def load_training_dataset(
    path: str | Path, feature_names: list[str], target_column: str
) -> pd.DataFrame:
    """Read a CSV and validate it against the contract before returning it."""
    path = Path(path)
    if not path.is_file():
        raise DatasetValidationError(f"training dataset not found at {path}")
    frame = pd.read_csv(path)
    validate_training_dataset(frame, feature_names, target_column)
    return frame


def select_features(
    frame: pd.DataFrame, feature_names: list[str], target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated frame into X (canonical column order) and y."""
    return frame[list(feature_names)].copy(), frame[target_column].copy()


@dataclass(frozen=True, slots=True)
class TrainingSplits:
    """The three-way split both pipelines use.

    ``val`` exists so the decision threshold is chosen without touching the test
    set - the pipelines already worked this way, and the structure makes the
    separation explicit rather than implicit in six loose locals.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: tuple[str, ...]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.X_train), "val": len(self.X_val), "test": len(self.X_test)}


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,
) -> TrainingSplits:
    """Stratified train/validation/test split.

    Semantics preserved exactly from both pipelines: first hold out ``test_size``
    of everything, then take ``val_size`` of the remainder as validation.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if stratify else None, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size,
        stratify=y_train_full if stratify else None,
        random_state=random_state,
    )
    return TrainingSplits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=tuple(X.columns),
    )


def positive_class_proba(model: ProbabilityEstimator, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class, as a plain 1-D array."""
    return np.asarray(model.predict_proba(X))[:, 1]


def evaluate_at_threshold(
    y_true: npt.ArrayLike, y_proba: np.ndarray, threshold: float
) -> tuple[np.ndarray, dict]:
    """Apply a decision threshold and evaluate. Returns (predictions, metrics)."""
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, evaluate_predictions(np.asarray(y_true), y_pred, y_proba)


def write_json_atomic(payload: dict, path: str | Path) -> Path:
    """Write JSON via a temp file and os.replace, with LF endings.

    Same discipline as the provenance manifest: a partially written metrics file
    would look valid while describing nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path
