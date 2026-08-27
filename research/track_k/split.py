"""The one split every Track K model sees.

A benchmark where two models are scored on different rows measures the split,
not the models. This module derives a single deterministic three-way split from
the frozen protocol, fingerprints it, and refuses to hand back a split that does
not match a recorded fingerprint.

It deliberately reuses ml_core.training.split_training_data rather than
re-deriving one: the production pipelines already split this way, so Track K
numbers sit on the same footing as the figures the repository publishes.

No row data is ever written to disk. The manifest records counts, class
balance and hashes of the index sets - enough to prove two runs used the same
rows, and not enough to leak the dataset into a JSON file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml_core import provenance, training
from research.track_k import protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = PROJECT_ROOT / protocol.DATASET_FILENAME


class SplitIntegrityError(RuntimeError):
    """The dataset or split does not match what a recorded run used.

    Raised rather than warned: a benchmark that silently continues on different
    rows produces numbers that look comparable and are not.
    """


@dataclass(frozen=True, slots=True)
class SplitFingerprint:
    """Identity of one split, independent of the data it indexes."""

    dataset_sha256: str
    rows: int
    seed: int
    test_size: float
    validation_size_of_remainder: float
    stratified: bool
    train_indices_sha256: str
    val_indices_sha256: str
    test_indices_sha256: str
    combined_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "rows": self.rows,
            "seed": self.seed,
            "test_size": self.test_size,
            "validation_size_of_remainder": self.validation_size_of_remainder,
            "stratified": self.stratified,
            "train_indices_sha256": self.train_indices_sha256,
            "val_indices_sha256": self.val_indices_sha256,
            "test_indices_sha256": self.test_indices_sha256,
            "combined_sha256": self.combined_sha256,
        }


def hash_indices(index: pd.Index) -> str:
    """Order-sensitive digest of a row-index set.

    Order matters: two splits containing the same rows in a different order
    would produce identical models but different per-row prediction files, and
    the manifest should be able to tell them apart.
    """
    payload = ",".join(str(int(value)) for value in index)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read the benchmark dataset and validate it against the feature contract.

    Validation is ml_core's, not a second implementation: a Track K run and a
    production training run reject the same malformed dataset for the same
    reasons.
    """
    resolved = Path(path) if path is not None else DATASET_PATH
    return training.load_training_dataset(
        resolved, list(protocol.FEATURE_NAMES), protocol.TARGET_COLUMN
    )


def build_split(frame: pd.DataFrame) -> training.TrainingSplits:
    """The frozen three-way split, identical for every model family."""
    features, target = training.select_features(
        frame, list(protocol.FEATURE_NAMES), protocol.TARGET_COLUMN
    )
    return training.split_training_data(
        features,
        target,
        test_size=protocol.TEST_SIZE,
        val_size=protocol.VALIDATION_SIZE_OF_REMAINDER,
        random_state=protocol.SPLIT_SEED,
        stratify=protocol.STRATIFY,
    )


def fingerprint_split(
    splits: training.TrainingSplits, *, dataset_path: Path | None = None
) -> SplitFingerprint:
    """Identify a split by its dataset and its exact row membership."""
    resolved = Path(dataset_path) if dataset_path is not None else DATASET_PATH
    train_hash = hash_indices(splits.X_train.index)
    val_hash = hash_indices(splits.X_val.index)
    test_hash = hash_indices(splits.X_test.index)
    combined = provenance.sha256_canonical_json(
        {"train": train_hash, "val": val_hash, "test": test_hash}
    )
    return SplitFingerprint(
        dataset_sha256=provenance.sha256_file(resolved),
        rows=int(len(splits.X_train) + len(splits.X_val) + len(splits.X_test)),
        seed=protocol.SPLIT_SEED,
        test_size=protocol.TEST_SIZE,
        validation_size_of_remainder=protocol.VALIDATION_SIZE_OF_REMAINDER,
        stratified=protocol.STRATIFY,
        train_indices_sha256=train_hash,
        val_indices_sha256=val_hash,
        test_indices_sha256=test_hash,
        combined_sha256=combined,
    )


def class_balance(target: pd.Series) -> dict[str, float | int]:
    """Counts and positive rate, recorded so a manifest states the base rate."""
    counts = target.value_counts().sort_index()
    return {
        "negative": int(counts.get(0, 0)),
        "positive": int(counts.get(1, 0)),
        "positive_rate": float(target.mean()),
    }


def build_split_manifest(
    splits: training.TrainingSplits, *, dataset_path: Path | None = None
) -> dict[str, Any]:
    """Everything needed to prove another run used this split, and nothing more."""
    resolved = Path(dataset_path) if dataset_path is not None else DATASET_PATH
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "dataset": provenance.fingerprint_dataset(
            resolved, PROJECT_ROOT, protocol.TARGET_COLUMN
        ),
        "features": provenance.fingerprint_features(
            list(protocol.FEATURE_NAMES), protocol.TARGET_COLUMN
        ),
        "split": fingerprint_split(splits, dataset_path=resolved).as_dict(),
        "sizes": splits.sizes,
        "class_balance": {
            "train": class_balance(splits.y_train),
            "val": class_balance(splits.y_val),
            "test": class_balance(splits.y_test),
        },
    }


def verify_split(
    splits: training.TrainingSplits,
    manifest: dict[str, Any],
    *,
    dataset_path: Path | None = None,
) -> list[str]:
    """Differences between a split and a recorded one. Empty means identical."""
    recorded = manifest.get("split", {})
    actual = fingerprint_split(splits, dataset_path=dataset_path).as_dict()

    problems: list[str] = []
    for key, expected in recorded.items():
        if actual.get(key) != expected:
            problems.append(f"{key}: expected {expected!r}, got {actual.get(key)!r}")
    if manifest.get("protocol_version") != protocol.PROTOCOL_VERSION:
        problems.append(
            f"protocol_version: manifest {manifest.get('protocol_version')!r} "
            f"!= current {protocol.PROTOCOL_VERSION!r}"
        )
    return problems


def load_frozen_split(
    manifest: dict[str, Any] | None = None, *, dataset_path: Path | None = None
) -> training.TrainingSplits:
    """The benchmark split, refusing to proceed if it has drifted.

    Passing a manifest turns this into a fail-closed load: if the dataset bytes
    or the derived row membership differ from the recorded run, it raises rather
    than returning rows that would silently invalidate every comparison.
    """
    frame = load_dataset(dataset_path)
    splits = build_split(frame)
    if manifest is not None:
        problems = verify_split(splits, manifest, dataset_path=dataset_path)
        if problems:
            raise SplitIntegrityError(
                "the frozen split no longer reproduces: " + "; ".join(problems)
            )
    return splits


def write_split_manifest(
    splits: training.TrainingSplits, path: Path, *, dataset_path: Path | None = None
) -> Path:
    """Persist the split manifest atomically."""
    return training.write_json_atomic(
        build_split_manifest(splits, dataset_path=dataset_path), path
    )
