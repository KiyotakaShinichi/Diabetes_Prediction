"""Deterministic, fingerprinted training subsets drawn from the frozen split.

Track K's development machine is CPU-only, and repeatedly training an
FT-Transformer on 40,125 rows is not a sensible use of it. The answer is not to
sample fewer rows each time something needs re-running - that would make every
comparison a comparison of two different datasets. The answer is ONE subset,
drawn once, fingerprinted, and reused by every model and every re-run.

Four properties, each load-bearing:

**Drawn only from train.** The validation partition still selects, the test
partition is still read once at the end, and neither is touched here. A subset
that borrowed rows from validation would quietly contaminate model selection;
one that borrowed from test would invalidate the entire study.

**Nested.** The 500-row subset is a subset of the 1,000, which is a subset of
the 2,500, which is a subset of the 5,000. This is what makes a sample-efficiency
curve mean anything: at each step the model sees strictly more data rather than
different data, so a change in score is attributable to the budget rather than to
which rows happened to be drawn.

**Stratified.** Each subset preserves the train partition's positive rate.
Sampling 500 rows without stratification from a 49.95% positive partition can
drift by several points by chance, and a drifted base rate would move
calibration and threshold selection for reasons that have nothing to do with
sample size.

**Fingerprinted and fail-closed.** A subset records the parent split it came
from and the exact rows it contains. If the dataset or the split moves, the
recorded subset no longer describes reality and every function here refuses to
proceed rather than silently training on different rows.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_core import training as core_training
from research.track_k import protocol, split


class SubsetIntegrityError(RuntimeError):
    """A recorded subset no longer describes the data underneath it."""


@dataclass(frozen=True, slots=True)
class SubsetFingerprint:
    """Identity of one training subset."""

    size: int
    requested_size: int
    seed: int
    positive_rate: float
    parent_partition: str
    parent_rows: int
    parent_indices_sha256: str
    indices_sha256: str
    dataset_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "requested_size": self.requested_size,
            "seed": self.seed,
            "positive_rate": self.positive_rate,
            "parent_partition": self.parent_partition,
            "parent_rows": self.parent_rows,
            "parent_indices_sha256": self.parent_indices_sha256,
            "indices_sha256": self.indices_sha256,
            "dataset_sha256": self.dataset_sha256,
        }


def _stratified_order(
    y_train: pd.Series, *, seed: int
) -> dict[int, np.ndarray]:
    """One shuffled ordering per class, drawn once.

    Every subset size reads a prefix of these same orderings, which is what
    makes the sizes nested: the first 250 positives of the 500-row subset are
    the first 250 positives of the 5,000-row subset too.
    """
    rng = np.random.default_rng(seed)
    orders = {}
    for label in (0, 1):
        members = np.asarray(y_train.index[y_train.to_numpy() == label])
        orders[label] = rng.permutation(members)
    return orders


def _class_quota(size: int, positive_rate: float) -> tuple[int, int]:
    """How many positives and negatives a subset of ``size`` rows should hold.

    Rounds the positive count and gives the remainder to the negatives, so the
    total is exactly ``size`` and the positive rate is as close to the parent's
    as an integer split allows.
    """
    positives = round(size * positive_rate)
    positives = max(1, min(size - 1, positives))
    return positives, size - positives


def build_nested_subsets(
    splits: core_training.TrainingSplits,
    *,
    sizes: tuple[int, ...] = protocol.SAMPLE_EFFICIENCY_SIZES,
    seed: int = protocol.SUBSET_SEED,
) -> dict[int, pd.Index]:
    """Nested, stratified training subsets, smallest first.

    Returns ``{size: index}``. Each index is a subset of the next larger one and
    of the train partition, and none of them touches validation or test.
    """
    ordered = sorted(sizes)
    if ordered[-1] > len(splits.X_train):
        raise ValueError(
            f"subset of {ordered[-1]} rows exceeds the {len(splits.X_train)}-row "
            "train partition"
        )

    y_train = splits.y_train
    orders = _stratified_order(y_train, seed=seed)
    positive_rate = float(y_train.mean())

    subsets: dict[int, pd.Index] = {}
    for size in ordered:
        n_positive, n_negative = _class_quota(size, positive_rate)
        if n_positive > len(orders[1]) or n_negative > len(orders[0]):
            raise ValueError(f"train partition cannot supply a stratified {size}-row subset")
        chosen = np.concatenate([orders[1][:n_positive], orders[0][:n_negative]])
        # Sort so membership, not draw order, defines the subset: two runs that
        # select the same rows produce the same fingerprint.
        subsets[size] = pd.Index(np.sort(chosen))
    return subsets


def fingerprint_subset(
    index: pd.Index,
    splits: core_training.TrainingSplits,
    *,
    requested_size: int,
    seed: int = protocol.SUBSET_SEED,
) -> SubsetFingerprint:
    """Identify a subset by its rows and by the partition it was drawn from."""
    target = splits.y_train.loc[index]
    return SubsetFingerprint(
        size=len(index),
        requested_size=int(requested_size),
        seed=int(seed),
        positive_rate=float(target.mean()),
        parent_partition="train",
        parent_rows=len(splits.X_train),
        parent_indices_sha256=split.hash_indices(splits.X_train.index),
        indices_sha256=split.hash_indices(index),
        dataset_sha256=hashlib.sha256(split.DATASET_PATH.read_bytes()).hexdigest(),
    )


def take(
    splits: core_training.TrainingSplits, index: pd.Index
) -> core_training.TrainingSplits:
    """The same split with its TRAIN partition narrowed to ``index``.

    Validation and test are passed through untouched and by reference: a
    constrained run selects on the same validation rows and is judged on the
    same test rows as the full run, which is exactly what makes the two
    comparable on everything except training budget.
    """
    missing = index.difference(splits.X_train.index)
    if len(missing):
        raise SubsetIntegrityError(
            f"{len(missing)} subset rows are not in the train partition"
        )
    return core_training.TrainingSplits(
        X_train=splits.X_train.loc[index],
        X_val=splits.X_val,
        X_test=splits.X_test,
        y_train=splits.y_train.loc[index],
        y_val=splits.y_val,
        y_test=splits.y_test,
        feature_names=splits.feature_names,
    )


def build_subset_manifest(
    splits: core_training.TrainingSplits,
    *,
    sizes: tuple[int, ...] = protocol.SAMPLE_EFFICIENCY_SIZES,
    seed: int = protocol.SUBSET_SEED,
) -> dict[str, Any]:
    """Everything needed to reconstruct and verify the subset ladder."""
    subsets = build_nested_subsets(splits, sizes=sizes, seed=seed)
    return {
        "seed": seed,
        "sizes": list(sorted(sizes)),
        "nested": True,
        "drawn_from": "train",
        "parent_positive_rate": float(splits.y_train.mean()),
        "subsets": {
            str(size): fingerprint_subset(
                index, splits, requested_size=size, seed=seed
            ).as_dict()
            for size, index in subsets.items()
        },
    }


def verify_subsets(
    splits: core_training.TrainingSplits, manifest: dict[str, Any]
) -> list[str]:
    """Problems with a recorded subset ladder. Empty means it still holds."""
    problems: list[str] = []
    recorded = manifest.get("subsets", {})
    seed = manifest.get("seed", protocol.SUBSET_SEED)
    sizes = tuple(manifest.get("sizes", protocol.SAMPLE_EFFICIENCY_SIZES))

    try:
        rebuilt = build_nested_subsets(splits, sizes=sizes, seed=seed)
    except ValueError as error:
        return [f"subsets cannot be rebuilt: {error}"]

    for size, index in rebuilt.items():
        was = recorded.get(str(size))
        if was is None:
            problems.append(f"subset {size}: not recorded")
            continue
        now = fingerprint_subset(index, splits, requested_size=size, seed=seed).as_dict()
        for field in ("indices_sha256", "parent_indices_sha256", "dataset_sha256", "size"):
            if was.get(field) != now.get(field):
                problems.append(
                    f"subset {size} {field}: recorded {was.get(field)!r}, now {now.get(field)!r}"
                )
    return problems


def load_verified_subsets(
    splits: core_training.TrainingSplits, manifest: dict[str, Any]
) -> dict[int, pd.Index]:
    """Rebuild the ladder, or refuse. Fail closed, never approximately right."""
    problems = verify_subsets(splits, manifest)
    if problems:
        raise SubsetIntegrityError("; ".join(problems))
    return build_nested_subsets(
        splits,
        sizes=tuple(manifest.get("sizes", protocol.SAMPLE_EFFICIENCY_SIZES)),
        seed=manifest.get("seed", protocol.SUBSET_SEED),
    )
