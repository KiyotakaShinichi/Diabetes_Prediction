"""Tiny deterministic datasets with known answers, for testing the harness.

Thirty models sharing one harness is thirty chances for a harness bug to look
like a research finding. These four datasets exist so the harness can be tested
against problems whose answers are known in advance, at a size where every model
in the zoo fits in under a second.

They deliberately do **not** assert a ranking between algorithms. "Random forest
should beat logistic regression on XOR" is the kind of assertion that is true
until a seed changes, and a test that fails for a legitimate reason teaches
people to ignore it. What they assert instead is directional and robust: a model
must learn a problem that is trivially learnable, and must *fail* to learn one
that contains nothing.

`NOISE_ONLY` is the important one. Labels there are independent of the features,
so no model can do better than chance on held-out rows. Any model that appears
to is not clever - it is being scored on rows it was fitted on, and that is the
single most likely bug in a shared benchmark harness.

Every dataset uses the real feature contract's column names and ranges, so the
same preprocessing and vocabulary code runs on them as on the real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from ml_core import feature_contract

FEATURES: tuple[str, ...] = feature_contract.FEATURE_NAMES


class SyntheticProblem(StrEnum):
    LINEARLY_SEPARABLE = "linearly_separable"
    NONLINEAR_XOR = "nonlinear_xor"
    CLASS_IMBALANCE = "class_imbalance"
    NOISE_ONLY = "noise_only"


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """A contract-valid frame, its labels, and what should happen on it."""

    problem: SyntheticProblem
    X: pd.DataFrame
    y: pd.Series
    #: The held-out ROC-AUC a working model should be able to exceed. None
    #: where no model should be able to learn anything.
    learnable_floor: float | None
    description: str


def _contract_frame(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random but contract-valid feature values, in canonical column order."""
    columns = {}
    for spec in feature_contract.FEATURE_SPECS:
        if spec.kind == "continuous":
            columns[spec.name] = rng.uniform(spec.minimum, spec.maximum, rows)
        else:
            columns[spec.name] = rng.integers(
                int(spec.minimum), int(spec.maximum) + 1, rows
            ).astype(float)
    return pd.DataFrame(columns)[list(FEATURES)]


def make(problem: SyntheticProblem, *, rows: int = 400, seed: int = 0) -> SyntheticDataset:
    """Build one synthetic problem deterministically."""
    rng = np.random.default_rng(seed)
    X = _contract_frame(rows, rng)

    if problem is SyntheticProblem.LINEARLY_SEPARABLE:
        # A clean linear rule in two standardised features, with a small margin
        # of noise so the problem is learnable but not degenerate.
        score = 0.6 * _z(X["GenHlth"]) + 0.6 * _z(X["BMI"]) + rng.normal(0, 0.25, rows)
        y = (score > 0).astype(int)
        return SyntheticDataset(
            problem, X, pd.Series(y, name="target"), 0.90,
            "A linear boundary in two features. Every model here should clear 0.90.",
        )

    if problem is SyntheticProblem.NONLINEAR_XOR:
        # Exclusive-or over two binary features: no linear boundary exists, but
        # a single depth-2 tree solves it exactly.
        parity = (X["HighBP"].astype(int) ^ X["HighChol"].astype(int)).to_numpy()
        flip = rng.random(rows) < 0.05
        y = np.where(flip, 1 - parity, parity)
        return SyntheticDataset(
            problem, X, pd.Series(y, name="target"), 0.80,
            "XOR of two binaries with 5% label noise, among eight irrelevant "
            "features. No linear boundary exists, and neither XOR column has "
            "marginal information gain, so a single greedy tree splits on the "
            "distractors instead and also fails. Feature subsampling recovers "
            "it, which makes this a test of the splitting strategy rather than "
            "of the hypothesis class.",
        )

    if problem is SyntheticProblem.CLASS_IMBALANCE:
        score = 0.8 * _z(X["GenHlth"]) + rng.normal(0, 0.4, rows)
        threshold = float(np.quantile(score, 0.9))  # ~10% positive
        y = (score > threshold).astype(int)
        return SyntheticDataset(
            problem, X, pd.Series(y, name="target"), 0.80,
            "A learnable signal at roughly 10% prevalence, unlike the real "
            "dataset's engineered 50%. Catches metrics that assume balance.",
        )

    if problem is SyntheticProblem.NOISE_ONLY:
        # Labels independent of every feature. The negative control.
        y = rng.integers(0, 2, rows)
        return SyntheticDataset(
            problem, X, pd.Series(y, name="target"), None,
            "Labels independent of the features. No model can beat chance on "
            "held-out rows; one that does reveals leakage in the harness.",
        )

    raise ValueError(f"unknown synthetic problem: {problem!r}")


def _z(column: pd.Series) -> np.ndarray:
    values = column.to_numpy(dtype=float)
    spread = float(values.std())
    if spread == 0:
        return np.zeros_like(values)
    standardised: np.ndarray = (values - values.mean()) / spread
    return standardised


def split(
    dataset: SyntheticDataset, *, train_fraction: float = 0.6, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """A deterministic train/held-out split of a synthetic dataset.

    Used by the negative control, which is only meaningful when scored on rows
    the model never saw.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.X))
    cut = int(len(order) * train_fraction)
    train, held_out = order[:cut], order[cut:]
    return (
        dataset.X.iloc[train].reset_index(drop=True),
        dataset.y.iloc[train].reset_index(drop=True),
        dataset.X.iloc[held_out].reset_index(drop=True),
        dataset.y.iloc[held_out].reset_index(drop=True),
    )
