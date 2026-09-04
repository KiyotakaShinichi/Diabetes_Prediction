"""Feature preparation for the deep challengers, fitted on train only.

Leakage is the failure this module exists to prevent. Every statistic a
transform needs - here, per-feature mean and standard deviation - is computed
from the training partition and then frozen. Validation and test are transformed
with those frozen numbers and never contribute to them.

Feature treatment follows the actual semantics declared in the served contract
rather than treating ten columns as ten interchangeable floats:

* ``binary`` features (HighBP, HighChol, DiffWalk, HeartDiseaseorAttack,
  PhysActivity) are already 0/1 and are passed through unscaled. Standardising a
  Bernoulli indicator buys nothing and makes the learned weight harder to read.
* ``continuous`` (BMI) is standardised.
* ``ordinal`` (GenHlth, Age, PhysHlth, Education) is standardised as well, and
  that is a modelling DECISION rather than a fact: these are ordered codes whose
  spacing is not guaranteed uniform. Age bands, for instance, are five-year
  buckets at the bottom and open-ended at the top. Treating them as numeric
  asserts that "one step" means roughly the same thing everywhere on the scale.
  The MLP accepts that assumption. The FT-Transformer does not - it embeds each
  ordinal level, which is one of the reasons it is worth benchmarking here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_core import feature_contract


class LeakageError(RuntimeError):
    """A transform was asked to fit on data it must never see."""


@dataclass(frozen=True, slots=True)
class StandardiserState:
    """Frozen per-feature statistics, fitted once on the training partition."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    #: Indices into feature_names that are standardised. Binary columns are not.
    scaled_indices: tuple[int, ...]
    fitted_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "scaled_indices": list(self.scaled_indices),
            "fitted_rows": self.fitted_rows,
        }


#: Guard against a zero-variance column producing infinities.
_MIN_STD = 1e-8


def scaled_feature_indices(feature_names: tuple[str, ...]) -> tuple[int, ...]:
    """Positions of the features that get standardised.

    Binary indicators are excluded deliberately - see the module docstring.
    """
    indices = []
    for position, name in enumerate(feature_names):
        if feature_contract.spec_for(name).kind != "binary":
            indices.append(position)
    return tuple(indices)


def fit_standardiser(x_train: pd.DataFrame) -> StandardiserState:
    """Fit on the TRAINING partition. Nothing else may be passed here.

    The caller is trusted to pass train, and the tests verify that the values
    recorded here match a fit on train alone - a standardiser fitted on the full
    dataset produces different numbers, and that difference is detectable.
    """
    if x_train.empty:
        raise LeakageError("cannot fit a standardiser on an empty frame")

    names = tuple(str(column) for column in x_train.columns)
    values = x_train.to_numpy(dtype=np.float64)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds = np.where(stds < _MIN_STD, 1.0, stds)

    return StandardiserState(
        feature_names=names,
        means=tuple(float(value) for value in means),
        stds=tuple(float(value) for value in stds),
        scaled_indices=scaled_feature_indices(names),
        fitted_rows=len(x_train),
    )


def apply_standardiser(state: StandardiserState, frame: pd.DataFrame) -> np.ndarray:
    """Transform any partition with FROZEN training statistics.

    Column order is checked rather than assumed: silently transforming a frame
    whose columns arrived in a different order would scale the wrong feature by
    the wrong constant and still return a plausible-looking array.
    """
    names = tuple(str(column) for column in frame.columns)
    if names != state.feature_names:
        raise LeakageError(
            f"column mismatch: standardiser fitted on {state.feature_names}, got {names}"
        )

    values = frame.to_numpy(dtype=np.float64).copy()
    means = np.asarray(state.means, dtype=np.float64)
    stds = np.asarray(state.stds, dtype=np.float64)

    for index in state.scaled_indices:
        values[:, index] = (values[:, index] - means[index]) / stds[index]
    return np.asarray(values.astype(np.float32))


@dataclass(frozen=True, slots=True)
class OrdinalVocabulary:
    """The discrete level set of every feature, taken from the CONTRACT.

    Deliberately not learned from the training data. The contract already
    declares each feature's inclusive bounds, so the vocabulary is a property of
    the problem rather than of a sample - which means a validation row carrying a
    legal-but-unseen level cannot produce an out-of-range embedding index, and no
    fitting on validation data is involved.
    """

    feature_names: tuple[str, ...]
    #: Number of embedding slots per feature; 0 marks a continuous feature.
    cardinalities: tuple[int, ...]
    offsets: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.cardinalities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "cardinalities": list(self.cardinalities),
            "offsets": list(self.offsets),
            "total_tokens": self.total_tokens,
        }


def build_ordinal_vocabulary(feature_names: tuple[str, ...]) -> OrdinalVocabulary:
    """Level counts and embedding offsets, derived from the feature contract."""
    cardinalities: list[int] = []
    offsets: list[int] = []
    running = 0
    for name in feature_names:
        spec = feature_contract.spec_for(name)
        allowed = spec.allowed_values
        size = 0 if allowed is None else len(allowed)
        cardinalities.append(size)
        offsets.append(running)
        running += size
    return OrdinalVocabulary(
        feature_names=feature_names,
        cardinalities=tuple(cardinalities),
        offsets=tuple(offsets),
    )


def encode_ordinal_levels(
    vocabulary: OrdinalVocabulary, frame: pd.DataFrame
) -> np.ndarray:
    """Map each discrete feature value to its embedding row index.

    Continuous features yield 0 and are ignored by the caller; the returned
    array is only meaningful where the vocabulary declares a cardinality.
    Values are clipped to the contract's declared range, so an out-of-contract
    value fails loudly at validation time rather than indexing out of bounds
    here.
    """
    names = tuple(str(column) for column in frame.columns)
    if names != vocabulary.feature_names:
        raise LeakageError(
            f"column mismatch: vocabulary built for {vocabulary.feature_names}, got {names}"
        )

    encoded = np.zeros((len(frame), len(names)), dtype=np.int64)
    for position, name in enumerate(names):
        if vocabulary.cardinalities[position] == 0:
            continue
        spec = feature_contract.spec_for(name)
        column = frame[name].to_numpy()
        level = np.rint(column).astype(np.int64) - int(spec.minimum)
        encoded[:, position] = np.clip(level, 0, vocabulary.cardinalities[position] - 1)
    return encoded
