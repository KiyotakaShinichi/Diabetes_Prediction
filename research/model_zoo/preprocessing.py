"""Preprocessing declared per model, fitted on training rows only.

Every transformer here is built inside an sklearn ``Pipeline`` so that fitting
the pipeline fits the transformer on exactly the rows the model was fitted on.
That is not a stylistic preference. A scaler fitted before the split - or on
train plus validation - leaks distributional information from rows the model is
later scored on, and the leak is invisible in the metrics because it makes them
better.

`tests/test_model_zoo_leakage.py` asserts the property directly, by fitting on
one partition and checking the learned statistics against that partition alone.
"""
from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from research.model_zoo.contracts import Preprocessing


def build_transformer(kind: Preprocessing) -> Any | None:
    """The transformer a preprocessing contract calls for, or None.

    ``RAW_NUMERIC`` and ``MODEL_NATIVE`` return None: the first because trees
    are scale-invariant and inserting a scaler would only add a fitted object
    to serialize, the second because the model owns its own preparation.
    """
    if kind is Preprocessing.STANDARDIZED:
        return StandardScaler()
    if kind is Preprocessing.ROBUST_SCALED:
        # Median and IQR rather than mean and variance. BMI and PhysHlth both
        # have long right tails in this dataset, and a mean-centred scale lets
        # those tails move every other feature's position.
        return RobustScaler()
    if kind in (Preprocessing.RAW_NUMERIC, Preprocessing.MODEL_NATIVE):
        return None
    raise ValueError(f"unknown preprocessing contract: {kind!r}")


def wrap(estimator: Any, kind: Preprocessing) -> Any:
    """Compose an estimator with its declared preprocessing.

    Returns the bare estimator when no transform is called for, so a tree model
    is not wrapped in a one-step pipeline for the sake of symmetry.
    """
    transformer = build_transformer(kind)
    if transformer is None:
        return estimator
    return Pipeline([("prepare", transformer), ("model", estimator)])


def requires_scaling(kind: Preprocessing) -> bool:
    """Whether this contract actually rescales the features."""
    return kind in (Preprocessing.STANDARDIZED, Preprocessing.ROBUST_SCALED)
