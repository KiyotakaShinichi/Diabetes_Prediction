"""Invariants of the canonical feature contract itself.

Self-contained: these exercise only ml_core.feature_contract, so the module can
be trusted before anything is migrated onto it. Cross-layer agreement - API,
artifacts, pipelines, UI, drift, provenance - lives in
tests/test_schema_equivalence.py.
"""
import dataclasses

import pandas as pd
import pytest

from conftest import VALID_PAYLOAD
from ml_core import feature_contract
from ml_core.feature_contract import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSpec,
)

# ================================================= the contract itself

def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_feature_order_is_stable_and_explicit():
    assert tuple(spec.name for spec in FEATURE_SPECS) == FEATURE_NAMES
    assert tuple(feature_contract.feature_list()) == FEATURE_NAMES
    assert feature_contract.FEATURE_COUNT == len(FEATURE_NAMES) == 10


def test_every_spec_has_a_display_label_and_description():
    for spec in FEATURE_SPECS:
        assert spec.display_label.strip(), spec.name
        assert spec.description.strip(), spec.name
    assert set(FEATURE_LABELS) == set(FEATURE_NAMES)


def test_bounds_are_internally_valid():
    for spec in FEATURE_SPECS:
        assert spec.minimum <= spec.maximum, spec.name


def test_discrete_domains_are_valid():
    for spec in FEATURE_SPECS:
        if spec.kind == "continuous":
            assert spec.allowed_values is None
            continue
        values = spec.allowed_values
        assert values == tuple(range(int(spec.minimum), int(spec.maximum) + 1))
        if spec.kind == "binary":
            assert values == (0, 1), spec.name


def test_specs_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FEATURE_SPECS[0].name = "Tampered"


def test_a_binary_feature_with_wrong_bounds_is_rejected():
    with pytest.raises(ValueError, match="binary"):
        FeatureSpec("X", "X", "x", "binary", int, 0, 5)


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="exceeds maximum"):
        FeatureSpec("X", "X", "x", "continuous", float, 10, 1)


def test_spec_lookup_rejects_an_unknown_feature():
    with pytest.raises(KeyError, match="not a served feature"):
        feature_contract.spec_for("NotAFeature")


def test_order_columns_rejects_a_missing_feature():
    frame = pd.DataFrame([{name: 1 for name in FEATURE_NAMES if name != "BMI"}])

    with pytest.raises(KeyError, match="BMI"):
        feature_contract.order_columns(frame)


def test_order_columns_ignores_incoming_column_order():
    shuffled = {name: VALID_PAYLOAD[name] for name in reversed(FEATURE_NAMES)}

    ordered = feature_contract.order_columns(pd.DataFrame([shuffled]))

    assert list(ordered.columns) == list(FEATURE_NAMES)
