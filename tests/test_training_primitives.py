"""Shared, model-agnostic training primitives.

Self-contained: these exercise only ml_core.training and the feature contract,
so the primitives can be trusted before either pipeline is decomposed onto them.
Pipeline stages and the end-to-end smokes live in tests/test_training_pipelines.py.
"""
import json

import numpy as np
import pandas as pd
import pytest

from ml_core import training
from ml_core.feature_contract import FEATURE_NAMES, TARGET_COLUMN
from ml_core.training import DatasetValidationError

FEATURES = list(FEATURE_NAMES)


def make_dataset(rows: int = 40, seed: int = 0) -> pd.DataFrame:
    """A tiny, deterministic, contract-shaped dataset.

    For software verification only - it makes no claim about model quality.
    """
    rng = np.random.RandomState(seed)
    target = np.array([0, 1] * (rows // 2))
    data = {}
    for name in FEATURES:
        spec_low, spec_high = 0, 1
        if name == "BMI":
            data[name] = np.clip(24.0 + 6.0 * target + rng.normal(0, 2, rows), 10, 80)
            continue
        if name == "GenHlth":
            spec_low, spec_high = 1, 5
        elif name == "Age":
            spec_low, spec_high = 1, 13
        elif name == "Education":
            spec_low, spec_high = 1, 6
        elif name == "PhysHlth":
            spec_low, spec_high = 0, 30
        span = spec_high - spec_low
        shifted = spec_low + (target * span * 0.5) + rng.randint(0, max(span // 2, 1) + 1, rows)
        data[name] = np.clip(shifted, spec_low, spec_high).astype(int)
    data[TARGET_COLUMN] = target
    return pd.DataFrame(data)


# ============================================================ dataset contract

def test_a_valid_dataset_passes():
    training.validate_training_dataset(make_dataset(), FEATURES, TARGET_COLUMN)


def test_missing_target_is_rejected():
    frame = make_dataset().drop(columns=[TARGET_COLUMN])

    with pytest.raises(DatasetValidationError, match="missing target column"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_missing_feature_is_rejected():
    frame = make_dataset().drop(columns=["BMI"])

    with pytest.raises(DatasetValidationError, match="BMI"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_duplicated_column_is_rejected():
    frame = make_dataset()
    frame = pd.concat([frame, frame[["BMI"]]], axis=1)

    with pytest.raises(DatasetValidationError, match="duplicated columns"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_empty_dataset_is_rejected():
    with pytest.raises(DatasetValidationError, match="empty"):
        training.validate_training_dataset(
            pd.DataFrame(columns=[*FEATURES, TARGET_COLUMN]), FEATURES, TARGET_COLUMN
        )


def test_single_target_class_is_rejected():
    frame = make_dataset()
    frame[TARGET_COLUMN] = 0

    with pytest.raises(DatasetValidationError, match="single class"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_non_binary_target_is_rejected():
    frame = make_dataset()
    frame.loc[0, TARGET_COLUMN] = 7

    with pytest.raises(DatasetValidationError, match="binary"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_entirely_null_feature_column_is_rejected():
    frame = make_dataset()
    frame["BMI"] = np.nan

    with pytest.raises(DatasetValidationError, match="entirely null"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_entirely_null_target_is_rejected():
    frame = make_dataset()
    frame[TARGET_COLUMN] = np.nan

    with pytest.raises(DatasetValidationError, match="entirely null"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_non_numeric_feature_is_rejected():
    frame = make_dataset()
    frame["BMI"] = "not a number"

    with pytest.raises(DatasetValidationError, match="not numeric"):
        training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_numeric_strings_are_accepted_as_coercible():
    """Existing pipelines rely on pandas coercion; do not tighten that."""
    frame = make_dataset()
    frame["BMI"] = frame["BMI"].astype(str)

    training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_the_real_committed_dataset_satisfies_the_contract():
    """The G3 gap: the contract was never checked against the actual data."""
    from conftest import REPO_ROOT

    frame = pd.read_csv(REPO_ROOT / "cleaned_data.csv")

    training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)


def test_loader_validates_and_reports_a_missing_file(tmp_path):
    with pytest.raises(DatasetValidationError, match="not found"):
        training.load_training_dataset(tmp_path / "absent.csv", FEATURES, TARGET_COLUMN)


def test_loader_returns_a_validated_frame(tmp_path):
    csv = tmp_path / "d.csv"
    make_dataset().to_csv(csv, index=False)

    frame = training.load_training_dataset(csv, FEATURES, TARGET_COLUMN)

    assert len(frame) == 40
    assert TARGET_COLUMN in frame.columns


def test_feature_selection_uses_canonical_order():
    frame = make_dataset()[[TARGET_COLUMN, *reversed(FEATURES)]]

    X, y = training.select_features(frame, FEATURES, TARGET_COLUMN)

    assert list(X.columns) == FEATURES
    assert y.name == TARGET_COLUMN


# ==================================================================== splits

def test_split_is_deterministic_for_a_fixed_seed():
    X, y = training.select_features(make_dataset(200), FEATURES, TARGET_COLUMN)

    first = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)
    second = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    assert list(first.X_test.index) == list(second.X_test.index)
    assert list(first.X_val.index) == list(second.X_val.index)


def test_split_seed_changes_the_partition():
    X, y = training.select_features(make_dataset(200), FEATURES, TARGET_COLUMN)

    a = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=1)
    b = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=2)

    assert list(a.X_test.index) != list(b.X_test.index)


def test_split_proportions_follow_the_documented_semantics():
    X, y = training.select_features(make_dataset(200), FEATURES, TARGET_COLUMN)

    splits = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    assert splits.sizes["test"] == 40           # 20% of 200
    assert splits.sizes["val"] == 40            # 25% of the remaining 160
    assert splits.sizes["train"] == 120
    assert sum(splits.sizes.values()) == 200


def test_split_partitions_are_disjoint():
    X, y = training.select_features(make_dataset(200), FEATURES, TARGET_COLUMN)

    splits = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    train, val, test = (set(part.index) for part in (splits.X_train, splits.X_val, splits.X_test))
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_split_is_stratified():
    X, y = training.select_features(make_dataset(200), FEATURES, TARGET_COLUMN)

    splits = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    for part in (splits.y_train, splits.y_val, splits.y_test):
        assert set(part.unique()) == {0, 1}
        assert abs(part.mean() - 0.5) < 0.05


def test_split_preserves_canonical_feature_order():
    X, y = training.select_features(make_dataset(80), FEATURES, TARGET_COLUMN)

    splits = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    assert splits.feature_names == tuple(FEATURE_NAMES)
    for part in (splits.X_train, splits.X_val, splits.X_test):
        assert list(part.columns) == FEATURES


def test_splits_are_immutable():
    import dataclasses

    X, y = training.select_features(make_dataset(40), FEATURES, TARGET_COLUMN)
    splits = training.split_training_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    with pytest.raises(dataclasses.FrozenInstanceError):
        splits.X_train = None


# ================================================================ evaluation

def test_evaluate_at_threshold_matches_manual_application():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])

    y_pred, metrics = training.evaluate_at_threshold(y_true, y_proba, 0.5)

    assert list(y_pred) == [0, 1, 0, 1]
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert json.loads(json.dumps(metrics)) == metrics


def test_positive_class_proba_returns_a_flat_array():
    class Stub:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.linspace(0, 1, len(X))])

    result = training.positive_class_proba(Stub(), pd.DataFrame({"a": [1, 2, 3]}))

    assert result.shape == (3,)
    assert result[-1] == 1.0


# ============================================================ atomic writing

def test_write_json_atomic_writes_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "out" / "metrics.json"

    training.write_json_atomic({"b": 2, "a": 1}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 2, "a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["metrics.json"]


def test_write_json_atomic_uses_lf_endings(tmp_path):
    target = tmp_path / "m.json"

    training.write_json_atomic({"a": 1}, target)

    assert b"\r\n" not in target.read_bytes()


def test_write_json_atomic_leaves_nothing_behind_on_failure(tmp_path):
    target = tmp_path / "m.json"

    with pytest.raises(TypeError):
        training.write_json_atomic({"bad": object()}, target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_json_atomic_replaces_an_existing_file(tmp_path):
    target = tmp_path / "m.json"
    training.write_json_atomic({"v": 1}, target)

    training.write_json_atomic({"v": 2}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


# ============================================================ fixture dataset

def test_fixture_dataset_satisfies_the_contract_and_both_classes():
    frame = make_dataset(60, seed=3)

    training.validate_training_dataset(frame, FEATURES, TARGET_COLUMN)
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}
    assert list(frame.columns) == [*FEATURES, TARGET_COLUMN]


def test_fixture_dataset_is_deterministic():
    assert make_dataset(40, seed=7).equals(make_dataset(40, seed=7))


def test_fixture_dataset_respects_contract_bounds():
    from ml_core import feature_contract

    frame = make_dataset(60, seed=5)
    for name in FEATURES:
        spec = feature_contract.spec_for(name)
        assert frame[name].min() >= spec.minimum, name
        assert frame[name].max() <= spec.maximum, name
