"""Deterministic training utilities and the committed model artifacts.

Nothing here retrains a model or writes to ``model_artifacts/``: the reusable
helpers are exercised on tiny synthetic arrays, and the shipped artifacts are
read-only inputs to contract checks.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from app import DiabetesFeatures
from conftest import ARTIFACTS_DIR

# Import-time side effect, deliberately contained: both training modules call
# warnings.filterwarnings("ignore") and optuna.logging.set_verbosity() at import,
# which would silence warnings for the whole pytest session. Snapshot the filters
# and put them back so the rest of the suite still reports warnings normally.
_SAVED_WARNING_FILTERS = warnings.filters[:]
import boostedtrees_ab as xgb_train  # noqa: E402
import logisticregression_only as lr_train  # noqa: E402

warnings.filters[:] = _SAVED_WARNING_FILTERS

TRAINING_MODULES = pytest.mark.parametrize(
    "training", [lr_train, xgb_train], ids=["logistic_regression", "boosted_trees"]
)

# Hand-computed fixture: confusion matrix [[TN=2, FP=2], [FN=1, TP=3]].
Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_PROBA = np.array([0.10, 0.20, 0.60, 0.70, 0.40, 0.80, 0.90, 0.55])
Y_PRED = (Y_PROBA >= 0.5).astype(int)

METRIC_NAMES = {"accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"}
EVALUATION_KEYS = METRIC_NAMES | {"cohen_kappa", "mcc", "confusion_matrix"}

ARTIFACT_CASES = [
    pytest.param("model_bundle.pkl", "metrics.json", "drift_baseline.pkl",
                 "logistic_regression", id="variant-A"),
    pytest.param("boosted_model_bundle.pkl", "boosted_metrics.json", "boosted_drift_baseline.pkl",
                 "xgboost_boosted_trees", id="variant-B"),
]


def _synthetic_scores(n: int = 60, seed: int = 7):
    """A separable-but-imperfect binary problem, large enough to bootstrap."""
    rng = np.random.RandomState(seed)
    y_true = np.array([0, 1] * (n // 2))
    y_proba = np.clip(rng.normal(loc=0.35 + 0.3 * y_true, scale=0.12), 0.001, 0.999)
    return y_true, y_proba


# ----------------------------------------------------------- Youden threshold

@TRAINING_MODULES
def test_youden_threshold_perfectly_separates_a_separable_problem(training):
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = training.compute_youden_threshold(y_true, y_proba)

    assert 0.0 <= threshold <= 1.0
    assert np.array_equal((y_proba >= threshold).astype(int), y_true)


@TRAINING_MODULES
def test_youden_threshold_is_deterministic_and_an_observed_score(training):
    y_true, y_proba = _synthetic_scores()

    first = training.compute_youden_threshold(y_true, y_proba)
    second = training.compute_youden_threshold(y_true, y_proba)

    assert first == second
    assert 0.0 <= first <= 1.0
    assert first in set(y_proba.tolist())


def test_both_training_modules_agree_on_the_youden_threshold():
    """Variant A and variant B must select thresholds the same way."""
    y_true, y_proba = _synthetic_scores()

    assert lr_train.compute_youden_threshold(y_true, y_proba) == pytest.approx(
        xgb_train.compute_youden_threshold(y_true, y_proba)
    )


@TRAINING_MODULES
@pytest.mark.parametrize(
    "y_proba",
    [np.full(6, 0.5), np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])],
    ids=["constant-scores", "inverted-scores"],
)
def test_youden_threshold_degenerates_to_infinity_for_useless_scores(training, y_proba):
    """Known sharp edge, pinned rather than papered over.

    When no cut-point achieves TPR > FPR, ``np.argmax`` lands on the synthetic
    ``inf`` entry that ``sklearn.metrics.roc_curve`` prepends, so the helper
    returns ``inf`` - a threshold that would classify everything as negative.
    Real training never hits this (the shipped thresholds are ~0.46 / ~0.52),
    but the behaviour is undefended and this test will fail loudly the moment
    someone fixes it, which is the point.
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])

    assert not np.isfinite(training.compute_youden_threshold(y_true, y_proba))


# ------------------------------------------------------- evaluate_predictions

@TRAINING_MODULES
def test_evaluate_predictions_returns_a_stable_key_set(training):
    metrics = training.evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA)

    assert set(metrics) == EVALUATION_KEYS


@TRAINING_MODULES
def test_evaluate_predictions_matches_hand_computed_values(training):
    metrics = training.evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA)

    assert metrics["confusion_matrix"] == [[2, 2], [1, 3]]
    assert metrics["accuracy"] == pytest.approx(5 / 8)
    assert metrics["precision"] == pytest.approx(3 / 5)
    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["brier_score"] == pytest.approx(1.5125 / 8)
    assert metrics["cohen_kappa"] == pytest.approx(0.25)
    assert metrics["mcc"] == pytest.approx(4 / np.sqrt(240))


@TRAINING_MODULES
def test_evaluate_predictions_is_json_serialisable(training):
    """metrics.json is written straight from this dict, so it must serialise."""
    metrics = training.evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA)

    assert json.loads(json.dumps(metrics)) == metrics


@TRAINING_MODULES
def test_evaluate_predictions_survives_an_all_negative_prediction(training):
    """zero_division=0 must keep a degenerate prediction from raising."""
    metrics = training.evaluate_predictions(Y_TRUE, np.zeros_like(Y_TRUE), Y_PROBA)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


# ---------------------------------------------- bootstrap confidence intervals

@TRAINING_MODULES
def test_bootstrap_intervals_bracket_their_own_mean(training):
    y_true, y_proba = _synthetic_scores()

    result = training.bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=40)

    assert set(result) == METRIC_NAMES
    for name, stats in result.items():
        assert set(stats) == {"mean", "ci_lower", "ci_upper"}
        assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"], name
        assert 0.0 <= stats["ci_lower"] <= 1.0
        assert 0.0 <= stats["ci_upper"] <= 1.0


@TRAINING_MODULES
def test_bootstrap_is_reproducible_for_a_fixed_seed(training):
    y_true, y_proba = _synthetic_scores()

    first = training.bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=123)
    second = training.bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=123)

    assert first == second


@TRAINING_MODULES
def test_bootstrap_seed_actually_changes_the_resampling(training):
    y_true, y_proba = _synthetic_scores()

    a = training.bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=1)
    b = training.bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=2)

    assert a["accuracy"]["mean"] != b["accuracy"]["mean"]


@TRAINING_MODULES
def test_bootstrap_fails_loudly_when_every_resample_is_single_class(training):
    """Degenerate input must not silently yield NaN intervals."""
    y_true = np.zeros(8, dtype=int)
    y_proba = np.linspace(0.0, 1.0, 8)

    with pytest.raises((IndexError, ValueError)):
        training.bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=5)


# ------------------------------------------------------------ drift baselines

def test_logistic_drift_baseline_is_per_feature_statistics():
    frame = pd.DataFrame({"BMI": [20.0, 30.0, 40.0], "Age": [1.0, 5.0, 9.0]})

    baseline = lr_train.compute_drift_baseline(frame)

    assert set(baseline) == {"BMI", "Age"}
    assert set(baseline["BMI"]) == {"mean", "std", "min", "max", "q25", "median", "q75"}
    assert baseline["BMI"]["mean"] == pytest.approx(30.0)
    assert baseline["BMI"]["median"] == pytest.approx(30.0)
    assert baseline["BMI"]["min"] == pytest.approx(20.0)
    assert baseline["BMI"]["max"] == pytest.approx(40.0)
    # app.py distinguishes the two formats on this key, so it must stay absent.
    assert "feature_columns" not in baseline


def test_boosted_drift_baseline_is_column_wise_statistics():
    frame = pd.DataFrame({"BMI": [20.0, 30.0, 40.0], "Age": [1.0, 5.0, 9.0]})

    baseline = xgb_train.compute_drift_baseline(frame)

    assert set(baseline) == {"means", "stds", "medians", "q25", "q75", "n_train", "feature_columns"}
    assert baseline["feature_columns"] == ["BMI", "Age"]
    assert baseline["n_train"] == 3
    assert baseline["means"]["BMI"] == pytest.approx(30.0)


# -------------------------------------------------------- committed artifacts

def test_the_feature_contract_is_identical_across_training_and_serving():
    """SELECTED_FEATURES, the API schema, and both bundles must not drift apart."""
    api_fields = list(DiabetesFeatures.model_fields)
    lr_bundle = joblib.load(ARTIFACTS_DIR / "model_bundle.pkl")
    xgb_bundle = joblib.load(ARTIFACTS_DIR / "boosted_model_bundle.pkl")

    assert lr_train.SELECTED_FEATURES == xgb_train.SELECTED_FEATURES
    assert sorted(api_fields) == sorted(lr_train.SELECTED_FEATURES)
    assert list(lr_bundle["feature_columns"]) == lr_train.SELECTED_FEATURES
    assert list(xgb_bundle["feature_columns"]) == xgb_train.SELECTED_FEATURES


@pytest.mark.parametrize(("bundle_name", "metrics_name", "drift_name", "model_name"), ARTIFACT_CASES)
def test_bundle_and_metrics_agree_on_the_serving_threshold(
    bundle_name, metrics_name, drift_name, model_name
):
    bundle = joblib.load(ARTIFACTS_DIR / bundle_name)
    metrics = json.loads((ARTIFACTS_DIR / metrics_name).read_text(encoding="utf-8"))

    assert bundle["model_name"] == model_name
    assert 0.0 < bundle["threshold"] < 1.0
    assert bundle["threshold"] == pytest.approx(metrics["threshold"])


@pytest.mark.parametrize(("bundle_name", "metrics_name", "drift_name", "model_name"), ARTIFACT_CASES)
def test_committed_metrics_are_internally_consistent(
    bundle_name, metrics_name, drift_name, model_name
):
    metrics = json.loads((ARTIFACTS_DIR / metrics_name).read_text(encoding="utf-8"))

    assert {"threshold", "test_metrics", "validation_metrics", "confidence_intervals"} <= set(metrics)
    test_metrics = metrics["test_metrics"]
    assert set(test_metrics) >= EVALUATION_KEYS

    (tn, fp), (fn, tp) = test_metrics["confusion_matrix"]
    assert test_metrics["accuracy"] == pytest.approx((tp + tn) / (tp + tn + fp + fn))
    assert test_metrics["precision"] == pytest.approx(tp / (tp + fp))
    assert test_metrics["recall"] == pytest.approx(tp / (tp + fn))
    assert 0.5 < test_metrics["roc_auc"] <= 1.0


@pytest.mark.parametrize(("bundle_name", "metrics_name", "drift_name", "model_name"), ARTIFACT_CASES)
def test_committed_confidence_intervals_contain_the_point_estimates(
    bundle_name, metrics_name, drift_name, model_name
):
    metrics = json.loads((ARTIFACTS_DIR / metrics_name).read_text(encoding="utf-8"))
    intervals = metrics["confidence_intervals"]

    assert set(intervals) == METRIC_NAMES
    for name, stats in intervals.items():
        point = metrics["test_metrics"][name]
        assert stats["ci_lower"] <= point <= stats["ci_upper"], name


@pytest.mark.parametrize(("bundle_name", "metrics_name", "drift_name", "model_name"), ARTIFACT_CASES)
def test_committed_drift_baselines_cover_every_served_feature(
    bundle_name, metrics_name, drift_name, model_name
):
    """Whichever of the two formats is on disk, /drift-check must be able to read it."""
    baseline = joblib.load(ARTIFACTS_DIR / drift_name)

    if "feature_columns" in baseline:
        covered = list(baseline["feature_columns"])
        stds = [baseline["stds"][feature] for feature in covered]
    else:
        covered = list(baseline)
        stds = [baseline[feature]["std"] for feature in covered]

    assert sorted(covered) == sorted(lr_train.SELECTED_FEATURES)
    assert all(std > 0 for std in stds), "a zero std would silently zero out every z-score"
