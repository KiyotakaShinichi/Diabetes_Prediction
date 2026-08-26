"""Contract and equivalence tests for the shared ML evaluation core.

The reference implementations below are verbatim copies of the code that lived
in logisticregression_only.py and boostedtrees_ab.py before extraction. They are
here so equivalence is proved against the *old behaviour itself*, not against
committed model outputs - tuning new code until it reproduces a stored
metrics.json would be circular.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import ml_core
from conftest import REPO_ROOT
from ml_core import (
    bootstrap_confidence_interval,
    compute_youden_threshold,
    evaluate_predictions,
)

MAINTAINED_PIPELINES = ["logisticregression_only.py", "boostedtrees_ab.py"]
SHARED_FUNCTIONS = [
    "evaluate_predictions",
    "bootstrap_confidence_interval",
    "compute_youden_threshold",
]


# --------------------------------------------------- pinned old behaviour

def _old_compute_youden_threshold(y_true, y_proba):
    """Pre-extraction implementation. Could return inf; that was the defect."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thresholds[best_idx])


def _old_evaluate_predictions(y_true, y_pred, y_proba):
    """Pre-extraction implementation, byte-identical in both pipelines."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _old_bootstrap_confidence_interval(y_true, y_proba, threshold, n_bootstrap=200,
                                       alpha=0.05, seed=42):
    """Pre-extraction implementation."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_boot = {"accuracy": [], "precision": [], "recall": [],
                    "f1": [], "roc_auc": [], "brier_score": []}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p = y_proba[idx]
        y_pred = (y_p >= threshold).astype(int)
        if len(np.unique(y_t)) < 2:
            continue
        metrics_boot["accuracy"].append(float(accuracy_score(y_t, y_pred)))
        metrics_boot["precision"].append(float(precision_score(y_t, y_pred, zero_division=0)))
        metrics_boot["recall"].append(float(recall_score(y_t, y_pred, zero_division=0)))
        metrics_boot["f1"].append(float(f1_score(y_t, y_pred, zero_division=0)))
        metrics_boot["roc_auc"].append(float(roc_auc_score(y_t, y_p)))
        metrics_boot["brier_score"].append(float(brier_score_loss(y_t, y_p)))
    result = {}
    for metric, values in metrics_boot.items():
        arr = np.array(values)
        result[metric] = {
            "mean": float(arr.mean()),
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return result


def _informative_case(seed, n=80):
    """A separable-but-imperfect problem, the shape real evaluation data takes."""
    rng = np.random.RandomState(seed)
    y_true = np.array([0, 1] * (n // 2))
    y_proba = np.clip(rng.normal(loc=0.35 + 0.3 * y_true, scale=0.12), 0.001, 0.999)
    return y_true, y_proba


# ================================================================ evaluation

# Hand-computed: confusion matrix [[TN=2, FP=2], [FN=1, TP=3]].
Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_PROBA = np.array([0.10, 0.20, 0.60, 0.70, 0.40, 0.80, 0.90, 0.55])
Y_PRED = (Y_PROBA >= 0.5).astype(int)


def test_evaluation_matches_hand_computed_values():
    metrics = evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA)

    assert metrics["confusion_matrix"] == [[2, 2], [1, 3]]
    assert metrics["accuracy"] == pytest.approx(5 / 8)
    assert metrics["precision"] == pytest.approx(3 / 5)
    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["brier_score"] == pytest.approx(1.5125 / 8)
    assert metrics["cohen_kappa"] == pytest.approx(0.25)
    assert metrics["mcc"] == pytest.approx(4 / np.sqrt(240))


def test_evaluation_key_set_is_stable():
    assert set(evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA)) == set(ml_core.EVALUATION_KEYS)


def test_evaluation_result_is_json_serializable():
    metrics = evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA)

    assert json.loads(json.dumps(metrics)) == metrics


def test_evaluation_zero_division_policy():
    metrics = evaluate_predictions(Y_TRUE, np.zeros_like(Y_TRUE), Y_PROBA)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluation_is_deterministic():
    assert evaluate_predictions(Y_TRUE, Y_PRED, Y_PROBA) == evaluate_predictions(
        Y_TRUE, Y_PRED, Y_PROBA
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
def test_evaluation_is_equivalent_to_the_old_implementation(seed):
    y_true, y_proba = _informative_case(seed)
    y_pred = (y_proba >= 0.5).astype(int)

    assert evaluate_predictions(y_true, y_pred, y_proba) == _old_evaluate_predictions(
        y_true, y_pred, y_proba
    )


# ================================================================= bootstrap

def test_bootstrap_is_reproducible_for_a_fixed_seed():
    y_true, y_proba = _informative_case(5)

    first = bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=123)
    second = bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=123)

    assert first == second


def test_bootstrap_seed_changes_the_resampling():
    y_true, y_proba = _informative_case(5)

    a = bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=1)
    b = bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=25, seed=2)

    assert a["accuracy"]["mean"] != b["accuracy"]["mean"]


def test_bootstrap_intervals_are_ordered_and_bracket_the_mean():
    y_true, y_proba = _informative_case(5)

    result = bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=40)

    assert set(result) == set(ml_core.BOOTSTRAP_METRICS)
    for name, stats in result.items():
        assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"], name
        assert 0.0 <= stats["ci_lower"] <= 1.0
        assert 0.0 <= stats["ci_upper"] <= 1.0


def test_bootstrap_does_not_touch_the_global_numpy_rng():
    """A local RandomState must be used; the global stream must be untouched."""
    y_true, y_proba = _informative_case(5)
    np.random.seed(1234)
    expected = np.random.random(3).tolist()

    np.random.seed(1234)
    bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=20)
    after = np.random.random(3).tolist()

    assert after == expected


def test_bootstrap_reports_a_clear_error_when_every_resample_is_degenerate():
    """Deliberate: the old code died on np.percentile of an empty array."""
    y_true = np.zeros(8, dtype=int)
    y_proba = np.linspace(0.0, 1.0, 8)

    with pytest.raises(ValueError, match="single class"):
        bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=5)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n_bootstrap", [10, 25])
def test_bootstrap_is_equivalent_to_the_old_implementation(seed, n_bootstrap):
    """Identical resample stream, identical intervals - to the last bit."""
    y_true, y_proba = _informative_case(seed)

    new = bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=n_bootstrap, seed=seed)
    old = _old_bootstrap_confidence_interval(y_true, y_proba, 0.5, n_bootstrap=n_bootstrap, seed=seed)

    assert new == old


# ================================================================ thresholds

def test_threshold_perfectly_separates_a_separable_problem():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = compute_youden_threshold(y_true, y_proba)

    assert np.array_equal((y_proba >= threshold).astype(int), y_true)


def test_threshold_on_an_ordinary_mixed_case():
    threshold = compute_youden_threshold(Y_TRUE, Y_PROBA)

    assert threshold == pytest.approx(0.8)
    assert threshold in set(Y_PROBA.tolist())


@pytest.mark.parametrize(
    ("name", "y_true", "y_proba"),
    [
        ("all-identical", np.array([0, 0, 0, 1, 1, 1]), np.full(6, 0.5)),
        ("no-better-than-random", np.array([0, 0, 0, 1, 1, 1]),
         np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])),
        ("extreme-0-1", np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])),
        ("tiny-balanced", np.array([0, 1]), np.array([0.3, 0.7])),
        ("perfect", np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])),
    ],
)
def test_threshold_is_always_finite_and_in_unit_interval(name, y_true, y_proba):
    """The core invariant: a serving threshold can never be inf, NaN or out of range."""
    threshold = compute_youden_threshold(y_true, y_proba)

    assert np.isfinite(threshold), name
    assert 0.0 <= threshold <= 1.0, name
    assert isinstance(threshold, float)


@pytest.mark.parametrize(
    ("name", "y_true", "y_proba"),
    [
        ("all-identical", np.array([0, 0, 0, 1, 1, 1]), np.full(6, 0.5)),
        ("no-better-than-random", np.array([0, 0, 0, 1, 1, 1]),
         np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])),
    ],
)
def test_degenerate_cases_that_used_to_return_infinity_now_return_a_real_cutpoint(
    name, y_true, y_proba
):
    """Documented, intentional behaviour change.

    The old helper returned inf here - a threshold that classifies every patient
    as negative. It now returns the best finite candidate under the same Youden
    criterion.
    """
    old = _old_compute_youden_threshold(y_true, y_proba)
    new = compute_youden_threshold(y_true, y_proba)

    assert np.isinf(old), f"{name}: expected the old defect to reproduce"
    assert np.isfinite(new)
    assert new in set(np.asarray(y_proba).tolist())


def test_single_class_target_is_rejected_rather_than_guessed():
    """The old helper returned inf; a neutral 0.5 would hide a broken test set."""
    y_true = np.zeros(5, dtype=int)
    y_proba = np.linspace(0.1, 0.9, 5)

    assert np.isinf(_old_compute_youden_threshold(y_true, y_proba))
    with pytest.raises(ValueError, match="both classes"):
        compute_youden_threshold(y_true, y_proba)


@pytest.mark.parametrize(
    ("y_true", "y_proba", "match"),
    [
        ([0, 1, 1], [0.1, 0.2], "same length"),
        ([], [], "must not be empty"),
        ([0, 1], [0.5, np.nan], "finite"),
        ([0, 1], [0.5, np.inf], "finite"),
        ([0, 1], [-0.1, 0.5], r"\[0, 1\]"),
        ([0, 1], [0.5, 1.5], r"\[0, 1\]"),
    ],
    ids=["mismatched", "empty", "nan", "inf", "below-zero", "above-one"],
)
def test_invalid_threshold_inputs_are_rejected(y_true, y_proba, match):
    with pytest.raises(ValueError, match=match):
        compute_youden_threshold(np.array(y_true), np.array(y_proba, dtype=float))


def test_threshold_tie_breaking_is_deterministic():
    """Ties resolve to the first maximum, i.e. the highest tied threshold."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])

    results = {compute_youden_threshold(y_true, y_proba) for _ in range(10)}

    assert len(results) == 1
    assert results.pop() == pytest.approx(0.6)


@pytest.mark.parametrize("seed", range(12))
def test_threshold_is_identical_to_the_old_implementation_when_the_old_one_was_valid(seed):
    """The equivalence guarantee: nothing changes where the old code worked."""
    y_true, y_proba = _informative_case(seed, n=60)

    old = _old_compute_youden_threshold(y_true, y_proba)
    new = compute_youden_threshold(y_true, y_proba)

    assert np.isfinite(old), "this case should exercise the ordinary path"
    assert new == old


def test_threshold_is_a_probability_that_was_actually_observed():
    y_true, y_proba = _informative_case(3, n=40)

    threshold = compute_youden_threshold(y_true, y_proba)

    assert threshold in set(y_proba.tolist())


# =========================================================== import safety

def test_ml_core_imports_cleanly_from_a_foreign_cwd(foreign_cwd, tmp_path):
    """No dataset, no Optuna, no SHAP, no plotting, no warnings-filter changes."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "\n".join([
            "import sys, warnings",
            "import sklearn.metrics",  # unavoidable dependency, registers its own filters
            "baseline = warnings.filters[:]",
            "import ml_core",
            "assert warnings.filters == baseline, 'ml_core mutated warnings filters'",
            "for mod in ('optuna', 'shap', 'matplotlib', 'xgboost', 'streamlit'):",
            "    assert mod not in sys.modules, mod",
            "assert callable(ml_core.compute_youden_threshold)",
            "print('clean')",
        ]),
        encoding="utf-8",
    )
    import os

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, str(probe)], cwd=foreign_cwd,
        capture_output=True, text=True, timeout=300, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_ml_core_writes_nothing_when_imported(tmp_path):
    before = set(tmp_path.rglob("*"))

    import ml_core.bootstrap
    import ml_core.evaluation
    import ml_core.thresholds  # noqa: F401

    assert set(tmp_path.rglob("*")) == before


# ============================================== single definition ownership

@pytest.mark.parametrize("pipeline", MAINTAINED_PIPELINES)
@pytest.mark.parametrize("function", SHARED_FUNCTIONS)
def test_maintained_pipeline_imports_rather_than_redefines(pipeline, function):
    source = (REPO_ROOT / pipeline).read_text(encoding="utf-8")

    assert f"def {function}(" not in source, f"{pipeline} still defines {function}"
    assert "from ml_core import" in source or "import ml_core" in source


@pytest.mark.parametrize("function", SHARED_FUNCTIONS)
def test_exactly_one_maintained_definition_exists(function):
    """One owner per shared behaviour, across the whole tracked repository."""
    listing = subprocess.run(
        # --others --exclude-standard includes files that are new but not
        # ignored, so a freshly added duplicate is caught before it is committed.
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    definers = []
    for name in listing:
        path = REPO_ROOT / name
        if not path.is_file() or path.name.startswith("test_"):
            continue
        if f"def {function}(" in path.read_text(encoding="utf-8", errors="ignore"):
            definers.append(name)

    assert definers == [f"ml_core/{_module_for(function)}.py"], definers


def _module_for(function: str) -> str:
    return {
        "evaluate_predictions": "evaluation",
        "bootstrap_confidence_interval": "bootstrap",
        "compute_youden_threshold": "thresholds",
    }[function]


def test_drift_baselines_are_not_merged():
    """compute_drift_baseline differs by design and must stay per-pipeline.

    Variant A stores {feature: {mean, std, ...}} and variant B stores
    {feature_columns, means, stds, ...}; app.py branches on that difference, so
    unifying them would break /drift-check against the committed baselines.
    """
    lr = (REPO_ROOT / "logisticregression_only.py").read_text(encoding="utf-8")
    xgb = (REPO_ROOT / "boostedtrees_ab.py").read_text(encoding="utf-8")

    assert "def build_logistic_drift_baseline(" in lr
    assert "def build_boosted_drift_baseline(" in xgb
    assert not (REPO_ROOT / "ml_core" / "drift.py").exists()


# ============================================== end-to-end evaluation smoke

def test_shared_utilities_compose_into_a_pipeline_style_evaluation(tmp_path):
    """Deterministic smoke over the exact sequence a training run performs.

    Threshold selection -> prediction -> metrics -> bootstrap intervals, on a
    tiny synthetic fixture. No dataset, no Optuna, no model fitting, no network,
    and the only write goes to tmp_path.
    """
    y_true, y_proba = _informative_case(seed=17, n=120)

    threshold = compute_youden_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = evaluate_predictions(y_true, y_pred, y_proba)
    intervals = bootstrap_confidence_interval(
        y_true, y_proba, threshold, n_bootstrap=30, seed=17
    )

    assert np.isfinite(threshold) and 0.0 <= threshold <= 1.0
    assert set(metrics) == set(ml_core.EVALUATION_KEYS)
    assert set(intervals) == set(ml_core.BOOTSTRAP_METRICS)
    for name, stats in intervals.items():
        assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"], name

    # The bundle shape a pipeline writes must round-trip as JSON unchanged.
    payload = {"threshold": threshold, "test_metrics": metrics,
               "confidence_intervals": intervals}
    out = tmp_path / "metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8")) == payload

    # Re-running is bit-identical.
    assert compute_youden_threshold(y_true, y_proba) == threshold
    assert bootstrap_confidence_interval(
        y_true, y_proba, threshold, n_bootstrap=30, seed=17
    ) == intervals
