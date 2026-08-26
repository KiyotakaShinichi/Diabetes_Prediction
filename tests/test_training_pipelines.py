"""Decomposed training stages, and the production configuration they must not drift.

Two jobs:

* prove the stages are independently callable on a tiny dataset, writing only to
  tmp_path and never touching model_artifacts/;
* pin the production training configuration, so a refactor cannot silently change
  what the next real training run would do.

The configuration values below are asserted, not aspirational: they are the
values the committed models were trained under. Newly fitted mini-models are
NOT compared against committed production metrics - that would be meaningless
on 200 synthetic rows.
"""
import json

import numpy as np
import pytest

import boostedtrees_ab as xgb_pipeline
import logisticregression_only as lr_pipeline
from ml_core import training
from ml_core.feature_contract import FEATURE_NAMES, TARGET_COLUMN
from tests.test_training_primitives import make_dataset

PIPELINES = [
    pytest.param(lr_pipeline, id="logistic_regression"),
    pytest.param(xgb_pipeline, id="boosted_trees"),
]

# Known-good hyperparameters, so stage tests never run a search.
LR_PARAMS = {"C": 1.0, "solver": "lbfgs"}
XGB_PARAMS = {"n_estimators": 12, "max_depth": 2, "learning_rate": 0.3}


@pytest.fixture
def dataset_csv(tmp_path):
    path = tmp_path / "mini.csv"
    make_dataset(rows=200, seed=11).to_csv(path, index=False)
    return path


@pytest.fixture
def splits(dataset_csv):
    return lr_pipeline.prepare_training_data(dataset_csv, verbose=False)


# =================================================== production configuration

def test_logistic_production_configuration_is_unchanged():
    assert lr_pipeline.RANDOM_STATE == 42
    assert lr_pipeline.TEST_SIZE == 0.2
    assert lr_pipeline.VALIDATION_SIZE == 0.25
    assert lr_pipeline.OPTUNA_TRIALS == 100
    assert lr_pipeline.CV_SPLITS == 5
    assert lr_pipeline.CALIBRATION_METHOD == "sigmoid"
    assert lr_pipeline.CALIBRATION_CV == 5
    assert lr_pipeline.MAX_ITER == 2000
    assert lr_pipeline.N_BOOTSTRAP == 200
    assert lr_pipeline.SHAP_BACKGROUND_SAMPLES == 500


def test_boosted_production_configuration_is_unchanged():
    assert xgb_pipeline.RANDOM_STATE == 42
    assert xgb_pipeline.TEST_SIZE == 0.2
    assert xgb_pipeline.VALIDATION_SIZE == 0.25
    assert xgb_pipeline.OPTUNA_TRIALS == 50
    assert xgb_pipeline.CV_SPLITS == 5
    assert xgb_pipeline.CALIBRATION_METHOD == "sigmoid"
    assert xgb_pipeline.CALIBRATION_CV == 5
    assert xgb_pipeline.EVAL_METRIC == "logloss"
    assert xgb_pipeline.N_BOOTSTRAP == 200


def test_logistic_search_space_is_unchanged():
    source = (lr_pipeline.PROJECT_ROOT / "logisticregression_only.py").read_text(encoding="utf-8")

    assert 'trial.suggest_float("C", 0.01, 10, log=True)' in source
    assert 'trial.suggest_categorical("solver", ["lbfgs", "liblinear"])' in source
    assert 'penalty="l2"' in source


def test_boosted_search_space_is_unchanged():
    source = (xgb_pipeline.PROJECT_ROOT / "boostedtrees_ab.py").read_text(encoding="utf-8")

    for fragment in (
        'trial.suggest_int("n_estimators", 100, 400)',
        'trial.suggest_int("max_depth", 3, 7)',
        'trial.suggest_float("learning_rate", 0.01, 0.3, log=True)',
        'trial.suggest_float("subsample", 0.6, 1.0)',
        'trial.suggest_float("colsample_bytree", 0.6, 1.0)',
        'trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)',
        'trial.suggest_float("reg_alpha", 0.01, 10.0, log=True)',
    ):
        assert fragment in source, fragment


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [
        (lr_pipeline, {"model_bundle.pkl", "shap_explainer.pkl", "drift_baseline.pkl",
                       "metrics.json", "test_predictions.csv", "training_manifest.json"}),
        (xgb_pipeline, {"boosted_model_bundle.pkl", "boosted_shap_explainer.pkl",
                        "boosted_drift_baseline.pkl", "boosted_metrics.json",
                        "boosted_training_manifest.json"}),
    ],
    ids=["logistic_regression", "boosted_trees"],
)
def test_artifact_filenames_are_unchanged(pipeline, expected, tmp_path):
    paths = pipeline.artifact_paths(tmp_path)

    assert {p.name for p in paths.values()} == expected
    assert all(p.parent == tmp_path for p in paths.values())


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipeline_uses_the_canonical_contract(pipeline):
    assert tuple(pipeline.SELECTED_FEATURES) == FEATURE_NAMES
    assert pipeline.TARGET_COLUMN == TARGET_COLUMN


# ============================================================ prepare stage

def test_prepare_validates_against_the_contract(tmp_path):
    bad = tmp_path / "bad.csv"
    make_dataset(40).drop(columns=["BMI"]).to_csv(bad, index=False)

    with pytest.raises(training.DatasetValidationError, match="BMI"):
        lr_pipeline.prepare_training_data(bad, verbose=False)


def test_prepare_returns_canonically_ordered_splits(splits):
    assert splits.feature_names == FEATURE_NAMES
    assert list(splits.X_train.columns) == list(FEATURE_NAMES)
    assert sum(splits.sizes.values()) == 200


def test_prepare_is_deterministic(dataset_csv):
    first = lr_pipeline.prepare_training_data(dataset_csv, verbose=False)
    second = lr_pipeline.prepare_training_data(dataset_csv, verbose=False)

    assert list(first.X_test.index) == list(second.X_test.index)


# ======================================================== logistic stages

def test_logistic_fit_evaluate_and_calibrate(splits):
    pipeline = lr_pipeline.fit_final_pipeline(splits, LR_PARAMS)
    threshold, val_metrics = lr_pipeline.select_threshold(pipeline, splits)
    test_proba, _, _ = lr_pipeline.evaluate_on_test(pipeline, splits, threshold)

    calibrated, proba_final, pred_final, metrics, before, after = lr_pipeline.calibrate_pipeline(
        pipeline, splits, threshold, test_proba
    )

    assert np.isfinite(threshold) and 0.0 <= threshold <= 1.0
    assert set(val_metrics) >= {"roc_auc", "f1"}
    assert np.all(np.isfinite(proba_final))
    assert set(np.unique(pred_final)) <= {0, 1}
    assert np.isfinite(before) and np.isfinite(after)
    assert hasattr(calibrated, "predict_proba")
    assert json.loads(json.dumps(metrics)) == metrics


def test_logistic_build_pipeline_uses_the_production_estimator():
    pipeline = lr_pipeline.build_pipeline(LR_PARAMS)

    assert list(dict(pipeline.steps)) == ["scaler", "model"]
    model = pipeline.named_steps["model"]
    assert model.penalty == "l2"
    assert model.max_iter == lr_pipeline.MAX_ITER
    assert model.random_state == lr_pipeline.RANDOM_STATE


def test_logistic_cross_validation_reports_one_accuracy_per_fold(splits):
    pipeline = lr_pipeline.fit_final_pipeline(splits, LR_PARAMS)

    folds = lr_pipeline.cross_validate_folds(splits, LR_PARAMS, pipeline)

    assert len(folds) == lr_pipeline.CV_SPLITS
    assert all(0.0 <= value <= 1.0 for value in folds)


def test_logistic_optuna_boundary_accepts_a_tiny_budget(splits):
    """Tests bypass the 100-trial study through the parameter, not a monkeypatch."""
    study = lr_pipeline.optimize_hyperparameters(splits, n_trials=2, show_progress_bar=False)

    assert len(study.trials) == 2
    assert set(study.best_params) == {"C", "solver"}
    assert lr_pipeline.OPTUNA_TRIALS == 100, "production default must not change"


def test_logistic_drift_baseline_keeps_schema_a(splits):
    baseline = lr_pipeline.build_logistic_drift_baseline(splits.X_train)

    assert set(baseline) == set(FEATURE_NAMES)
    assert set(baseline["BMI"]) == {"mean", "std", "min", "max", "q25", "median", "q75"}
    assert "feature_columns" not in baseline


def test_logistic_shap_explainer_is_separately_callable(splits):
    pipeline = lr_pipeline.fit_final_pipeline(splits, LR_PARAMS)

    explainer = lr_pipeline.build_shap_explainer(pipeline, splits)

    assert explainer is not None
    assert np.isfinite(float(explainer.expected_value))


# ========================================================= boosted stages

def test_boosted_fit_evaluate_and_calibrate(splits):
    model = xgb_pipeline.fit_final_model(splits, XGB_PARAMS)
    threshold, _ = xgb_pipeline.select_threshold(model, splits)
    test_proba, _, _ = xgb_pipeline.evaluate_on_test(model, splits, threshold)

    calibrated, proba_final, pred_final, metrics, before, after = xgb_pipeline.calibrate_model(
        model, splits, threshold, test_proba
    )

    assert np.isfinite(threshold) and 0.0 <= threshold <= 1.0
    assert np.all(np.isfinite(proba_final))
    assert set(np.unique(pred_final)) <= {0, 1}
    assert np.isfinite(before) and np.isfinite(after)
    assert hasattr(calibrated, "predict_proba")
    assert json.loads(json.dumps(metrics)) == metrics


def test_boosted_build_model_uses_the_production_estimator():
    model = xgb_pipeline.build_model(XGB_PARAMS)

    assert model.random_state == xgb_pipeline.RANDOM_STATE
    assert model.eval_metric == xgb_pipeline.EVAL_METRIC
    assert model.n_jobs == 1


def test_boosted_optuna_boundary_accepts_a_tiny_budget(splits):
    study = xgb_pipeline.optimize_hyperparameters(splits, n_trials=1, show_progress_bar=False)

    assert len(study.trials) == 1
    assert xgb_pipeline.OPTUNA_TRIALS == 50, "production default must not change"


def test_boosted_drift_baseline_keeps_schema_b(splits):
    baseline = xgb_pipeline.build_boosted_drift_baseline(splits.X_train)

    assert set(baseline) == {"means", "stds", "medians", "q25", "q75",
                             "n_train", "feature_columns"}
    assert baseline["feature_columns"] == list(FEATURE_NAMES)
    assert baseline["n_train"] == len(splits.X_train)


def test_boosted_shap_explainer_is_separately_callable(splits):
    model = xgb_pipeline.fit_final_model(splits, XGB_PARAMS)

    explainer = xgb_pipeline.build_shap_explainer(model, splits)

    assert explainer is not None
    assert np.isfinite(float(explainer.expected_value))


def test_the_two_drift_schemas_remain_different(splits):
    """Unifying them is deferred; app.py branches on the difference."""
    a = lr_pipeline.build_logistic_drift_baseline(splits.X_train)
    b = xgb_pipeline.build_boosted_drift_baseline(splits.X_train)

    assert "feature_columns" not in a
    assert "feature_columns" in b


# ==================================================== writer boundary

def test_writers_never_touch_the_committed_artifacts_directory(tmp_path):
    """Every write function takes an explicit directory."""
    for pipeline in (lr_pipeline, xgb_pipeline):
        paths = pipeline.artifact_paths(tmp_path)
        for path in paths.values():
            assert tmp_path in path.parents
            assert pipeline.ARTIFACTS_DIR not in path.parents


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_main_reads_as_orchestration(pipeline):
    """main() should call stages, not reimplement them."""
    import inspect

    source = inspect.getsource(pipeline.main)

    for stage in ("prepare_training_data", "optimize_hyperparameters",
                  "write_training_outputs", "emit_provenance"):
        assert stage in source, stage
    # No model construction or persistence inline in main().
    assert "joblib.dump" not in source
    assert "CalibratedClassifierCV(" not in source
    assert len(source.splitlines()) < 100
