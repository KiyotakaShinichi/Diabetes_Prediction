"""Deterministic end-to-end training smokes.

Each runs a real fit through the full lifecycle - validate, split, optimize,
fit, calibrate, threshold, evaluate, bootstrap, persist, attest - on a tiny
synthetic dataset, then verifies the outputs with the provenance verifier.

WHAT THIS PROVES: the training plumbing works end to end from scratch.

WHAT THIS DOES NOT PROVE: anything about model quality. These fit a few hundred
synthetic rows with a deliberately tiny search budget. The metrics they produce
are not evidence about the committed production models and are never compared
against them.

Everything is written to tmp_path. Nothing here can touch model_artifacts/.
"""
import json

import joblib
import numpy as np
import pytest

import boostedtrees_ab as xgb_pipeline
import logisticregression_only as lr_pipeline
from ml_core import bootstrap_confidence_interval, provenance
from ml_core.feature_contract import FEATURE_NAMES, TARGET_COLUMN
from tests.test_training_primitives import make_dataset

# Small enough to stay in seconds; large enough for a stratified 3-way split
# plus 5-fold calibration to have data in every fold.
SMOKE_ROWS = 240
SMOKE_BOOTSTRAP = 15


@pytest.fixture
def workspace(tmp_path):
    """A self-contained project root: dataset in, artifacts out."""
    dataset = tmp_path / "mini.csv"
    make_dataset(rows=SMOKE_ROWS, seed=23).to_csv(dataset, index=False)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return dataset, artifacts


def _assert_manifest_is_sound(manifest_path, artifacts_dir, project_root):
    """Shared verification of a freshly emitted training manifest."""
    problems = provenance.verify_manifest_file(manifest_path, project_root)
    assert problems == [], problems

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["provenance_type"] == provenance.TRAINING_RUN
    assert manifest["schema_version"] == provenance.SCHEMA_VERSION

    # Feature section derives from the canonical contract and recomputes.
    features = manifest["features"]
    assert features["feature_names"] == list(FEATURE_NAMES)
    assert features["feature_schema_sha256"] == provenance.sha256_canonical_json(
        {"features": list(FEATURE_NAMES), "target": TARGET_COLUMN}
    )

    # Dirty/commit state is reported, never asserted to be clean.
    git = manifest["run"]["git"]
    assert set(git) == {"commit_sha", "dirty", "branch"}
    assert git["dirty"] in (True, False, None)

    # No absolute paths, no secrets.
    rendered = manifest_path.read_text(encoding="utf-8")
    assert str(project_root) not in rendered
    assert "ADMIN_PASSWORD" not in rendered
    assert "postgresql://" not in rendered
    for entry in manifest["artifacts"]:
        assert not provenance._is_absolute_like(entry["path"])
        assert (project_root / entry["path"]).is_file()

    threshold = manifest["training"]["selected_threshold"]
    assert np.isfinite(threshold)
    assert 0.0 <= threshold <= 1.0
    return manifest


# ================================================== logistic end-to-end

def test_logistic_training_runs_end_to_end(workspace):
    dataset, artifacts = workspace
    project_root = artifacts.parent

    splits = lr_pipeline.prepare_training_data(dataset, verbose=False)
    study = lr_pipeline.optimize_hyperparameters(splits, n_trials=2, show_progress_bar=False)
    best_params = study.best_params

    pipeline = lr_pipeline.fit_final_pipeline(splits, best_params)
    fold_accuracies = lr_pipeline.cross_validate_folds(splits, best_params, pipeline)
    threshold, val_metrics = lr_pipeline.select_threshold(pipeline, splits)
    test_proba, _, _ = lr_pipeline.evaluate_on_test(pipeline, splits, threshold)
    (calibrated, proba_final, pred_final, test_metrics,
     brier_before, brier_after) = lr_pipeline.calibrate_pipeline(
        pipeline, splits, threshold, test_proba
    )
    ci_results = bootstrap_confidence_interval(
        splits.y_test.values, proba_final, threshold, n_bootstrap=SMOKE_BOOTSTRAP
    )
    explainer = lr_pipeline.build_shap_explainer(pipeline, splits)
    drift_baseline = lr_pipeline.build_logistic_drift_baseline(splits.X_train)

    paths = lr_pipeline.write_training_outputs(
        artifacts,
        calibrated_pipeline=calibrated, raw_pipeline=pipeline, explainer=explainer,
        drift_baseline=drift_baseline, splits=splits, threshold=threshold,
        best_params=best_params, best_cv_auc=study.best_value,
        fold_accuracies=fold_accuracies, val_metrics=val_metrics,
        test_metrics=test_metrics, ci_results=ci_results,
        brier_before=brier_before, brier_after=brier_after,
        test_proba_final=proba_final, test_pred_final=pred_final,
    )
    manifest_path = lr_pipeline.emit_provenance(
        paths, data_path=dataset, artifacts_dir=artifacts,
        best_params=best_params, best_cv_auc=study.best_value, threshold=threshold,
        n_trials=2, val_metrics=val_metrics, test_metrics=test_metrics,
        ci_results=ci_results, project_root=project_root,
    )

    # Every declared output exists.
    for name in ("model_bundle", "shap_explainer", "drift_baseline",
                 "metrics", "test_predictions", "provenance"):
        assert paths[name].is_file(), name

    manifest = _assert_manifest_is_sound(manifest_path, artifacts, project_root)
    assert manifest["run"]["variant"] == "A"
    assert manifest["training"]["scaler"] == "StandardScaler"
    assert manifest["training"]["optuna_n_trials"] == 2

    # The bundle is loadable and carries the canonical contract.
    bundle = joblib.load(paths["model_bundle"])
    assert bundle["model_name"] == "logistic_regression"
    assert bundle["feature_columns"] == list(FEATURE_NAMES)
    assert np.isfinite(bundle["threshold"])

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert json.loads(json.dumps(metrics)) == metrics
    assert np.isfinite(metrics["test_metrics"]["roc_auc"])
    assert np.all(np.isfinite(proba_final))


def test_logistic_smoke_writes_nothing_into_the_repository(workspace):
    dataset, artifacts = workspace
    from conftest import ARTIFACTS_DIR

    before = {p.name for p in ARTIFACTS_DIR.iterdir()}
    splits = lr_pipeline.prepare_training_data(dataset, verbose=False)
    pipeline = lr_pipeline.fit_final_pipeline(splits, {"C": 1.0, "solver": "lbfgs"})
    lr_pipeline.write_training_outputs(
        artifacts, calibrated_pipeline=pipeline, raw_pipeline=pipeline, explainer=None,
        drift_baseline={}, splits=splits, threshold=0.5, best_params={}, best_cv_auc=0.0,
        fold_accuracies=[], val_metrics={}, test_metrics={}, ci_results={},
        brier_before=0.0, brier_after=0.0,
        test_proba_final=np.zeros(len(splits.X_test)),
        test_pred_final=np.zeros(len(splits.X_test), dtype=int),
    )

    assert {p.name for p in ARTIFACTS_DIR.iterdir()} == before


# =================================================== boosted end-to-end

def test_boosted_training_runs_end_to_end(workspace):
    """Fits a real XGBoost model. SHAP is covered separately to keep this fast."""
    dataset, artifacts = workspace
    project_root = artifacts.parent

    splits = xgb_pipeline.prepare_training_data(dataset, verbose=False)
    best_params = {"n_estimators": 15, "max_depth": 2, "learning_rate": 0.3}

    model = xgb_pipeline.fit_final_model(splits, best_params)
    threshold, val_metrics = xgb_pipeline.select_threshold(model, splits)
    test_proba, _, _ = xgb_pipeline.evaluate_on_test(model, splits, threshold)
    (calibrated, proba_final, _pred_final, test_metrics,
     brier_before, brier_after) = xgb_pipeline.calibrate_model(
        model, splits, threshold, test_proba
    )
    ci_results = bootstrap_confidence_interval(
        splits.y_test.values, proba_final, threshold, n_bootstrap=SMOKE_BOOTSTRAP
    )
    drift_baseline = xgb_pipeline.build_boosted_drift_baseline(splits.X_train)

    paths = xgb_pipeline.write_training_outputs(
        artifacts,
        calibrated_model=calibrated, raw_model=model, explainer=None,
        drift_baseline=drift_baseline, threshold=threshold, best_params=best_params,
        best_cv_auc=0.0, val_metrics=val_metrics, test_metrics=test_metrics,
        ci_results=ci_results, brier_before=brier_before, brier_after=brier_after,
    )
    # The SHAP explainer is optional above, so it must not be attested here.
    manifest_path = provenance.emit_training_manifest(
        project_root=project_root, output_path=paths["provenance"], variant="B",
        model_name="xgboost_boosted_trees", dataset_path=dataset,
        target_column=TARGET_COLUMN, feature_names=list(FEATURE_NAMES),
        training={"selected_threshold": threshold, "optuna_n_trials": 0},
        evaluation={"test_metrics": test_metrics},
        artifact_specs=[
            ("model_bundle", paths["model_bundle"], True),
            ("drift_baseline", paths["drift_baseline"], True),
            ("metrics", paths["metrics"], True),
        ],
        source_files=[], lockfile=None,
    )

    for name in ("model_bundle", "drift_baseline", "metrics"):
        assert paths[name].is_file(), name

    _assert_manifest_is_sound(manifest_path, artifacts, project_root)

    bundle = joblib.load(paths["model_bundle"])
    assert bundle["model_name"] == "xgboost_boosted_trees"
    assert bundle["feature_columns"] == list(FEATURE_NAMES)
    assert np.all(np.isfinite(proba_final))
    assert np.isfinite(threshold) and 0.0 <= threshold <= 1.0


def test_boosted_shap_stage_is_covered_separately(workspace):
    """The smoke skips SHAP for speed, so the constructor is proven here."""
    dataset, _ = workspace
    splits = xgb_pipeline.prepare_training_data(dataset, verbose=False)
    model = xgb_pipeline.fit_final_model(
        splits, {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.3}
    )

    explainer = xgb_pipeline.build_shap_explainer(model, splits)

    assert np.isfinite(float(explainer.expected_value))


# ================================================ manifest ordering

def test_no_manifest_survives_a_failed_artifact_write(workspace, monkeypatch):
    """The manifest is written last; a failure earlier must leave none."""
    dataset, artifacts = workspace
    splits = lr_pipeline.prepare_training_data(dataset, verbose=False)
    paths = lr_pipeline.artifact_paths(artifacts)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(lr_pipeline.joblib, "dump", boom)

    with pytest.raises(OSError):
        lr_pipeline.write_training_outputs(
            artifacts, calibrated_pipeline=None, raw_pipeline=None, explainer=None,
            drift_baseline={}, splits=splits, threshold=0.5, best_params={},
            best_cv_auc=0.0, fold_accuracies=[], val_metrics={}, test_metrics={},
            ci_results={}, brier_before=0.0, brier_after=0.0,
            test_proba_final=np.zeros(len(splits.X_test)),
            test_pred_final=np.zeros(len(splits.X_test), dtype=int),
        )

    assert not paths["provenance"].exists()


def test_manifest_hashes_change_when_an_artifact_changes(workspace):
    dataset, artifacts = workspace
    splits = lr_pipeline.prepare_training_data(dataset, verbose=False)
    pipeline = lr_pipeline.fit_final_pipeline(splits, {"C": 1.0, "solver": "lbfgs"})
    paths = lr_pipeline.write_training_outputs(
        artifacts, calibrated_pipeline=pipeline, raw_pipeline=pipeline, explainer=None,
        drift_baseline={}, splits=splits, threshold=0.5, best_params={}, best_cv_auc=0.0,
        fold_accuracies=[], val_metrics={}, test_metrics={}, ci_results={},
        brier_before=0.0, brier_after=0.0,
        test_proba_final=np.zeros(len(splits.X_test)),
        test_pred_final=np.zeros(len(splits.X_test), dtype=int),
    )
    manifest_path = lr_pipeline.emit_provenance(
        paths, data_path=dataset, artifacts_dir=artifacts, best_params={},
        best_cv_auc=0.0, threshold=0.5, n_trials=0,
        val_metrics={}, test_metrics={}, ci_results={},
        project_root=artifacts.parent,
    )
    assert provenance.verify_manifest_file(manifest_path, artifacts.parent) == []

    paths["metrics"].write_text('{"tampered": true}', encoding="utf-8")

    problems = provenance.verify_manifest_file(manifest_path, artifacts.parent)
    assert any("metrics.json" in problem for problem in problems)
