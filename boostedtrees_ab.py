# boostedtrees_ab.py
# XGBoost pipeline with Optuna hyperparameter tuning for A/B testing (variant B).
"""
XGBoost model for diabetes prediction - A/B testing variant B.

Uses the same feature set as logistic regression for consistent comparison.

The lifecycle is decomposed into independently callable stages - prepare,
optimize, fit, calibrate, evaluate, explain, persist, attest - so each can be
exercised without running the full 50-trial study. main() is orchestration only.

Deliberately NOT forced into the logistic pipeline's abstractions: this variant
has no scaler, uses a TreeExplainer rather than a LinearExplainer, and emits a
different drift-baseline schema. Those are real differences, not incidental ones.
"""
from functools import partial
from pathlib import Path
from types import MappingProxyType
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from ml_core import feature_contract, pipeline, training
# evaluate_predictions is re-exported deliberately: tests treat the pipeline
# module as the surface for the shared evaluation helpers.
from ml_core import (  # noqa: F401
    bootstrap_confidence_interval,
    compute_youden_threshold,
    evaluate_predictions,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------
# Configuration
# ---------------------------
RANDOM_STATE = 42
# Defaults resolve from the project directory, not the caller's working
# directory. Override either with --data-path / --artifacts-dir.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "cleaned_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
PROVENANCE_PATH = ARTIFACTS_DIR / "boosted_training_manifest.json"

N_BOOTSTRAP = 200

# Production training configuration. These are the values the committed model was
# trained under; tests pin them so a refactor cannot drift them silently.
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
OPTUNA_TRIALS = 50
CV_SPLITS = 5
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5
EVAL_METRIC = "logloss"

# Feature names, order, labels and the target column come from the single
# canonical contract. They used to be maintained separately in each pipeline,
# in app.py and in streamlit_app.py.
SELECTED_FEATURES = feature_contract.feature_list()
TARGET_COLUMN = feature_contract.TARGET_COLUMN
FEATURE_LABELS = dict(feature_contract.FEATURE_LABELS)


def build_boosted_drift_baseline(X_train: pd.DataFrame) -> dict:
    """Column-wise training statistics (variant B drift schema).

    Structurally different from variant A's per-feature mapping, and app.py
    branches on that difference. Unifying the two is a separate migration.
    """
    return {
        "means": X_train.mean().to_dict(),
        "stds": X_train.std().to_dict(),
        "medians": X_train.median().to_dict(),
        "q25": X_train.quantile(0.25).to_dict(),
        "q75": X_train.quantile(0.75).to_dict(),
        "n_train": len(X_train),
        "feature_columns": list(X_train.columns),
    }


# ---------------------------------------------------------------------------
# Per-variant configuration for the shared orchestration helpers. Data only -
# this pipeline still owns its own filenames, model and reporting.
# ---------------------------------------------------------------------------
PIPELINE_SPEC = pipeline.PipelineSpec(
    variant="B",
    model_name="xgboost_boosted_trees",
    scaler=None,
    filenames=MappingProxyType({
        "model_bundle": "boosted_model_bundle.pkl",
        "shap_explainer": "boosted_shap_explainer.pkl",
        "drift_baseline": "boosted_drift_baseline.pkl",
        "metrics": "boosted_metrics.json",
        "provenance": "boosted_training_manifest.json",
    }),
    serving_roles=('model_bundle', 'shap_explainer', 'drift_baseline', 'metrics'),
    optional_roles=(),
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
    cv_splits=CV_SPLITS,
    calibration_method=CALIBRATION_METHOD,
    calibration_cv=CALIBRATION_CV,
    n_bootstrap=N_BOOTSTRAP,
)

# One implementation owner: these bind the shared helpers to this variant's
# configuration. They are partials, not reimplementations - the module
# attributes stay importable for existing callers and tests.
artifact_paths = partial(pipeline.resolve_artifact_paths, spec=PIPELINE_SPEC)
parse_args = partial(
    pipeline.parse_pipeline_args,
    description=__doc__,
    default_data_path=DATA_PATH,
    default_artifacts_dir=ARTIFACTS_DIR,
    default_optuna_trials=OPTUNA_TRIALS,
)
prepare_training_data = partial(
    pipeline.prepare_training_data,
    spec=PIPELINE_SPEC,
    feature_names=SELECTED_FEATURES,
    target_column=TARGET_COLUMN,
    feature_labels=FEATURE_LABELS,
    report_class_distribution=False,
)
select_threshold = pipeline.select_threshold
evaluate_on_test = pipeline.evaluate_on_test
calibrate_model = partial(
    pipeline.calibrate_estimator,
    method=CALIBRATION_METHOD,
    cv=CALIBRATION_CV,
)
emit_provenance = partial(
    pipeline.emit_pipeline_provenance,
    PIPELINE_SPEC,
    feature_names=list(feature_contract.FEATURE_NAMES),
    target_column=TARGET_COLUMN,
)


# ---------------------------------------------------------------- stages


def optimize_hyperparameters(
    splits: training.TrainingSplits,
    *,
    n_trials: int = OPTUNA_TRIALS,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """Run the Optuna study. `n_trials` is injectable so tests need not run 50."""
    print(f"\n🔄 Running Optuna hyperparameter optimization ({n_trials} trials)...")
    X_train, y_train = splits.X_train, splits.y_train

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        }

        model = build_model(params)

        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    print(f"\n✅ Best Optuna params: {study.best_params}")
    print(f"   Best CV ROC-AUC: {study.best_value:.4f}")
    return study


def build_model(params: dict) -> XGBClassifier:
    """The production estimator. No scaler: trees do not need one."""
    return XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        eval_metric=EVAL_METRIC,
        n_jobs=1,
    )


def fit_final_model(splits: training.TrainingSplits, best_params: dict) -> XGBClassifier:
    print("\n🏗️ Training final XGBoost model...")
    final_model = build_model(best_params)
    final_model.fit(splits.X_train, splits.y_train)
    return final_model


def build_shap_explainer(model, splits: training.TrainingSplits):
    """TreeExplainer over the raw feature space - no scaling for trees.

    Separated so training-stage tests need not pay for SHAP; the end-to-end
    smoke opts in explicitly.
    """
    print("\n🔍 Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values_test = explainer.shap_values(splits.X_test)

    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Mean_SHAP": mean_abs_shap
    }).sort_values("Mean_SHAP", ascending=False)

    print("\n   📈 Feature Importance (mean |SHAP|):")
    for _, row in shap_importance.iterrows():
        print(f"      {row['Feature']:25s}: {row['Mean_SHAP']:.4f}")
    return explainer


def report_gain_importance(model) -> pd.DataFrame:
    """Operator-facing importance report. No effect on any artifact."""
    print("\n📈 Feature Importance (XGBoost gain):")
    importance_df = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    for _, row in importance_df.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    return importance_df


def write_training_outputs(
    artifacts_dir: Path,
    *,
    calibrated_model,
    raw_model,
    explainer,
    drift_baseline: dict,
    threshold: float,
    best_params: dict,
    best_cv_auc: float,
    val_metrics: dict,
    test_metrics: dict,
    ci_results: dict,
    brier_before: float,
    brier_after: float,
) -> dict[str, Path]:
    """Persist every artifact. Takes an explicit directory - never a global."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(artifacts_dir)

    bundle = {
        "pipeline": calibrated_model,
        "raw_model": raw_model,
        "threshold": threshold,
        "feature_columns": SELECTED_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "model_name": "xgboost_boosted_trees",
        "optuna_params": best_params,
        "optuna_best_cv_auc": best_cv_auc,
        "confidence_intervals": ci_results,
        "calibration": {
            "method": "platt_scaling",
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
    }
    joblib.dump(bundle, paths["model_bundle"])
    print(f"\n💾 Model bundle saved: {paths['model_bundle']}")

    joblib.dump(drift_baseline, paths["drift_baseline"])
    print(f"💾 Drift baseline saved: {paths['drift_baseline']}")

    if explainer is not None:
        joblib.dump({
            "explainer": explainer,
            "expected_value": float(explainer.expected_value),
            "feature_names": SELECTED_FEATURES,
        }, paths["shap_explainer"])
        print(f"💾 SHAP explainer saved: {paths['shap_explainer']}")

    metrics_output = {
        "threshold": threshold,
        "optuna_params": best_params,
        "optuna_best_cv_auc": best_cv_auc,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confidence_intervals": ci_results,
        "calibration": {
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
    }
    training.write_json_atomic(metrics_output, paths["metrics"])
    print(f"💾 Metrics saved: {paths['metrics']}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Orchestration: load -> optimize -> train -> evaluate -> explain -> persist -> attest."""
    args = parse_args(argv)

    print("=" * 60)
    print("DIABETES PREDICTION - XGBoost Pipeline (A/B Variant B)")
    print("Optuna Hyperparameter Tuning + Youden's J Threshold")
    print("=" * 60)

    splits = prepare_training_data(args.data_path)
    study = optimize_hyperparameters(splits, n_trials=args.optuna_trials)
    best_params = study.best_params

    final_model = fit_final_model(splits, best_params)

    threshold, val_metrics = select_threshold(final_model, splits)
    test_proba, _test_pred, _ = evaluate_on_test(final_model, splits, threshold)

    (calibrated_model, test_proba_final, _test_pred_final,
     test_metrics, brier_before, brier_after) = calibrate_model(
        final_model, splits, threshold, test_proba
    )

    print(f"\n📊 Computing {N_BOOTSTRAP}-iteration bootstrap confidence intervals...")
    ci_results = bootstrap_confidence_interval(
        splits.y_test.values, test_proba_final, threshold
    )
    print("   95% Confidence Intervals:")
    for metric, vals in ci_results.items():
        print(f"      {metric:12s}: {vals['mean']:.4f}  [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]")

    explainer = build_shap_explainer(final_model, splits)
    report_gain_importance(final_model)

    drift_baseline = build_boosted_drift_baseline(splits.X_train)

    paths = write_training_outputs(
        args.artifacts_dir,
        calibrated_model=calibrated_model,
        raw_model=final_model,
        explainer=explainer,
        drift_baseline=drift_baseline,
        threshold=threshold,
        best_params=best_params,
        best_cv_auc=study.best_value,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        ci_results=ci_results,
        brier_before=brier_before,
        brier_after=brier_after,
    )
    emit_provenance(
        paths,
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        best_params=best_params,
        best_cv_auc=study.best_value,
        threshold=threshold,
        n_trials=args.optuna_trials,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        ci_results=ci_results,
    )

    print("\n" + "=" * 60)
    print("✅ XGBoost pipeline complete!")
    print(f"   - Optuna trials: {args.optuna_trials}")
    print(f"   - Best threshold: {threshold:.4f} (Youden's J)")
    print(f"   - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   - Brier score: {brier_after:.4f} (calibrated)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
