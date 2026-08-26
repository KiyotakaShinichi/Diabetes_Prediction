# logreg_optuna_youden.py
# Logistic Regression pipeline with Optuna hyperparameter tuning, Youden's J threshold,
# and evaluation metrics for diabetes prediction.
"""
Industry-grade diabetes prediction training script.

Key Components:
  - Optuna hyperparameter optimization (100 trials)
  - Youden's J statistic for optimal threshold selection
  - Proper train/validation/test split (no data leakage)
  - sklearn Pipeline with StandardScaler
  - Comprehensive evaluation metrics

The lifecycle is decomposed into independently callable stages - prepare, optimize,
fit, calibrate, evaluate, explain, persist, attest - so each can be exercised
without running the full 100-trial study. main() is orchestration only.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
)

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
PROVENANCE_PATH = ARTIFACTS_DIR / "training_manifest.json"

N_BOOTSTRAP = 200  # bootstrap iterations for confidence intervals

# Production training configuration. These are the values the committed model was
# trained under; tests pin them so a refactor cannot drift them silently.
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
OPTUNA_TRIALS = 100
CV_SPLITS = 5
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5
MAX_ITER = 2000
SHAP_BACKGROUND_SAMPLES = 500

# Feature names, order, labels and the target column come from the single
# canonical contract. They used to be maintained separately in each pipeline,
# in app.py and in streamlit_app.py.
SELECTED_FEATURES = feature_contract.feature_list()
TARGET_COLUMN = feature_contract.TARGET_COLUMN
FEATURE_LABELS = dict(feature_contract.FEATURE_LABELS)


def build_logistic_drift_baseline(X_train: pd.DataFrame) -> dict:
    """Per-feature training statistics (variant A drift schema).

    Deliberately NOT shared with the boosted pipeline: variant B emits a
    different structure and app.py branches on the difference. Unifying them is
    a separate migration.
    """
    stats = {}
    for col in X_train.columns:
        series = X_train[col].astype(float)
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
        }
    return stats


# ---------------------------------------------------------------------------
# Per-variant configuration for the shared orchestration helpers. Data only -
# this pipeline still owns its own filenames, model and reporting.
# ---------------------------------------------------------------------------
PIPELINE_SPEC = pipeline.PipelineSpec(
    variant="A",
    model_name="logistic_regression",
    scaler="StandardScaler",
    filenames=MappingProxyType({
        "model_bundle": "model_bundle.pkl",
        "shap_explainer": "shap_explainer.pkl",
        "drift_baseline": "drift_baseline.pkl",
        "metrics": "metrics.json",
        "test_predictions": "test_predictions.csv",
        "provenance": "training_manifest.json",
    }),
    serving_roles=('model_bundle', 'shap_explainer', 'drift_baseline', 'metrics'),
    optional_roles=('test_predictions',),
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
    report_class_distribution=True,
)
select_threshold = pipeline.select_threshold
evaluate_on_test = pipeline.evaluate_on_test
calibrate_pipeline = partial(
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
    """Run the Optuna study. `n_trials` is injectable so tests need not run 100."""
    print(f"\n🔄 Running Optuna hyperparameter optimization ({n_trials} trials)...")

    # Scale training data for Optuna CV
    scaler_optuna = StandardScaler()
    X_train_scaled = scaler_optuna.fit_transform(splits.X_train)
    y_train = splits.y_train

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 10, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

        lr = LogisticRegression(
            C=C,
            solver=solver,
            penalty="l2",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        )

        # 5-fold cross-validation scoring ROC-AUC
        scores = cross_val_score(
            lr, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc"
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    print(f"\n✅ Best Optuna params: {study.best_params}")
    print(f"   Best CV ROC-AUC: {study.best_value:.4f}")
    return study


def build_pipeline(best_params: dict) -> Pipeline:
    """The production estimator: StandardScaler + L2 LogisticRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver"],
            penalty="l2",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        ))
    ])


def fit_final_pipeline(splits: training.TrainingSplits, best_params: dict) -> Pipeline:
    print("\n🏗️ Training final pipeline with best parameters...")
    pipeline = build_pipeline(best_params)
    pipeline.fit(splits.X_train, splits.y_train)
    return pipeline


def cross_validate_folds(
    splits: training.TrainingSplits, best_params: dict, pipeline: Pipeline
) -> list[float]:
    """Per-fold accuracy on the training set, reported in metrics.json."""
    print("\n📊 5-Fold Cross-validation on training set:")
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X_train_scaled_final = pipeline.named_steps["scaler"].transform(splits.X_train)

    fold_accuracies = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(splits.X_train, splits.y_train), 1):
        model_clone = LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver"],
            penalty="l2",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        )
        model_clone.fit(X_train_scaled_final[train_idx], splits.y_train.iloc[train_idx])
        preds = model_clone.predict(X_train_scaled_final[val_idx])
        acc = accuracy_score(splits.y_train.iloc[val_idx], preds)
        fold_accuracies.append(acc)
        print(f"   Fold {fold_idx}: {acc:.4f}")

    print(f"   Mean: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    return fold_accuracies


def build_shap_explainer(pipeline: Pipeline, splits: training.TrainingSplits):
    """LinearExplainer over the scaled feature space.

    Separated so training-stage tests need not pay for SHAP; the end-to-end
    smoke opts in explicitly.
    """
    print("\n🔍 Computing SHAP values for model explainability...")
    X_train_scaled_shap = pipeline.named_steps["scaler"].transform(splits.X_train)
    X_test_scaled_shap = pipeline.named_steps["scaler"].transform(splits.X_test)
    inner_model = pipeline.named_steps["model"]

    bg_sample = shap.sample(
        X_train_scaled_shap, min(SHAP_BACKGROUND_SAMPLES, len(X_train_scaled_shap))
    )
    explainer = shap.LinearExplainer(inner_model, bg_sample, feature_names=SELECTED_FEATURES)
    shap_values_test = explainer.shap_values(X_test_scaled_shap)

    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Mean_SHAP": mean_abs_shap
    }).sort_values("Mean_SHAP", ascending=False)

    print("\n   📈 Feature Importance (mean |SHAP|):")
    for _, row in shap_importance.iterrows():
        print(f"      {row['Feature']:25s}: {row['Mean_SHAP']:.4f}")
    return explainer


def report_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Operator-facing coefficient report. No effect on any artifact."""
    print("\n📈 Feature Importance (Logistic Regression Coefficients):")
    model = pipeline.named_steps["model"]
    coef_df = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0])
    }).sort_values("Coefficient", ascending=False)

    print("\n   🚀 Top factors increasing diabetes risk:")
    for _, row in coef_df.head(5).iterrows():
        direction = "↑" if row["Coefficient"] > 0 else "↓"
        print(f"      {direction} {row['Feature']}: coef={row['Coefficient']:.4f}, OR={row['Odds_Ratio']:.3f}")

    print("\n   🧊 Top factors decreasing diabetes risk:")
    for _, row in coef_df.tail(3).iterrows():
        direction = "↑" if row["Coefficient"] > 0 else "↓"
        print(f"      {direction} {row['Feature']}: coef={row['Coefficient']:.4f}, OR={row['Odds_Ratio']:.3f}")
    return coef_df


def write_training_outputs(
    artifacts_dir: Path,
    *,
    calibrated_pipeline,
    raw_pipeline,
    explainer,
    drift_baseline: dict,
    splits: training.TrainingSplits,
    threshold: float,
    best_params: dict,
    best_cv_auc: float,
    fold_accuracies: list[float],
    val_metrics: dict,
    test_metrics: dict,
    ci_results: dict,
    brier_before: float,
    brier_after: float,
    test_proba_final: np.ndarray,
    test_pred_final: np.ndarray,
) -> dict[str, Path]:
    """Persist every artifact. Takes an explicit directory - never a global."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(artifacts_dir)

    bundle = {
        "pipeline": calibrated_pipeline,
        "raw_pipeline": raw_pipeline,  # keep original for SHAP
        "threshold": threshold,
        "feature_columns": SELECTED_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "model_name": "logistic_regression",
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

    if explainer is not None:
        joblib.dump({
            "explainer": explainer,
            "expected_value": float(explainer.expected_value),
            "feature_names": SELECTED_FEATURES,
        }, paths["shap_explainer"])
        print(f"💾 SHAP explainer saved: {paths['shap_explainer']}")

    joblib.dump(drift_baseline, paths["drift_baseline"])
    print(f"💾 Drift baseline saved: {paths['drift_baseline']}")

    metrics_output = {
        "threshold": threshold,
        "optuna_params": best_params,
        "optuna_best_cv_auc": best_cv_auc,
        "cv_fold_accuracies": fold_accuracies,
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

    predictions_df = splits.X_test.copy()
    predictions_df["Actual"] = splits.y_test.values
    predictions_df["Predicted"] = test_pred_final
    predictions_df["Probability"] = test_proba_final
    predictions_df.to_csv(paths["test_predictions"], index=False)
    print(f"💾 Test predictions saved: {paths['test_predictions']}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Orchestration: load -> optimize -> train -> evaluate -> explain -> persist -> attest."""
    args = parse_args(argv)

    print("=" * 60)
    print("DIABETES PREDICTION - Logistic Regression Pipeline")
    print("Optuna Hyperparameter Tuning + Youden's J Threshold")
    print("=" * 60)

    splits = prepare_training_data(args.data_path)
    study = optimize_hyperparameters(splits, n_trials=args.optuna_trials)
    best_params = study.best_params

    pipeline = fit_final_pipeline(splits, best_params)
    fold_accuracies = cross_validate_folds(splits, best_params, pipeline)

    threshold, val_metrics = select_threshold(pipeline, splits)
    test_proba, _test_pred, _ = evaluate_on_test(pipeline, splits, threshold)

    (calibrated_pipeline, test_proba_final, test_pred_final,
     test_metrics, brier_before, brier_after) = calibrate_pipeline(
        pipeline, splits, threshold, test_proba
    )

    print(f"\n📊 Computing {N_BOOTSTRAP}-iteration bootstrap confidence intervals...")
    ci_results = bootstrap_confidence_interval(
        splits.y_test.values, test_proba_final, threshold, n_bootstrap=N_BOOTSTRAP
    )
    print("   95% Confidence Intervals:")
    for metric, vals in ci_results.items():
        print(f"      {metric:12s}: {vals['mean']:.4f}  [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]")

    explainer = build_shap_explainer(pipeline, splits)
    report_coefficients(pipeline)

    print("\n📊 Computing drift detection baseline from training set...")
    drift_baseline = build_logistic_drift_baseline(splits.X_train)

    paths = write_training_outputs(
        args.artifacts_dir,
        calibrated_pipeline=calibrated_pipeline,
        raw_pipeline=pipeline,
        explainer=explainer,
        drift_baseline=drift_baseline,
        splits=splits,
        threshold=threshold,
        best_params=best_params,
        best_cv_auc=study.best_value,
        fold_accuracies=fold_accuracies,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        ci_results=ci_results,
        brier_before=brier_before,
        brier_after=brier_after,
        test_proba_final=test_proba_final,
        test_pred_final=test_pred_final,
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
    print("✅ Logistic Regression pipeline complete!")
    print(f"   - Optuna trials: {args.optuna_trials}")
    print(f"   - Best threshold: {threshold:.4f} (Youden's J)")
    print(f"   - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   - Brier score: {brier_after:.4f} (calibrated)")
    print("   - SHAP explainer: saved")
    print("   - Drift baseline: saved")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
