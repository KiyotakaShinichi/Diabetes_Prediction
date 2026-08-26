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
from pathlib import Path
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
)

from ml_core import feature_contract, provenance, training
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


def artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    """Output filenames for a run. One definition, so a run cannot half-relocate."""
    return {
        "model_bundle": artifacts_dir / "model_bundle.pkl",
        "shap_explainer": artifacts_dir / "shap_explainer.pkl",
        "drift_baseline": artifacts_dir / "drift_baseline.pkl",
        "metrics": artifacts_dir / "metrics.json",
        "test_predictions": artifacts_dir / "test_predictions.csv",
        "provenance": artifacts_dir / "training_manifest.json",
    }


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


# ---------------------------------------------------------------- stages

def prepare_training_data(data_path: Path, *, verbose: bool = True) -> training.TrainingSplits:
    """Load, validate against the feature contract, and split."""
    if verbose:
        print(f"\n📥 Loading dataset from {data_path}...")
    frame = training.load_training_dataset(data_path, SELECTED_FEATURES, TARGET_COLUMN)
    if verbose:
        print(f"✅ Data loaded: {frame.shape[0]:,} rows, {frame.shape[1]} columns")

    X, y = training.select_features(frame, SELECTED_FEATURES, TARGET_COLUMN)

    if verbose:
        print(f"\n📊 Selected Features ({len(SELECTED_FEATURES)}):")
        for feat in SELECTED_FEATURES:
            print(f"   - {feat}: {FEATURE_LABELS.get(feat, feat)}")
        print(f"\n🎯 Target: {TARGET_COLUMN}")
        print(f"   Class distribution: {dict(y.value_counts())}")
        print("\n✂️ Splitting data: 60% train / 20% validation / 20% test (stratified)...")

    splits = training.split_training_data(
        X, y, test_size=TEST_SIZE, val_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    if verbose:
        for part, size in splits.sizes.items():
            print(f"   {part.capitalize()}: {size:,} samples")
    return splits


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


def select_threshold(
    pipeline: Pipeline, splits: training.TrainingSplits
) -> tuple[float, dict]:
    """Youden's J on the VALIDATION set, so the test set stays untouched."""
    print("\n🎯 Computing optimal threshold using Youden's J on validation set...")
    val_proba = training.positive_class_proba(pipeline, splits.X_val)
    best_threshold = compute_youden_threshold(splits.y_val.values, val_proba)
    print(f"   Best threshold (Youden's J): {best_threshold:.4f}")

    _, val_metrics = training.evaluate_at_threshold(splits.y_val.values, val_proba, best_threshold)
    print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"   Validation F1: {val_metrics['f1']:.4f}")
    return best_threshold, val_metrics


def evaluate_on_test(
    pipeline: Pipeline, splits: training.TrainingSplits, threshold: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    print("\n🔍 Evaluating on held-out TEST set...")
    test_proba = training.positive_class_proba(pipeline, splits.X_test)
    test_pred, test_metrics = training.evaluate_at_threshold(
        splits.y_test.values, test_proba, threshold
    )

    print("\n📊 TEST SET METRICS:")
    for label, key in (("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc"),
                       ("Cohen's Kappa", "cohen_kappa"), ("MCC", "mcc")):
        print(f"   {label + ':':15s}{test_metrics[key]:.4f}")
    cm = test_metrics["confusion_matrix"]
    print("\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:5d}  FP={cm[0][1]:5d}]")
    print(f"    [FN={cm[1][0]:5d}  TP={cm[1][1]:5d}]]")
    print("\n   Classification Report:")
    print(classification_report(
        splits.y_test, test_pred, digits=4, target_names=["No Diabetes", "Diabetes"]
    ))
    return test_proba, test_pred, test_metrics


def calibrate_pipeline(
    pipeline: Pipeline,
    splits: training.TrainingSplits,
    threshold: float,
    uncalibrated_test_proba: np.ndarray,
) -> tuple[CalibratedClassifierCV, np.ndarray, np.ndarray, dict, float, float]:
    """Platt scaling over train+validation, then re-evaluate on the test set."""
    print("\n🎯 Calibrating probabilities (Platt scaling on validation set)...")
    calibrated_pipeline = CalibratedClassifierCV(
        pipeline,
        cv=CALIBRATION_CV,
        method=CALIBRATION_METHOD,
    )
    calibrated_pipeline.fit(
        pd.concat([splits.X_train, splits.X_val]),
        pd.concat([splits.y_train, splits.y_val]),
    )

    cal_proba = training.positive_class_proba(calibrated_pipeline, splits.X_test)
    brier_before = brier_score_loss(splits.y_test, uncalibrated_test_proba)
    brier_after = brier_score_loss(splits.y_test, cal_proba)
    print(f"   Brier score (before calibration): {brier_before:.4f}")
    print(f"   Brier score (after calibration):  {brier_after:.4f}")

    test_pred_final, test_metrics = training.evaluate_at_threshold(
        splits.y_test.values, cal_proba, threshold
    )
    print(f"   Calibrated ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Calibrated F1:      {test_metrics['f1']:.4f}")
    return calibrated_pipeline, cal_proba, test_pred_final, test_metrics, brier_before, brier_after


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


def emit_provenance(
    paths: dict[str, Path],
    *,
    data_path: Path,
    artifacts_dir: Path,
    best_params: dict,
    best_cv_auc: float,
    threshold: float,
    n_trials: int,
    val_metrics: dict,
    test_metrics: dict,
    ci_results: dict,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Written LAST, after every artifact above is on disk, so its hashes attest
    to completed outputs. If anything above failed, no manifest is produced."""
    written = provenance.emit_training_manifest(
        project_root=project_root,
        output_path=paths["provenance"],
        variant="A",
        model_name="logistic_regression",
        dataset_path=data_path,
        target_column=TARGET_COLUMN,
        # Straight from the canonical contract, so the schema hash can never
        # describe a feature list the models were not trained on.
        feature_names=list(feature_contract.FEATURE_NAMES),
        training={
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "validation_size_of_train": VALIDATION_SIZE,
            "stratified": True,
            "scaler": "StandardScaler",
            "optuna_sampler": "TPESampler",
            "optuna_sampler_seed": RANDOM_STATE,
            "optuna_n_trials": n_trials,
            "optuna_direction": "maximize",
            "optuna_best_params": best_params,
            "optuna_best_cv_auc": best_cv_auc,
            "cv_splits": CV_SPLITS,
            "calibration_method": CALIBRATION_METHOD,
            "calibration_cv": CALIBRATION_CV,
            "threshold_method": "youden_j",
            "selected_threshold": threshold,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_alpha": 0.05,
            "artifacts_dir": provenance.relative_path(artifacts_dir, project_root),
        },
        evaluation={
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "confidence_intervals": ci_results,
        },
        artifact_specs=[
            ("model_bundle", paths["model_bundle"], True),
            ("shap_explainer", paths["shap_explainer"], True),
            ("drift_baseline", paths["drift_baseline"], True),
            ("metrics", paths["metrics"], True),
            ("test_predictions", paths["test_predictions"], False),
        ],
        source_files=[] if project_root != PROJECT_ROOT else [
            Path(__file__).resolve(),
            PROJECT_ROOT / "ml_core" / "evaluation.py",
            PROJECT_ROOT / "ml_core" / "bootstrap.py",
            PROJECT_ROOT / "ml_core" / "thresholds.py",
            PROJECT_ROOT / "ml_core" / "training.py",
            PROJECT_ROOT / "ml_core" / "feature_contract.py",
            PROJECT_ROOT / "ml_core" / "provenance.py",
        ],
        lockfile=(PROJECT_ROOT / "requirements.lock"
                  if project_root == PROJECT_ROOT else None),
    )
    print(f"💾 Provenance manifest saved: {written}")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for the training run. --help exits before any data is read."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, metavar="CSV",
                        help="Training dataset.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR, metavar="DIR",
                        help="Directory the model bundle and metrics are written to.")
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS, metavar="N",
                        help="Hyperparameter search budget.")
    return parser.parse_args(argv)


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
