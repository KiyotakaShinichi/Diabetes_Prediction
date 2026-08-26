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
from pathlib import Path
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
)
from xgboost import XGBClassifier

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


def artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    """Output filenames for a run. One definition, so a run cannot half-relocate."""
    return {
        "model_bundle": artifacts_dir / "boosted_model_bundle.pkl",
        "shap_explainer": artifacts_dir / "boosted_shap_explainer.pkl",
        "drift_baseline": artifacts_dir / "boosted_drift_baseline.pkl",
        "metrics": artifacts_dir / "boosted_metrics.json",
        "provenance": artifacts_dir / "boosted_training_manifest.json",
    }


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


def select_threshold(model, splits: training.TrainingSplits) -> tuple[float, dict]:
    """Youden's J on the VALIDATION set, so the test set stays untouched."""
    print("\n🎯 Computing optimal threshold using Youden's J on validation set...")
    val_proba = training.positive_class_proba(model, splits.X_val)
    best_threshold = compute_youden_threshold(splits.y_val.values, val_proba)
    print(f"   Best threshold (Youden's J): {best_threshold:.4f}")

    _, val_metrics = training.evaluate_at_threshold(splits.y_val.values, val_proba, best_threshold)
    print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"   Validation F1: {val_metrics['f1']:.4f}")
    return best_threshold, val_metrics


def evaluate_on_test(
    model, splits: training.TrainingSplits, threshold: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    print("\n🔍 Evaluating on held-out TEST set...")
    test_proba = training.positive_class_proba(model, splits.X_test)
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


def calibrate_model(
    model,
    splits: training.TrainingSplits,
    threshold: float,
    uncalibrated_test_proba: np.ndarray,
) -> tuple[CalibratedClassifierCV, np.ndarray, np.ndarray, dict, float, float]:
    """Platt scaling over train+validation, then re-evaluate on the test set."""
    print("\n🎯 Calibrating probabilities (Platt scaling on validation set)...")
    calibrated_model = CalibratedClassifierCV(
        model,
        cv=CALIBRATION_CV,
        method=CALIBRATION_METHOD,
    )
    calibrated_model.fit(
        pd.concat([splits.X_train, splits.X_val]),
        pd.concat([splits.y_train, splits.y_val]),
    )

    cal_proba = training.positive_class_proba(calibrated_model, splits.X_test)
    brier_before = brier_score_loss(splits.y_test, uncalibrated_test_proba)
    brier_after = brier_score_loss(splits.y_test, cal_proba)
    print(f"   Brier score (before calibration): {brier_before:.4f}")
    print(f"   Brier score (after calibration):  {brier_after:.4f}")

    test_pred_final, test_metrics = training.evaluate_at_threshold(
        splits.y_test.values, cal_proba, threshold
    )
    print(f"   Calibrated ROC-AUC: {test_metrics['roc_auc']:.4f}")
    return calibrated_model, cal_proba, test_pred_final, test_metrics, brier_before, brier_after


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
        variant="B",
        model_name="xgboost_boosted_trees",
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
            "scaler": None,
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
