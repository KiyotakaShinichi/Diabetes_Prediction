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
"""
from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
)

from ml_core import provenance
from ml_core import (
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
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
SHAP_PATH = ARTIFACTS_DIR / "shap_explainer.pkl"
DRIFT_BASELINE_PATH = ARTIFACTS_DIR / "drift_baseline.pkl"
PROVENANCE_PATH = ARTIFACTS_DIR / "training_manifest.json"

N_BOOTSTRAP = 200  # bootstrap iterations for confidence intervals

# Selected features (mapped from original design, IncomeLevel removed as requested)
# Original Name           -> Current Column
# GeneralHealth           -> GenHlth (1-5 scale)
# HasHighBP               -> HighBP (binary)
# BMI                     -> BMI (numeric)
# HasHighChol             -> HighChol (binary)
# AgeCategory             -> Age (1-13)
# HasWalkingDifficulty    -> DiffWalk (binary)
# HadHeartIssues          -> HeartDiseaseorAttack (binary)
# PoorPhysicalHealthDays  -> PhysHlth (0-30)
# EducationLevel          -> Education (1-6)
# IsPhysicallyActive      -> PhysActivity (binary)

SELECTED_FEATURES = [
    "GenHlth",
    "HighBP",
    "BMI",
    "HighChol",
    "Age",
    "DiffWalk",
    "HeartDiseaseorAttack",
    "PhysHlth",
    "Education",
    "PhysActivity",
]
TARGET_COLUMN = "Diabetes_binary"

# Human-readable labels for UI/reporting
FEATURE_LABELS = {
    "GenHlth": "General Health (1=Excellent to 5=Poor)",
    "HighBP": "Has High Blood Pressure",
    "BMI": "Body Mass Index",
    "HighChol": "Has High Cholesterol",
    "Age": "Age Category (1=18-24 to 13=80+)",
    "DiffWalk": "Has Walking Difficulty",
    "HeartDiseaseorAttack": "Has Heart Disease or Had Heart Attack",
    "PhysHlth": "Poor Physical Health Days (last 30 days)",
    "Education": "Education Level (1-6)",
    "PhysActivity": "Is Physically Active",
}


def compute_drift_baseline(X_train: pd.DataFrame) -> dict:
    """
    Compute training-set feature statistics used as baseline for drift detection.
    Stores mean, std, min, max, and quantiles per feature.
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


def main() -> None:
    print("=" * 60)
    print("DIABETES PREDICTION - Logistic Regression Pipeline")
    print("Optuna Hyperparameter Tuning + Youden's J Threshold")
    print("=" * 60)

    # ---------------------------
    # 1) Load & prepare data
    # ---------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    print(f"\n📥 Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Validate columns
    required_columns = set(SELECTED_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\n📊 Selected Features ({len(SELECTED_FEATURES)}):")
    for feat in SELECTED_FEATURES:
        print(f"   - {feat}: {FEATURE_LABELS.get(feat, feat)}")

    print(f"\n🎯 Target: {TARGET_COLUMN}")
    print(f"   Class distribution: {dict(y.value_counts())}")

    # ---------------------------
    # 2) Split data (train/validation/test)
    # ---------------------------
    print("\n✂️ Splitting data: 60% train / 20% validation / 20% test (stratified)...")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE
    )

    print(f"   Train: {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Test: {len(X_test):,} samples")

    # ---------------------------
    # 3) Optuna hyperparameter tuning
    # ---------------------------
    print("\n🔄 Running Optuna hyperparameter optimization (100 trials)...")

    # Scale training data for Optuna CV
    scaler_optuna = StandardScaler()
    X_train_scaled = scaler_optuna.fit_transform(X_train)

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 10, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

        lr = LogisticRegression(
            C=C,
            solver=solver,
            penalty="l2",
            max_iter=2000,
            random_state=RANDOM_STATE
        )

        # 5-fold cross-validation scoring ROC-AUC
        scores = cross_val_score(
            lr, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc"
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best_params = study.best_params
    print(f"\n✅ Best Optuna params: {best_params}")
    print(f"   Best CV ROC-AUC: {study.best_value:.4f}")

    # ---------------------------
    # 4) Train final pipeline
    # ---------------------------
    print("\n🏗️ Training final pipeline with best parameters...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver"],
            penalty="l2",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)

    # ---------------------------
    # 5) Cross-validation diagnostics
    # ---------------------------
    print("\n📊 5-Fold Cross-validation on training set:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train_scaled_final = pipeline.named_steps["scaler"].transform(X_train)

    fold_accuracies = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        model_clone = LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver"],
            penalty="l2",
            max_iter=2000,
            random_state=RANDOM_STATE
        )
        model_clone.fit(X_train_scaled_final[train_idx], y_train.iloc[train_idx])
        preds = model_clone.predict(X_train_scaled_final[val_idx])
        acc = accuracy_score(y_train.iloc[val_idx], preds)
        fold_accuracies.append(acc)
        print(f"   Fold {fold_idx}: {acc:.4f}")

    print(f"   Mean: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    # ---------------------------
    # 6) Youden's J threshold (on VALIDATION set - no leakage)
    # ---------------------------
    print("\n🎯 Computing optimal threshold using Youden's J on validation set...")
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    best_threshold = compute_youden_threshold(y_val.values, val_proba)
    print(f"   Best threshold (Youden's J): {best_threshold:.4f}")

    # Validation metrics at optimal threshold
    val_pred = (val_proba >= best_threshold).astype(int)
    val_metrics = evaluate_predictions(y_val.values, val_pred, val_proba)
    print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"   Validation F1: {val_metrics['f1']:.4f}")

    # ---------------------------
    # 7) Final evaluation on TEST set
    # ---------------------------
    print("\n🔍 Evaluating on held-out TEST set...")
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    test_metrics = evaluate_predictions(y_test.values, test_pred, test_proba)

    print(f"\n📊 TEST SET METRICS:")
    print(f"   Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"   Precision:     {test_metrics['precision']:.4f}")
    print(f"   Recall:        {test_metrics['recall']:.4f}")
    print(f"   F1-score:      {test_metrics['f1']:.4f}")
    print(f"   ROC-AUC:       {test_metrics['roc_auc']:.4f}")
    print(f"   Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
    print(f"   MCC:           {test_metrics['mcc']:.4f}")
    print(f"\n   Confusion Matrix:")
    cm = test_metrics["confusion_matrix"]
    print(f"   [[TN={cm[0][0]:5d}  FP={cm[0][1]:5d}]")
    print(f"    [FN={cm[1][0]:5d}  TP={cm[1][1]:5d}]]")

    print("\n   Classification Report:")
    print(classification_report(y_test, test_pred, digits=4, target_names=["No Diabetes", "Diabetes"]))

    # ---------------------------
    # 8) Probability Calibration (Platt scaling on validation set)
    # ---------------------------
    print("\n🎯 Calibrating probabilities (Platt scaling on validation set)...")
    calibrated_pipeline = CalibratedClassifierCV(
        pipeline,
        cv=5,
        method="sigmoid",
    )
    calibrated_pipeline.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
    )

    # Compare calibration before/after
    cal_proba = calibrated_pipeline.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, test_proba)
    brier_after = brier_score_loss(y_test, cal_proba)
    print(f"   Brier score (before calibration): {brier_before:.4f}")
    print(f"   Brier score (after calibration):  {brier_after:.4f}")

    # Use calibrated probabilities for final evaluation
    test_proba_final = cal_proba
    test_pred_final = (test_proba_final >= best_threshold).astype(int)
    test_metrics = evaluate_predictions(y_test.values, test_pred_final, test_proba_final)

    print(f"   Calibrated ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Calibrated F1:      {test_metrics['f1']:.4f}")

    # ---------------------------
    # 9) Bootstrap Confidence Intervals
    # ---------------------------
    print(f"\n📊 Computing {N_BOOTSTRAP}-iteration bootstrap confidence intervals...")
    ci_results = bootstrap_confidence_interval(
        y_test.values, test_proba_final, best_threshold, n_bootstrap=N_BOOTSTRAP
    )
    print("   95% Confidence Intervals:")
    for metric, vals in ci_results.items():
        print(f"      {metric:12s}: {vals['mean']:.4f}  [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]")

    # ---------------------------
    # 10) SHAP Explainability
    # ---------------------------
    print("\n🔍 Computing SHAP values for model explainability...")
    # Use the inner pipeline's scaler to transform background data
    X_train_scaled_shap = pipeline.named_steps["scaler"].transform(X_train)
    X_test_scaled_shap = pipeline.named_steps["scaler"].transform(X_test)
    inner_model = pipeline.named_steps["model"]

    # Use 500 background samples for efficiency
    bg_sample = shap.sample(X_train_scaled_shap, min(500, len(X_train_scaled_shap)))
    explainer = shap.LinearExplainer(inner_model, bg_sample, feature_names=SELECTED_FEATURES)
    shap_values_test = explainer.shap_values(X_test_scaled_shap)

    # Global feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Mean_SHAP": mean_abs_shap
    }).sort_values("Mean_SHAP", ascending=False)

    print("\n   📈 Feature Importance (mean |SHAP|):")
    for _, row in shap_importance.iterrows():
        print(f"      {row['Feature']:25s}: {row['Mean_SHAP']:.4f}")

    # ---------------------------
    # 11) Feature importance (coefficients)
    # ---------------------------
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

    # ---------------------------
    # 12) Drift baseline
    # ---------------------------
    print("\n📊 Computing drift detection baseline from training set...")
    drift_baseline = compute_drift_baseline(X_train)

    # ---------------------------
    # 13) Save artifacts
    # ---------------------------
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Model bundle (for deployment) - uses CALIBRATED pipeline
    bundle = {
        "pipeline": calibrated_pipeline,
        "raw_pipeline": pipeline,  # keep original for SHAP
        "threshold": best_threshold,
        "feature_columns": SELECTED_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "model_name": "logistic_regression",
        "optuna_params": best_params,
        "optuna_best_cv_auc": study.best_value,
        "confidence_intervals": ci_results,
        "calibration": {
            "method": "platt_scaling",
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"\n💾 Model bundle saved: {MODEL_BUNDLE_PATH}")

    # SHAP explainer
    joblib.dump({
        "explainer": explainer,
        "expected_value": float(explainer.expected_value),
        "feature_names": SELECTED_FEATURES,
    }, SHAP_PATH)
    print(f"💾 SHAP explainer saved: {SHAP_PATH}")

    # Drift baseline
    joblib.dump(drift_baseline, DRIFT_BASELINE_PATH)
    print(f"💾 Drift baseline saved: {DRIFT_BASELINE_PATH}")

    # Metrics JSON
    metrics_output = {
        "threshold": best_threshold,
        "optuna_params": best_params,
        "optuna_best_cv_auc": study.best_value,
        "cv_fold_accuracies": fold_accuracies,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confidence_intervals": ci_results,
        "calibration": {
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"💾 Metrics saved: {METRICS_PATH}")

    # Test predictions CSV
    predictions_df = X_test.copy()
    predictions_df["Actual"] = y_test.values
    predictions_df["Predicted"] = test_pred_final
    predictions_df["Probability"] = test_proba_final
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"💾 Test predictions saved: {PREDICTIONS_PATH}")

    # ------------------------------------------------------------------
    # Provenance manifest - written LAST, after every artifact above is on
    # disk, so its hashes attest to completed outputs. If anything above
    # failed, no manifest is produced at all.
    # ------------------------------------------------------------------
    provenance.emit_training_manifest(
        project_root=PROJECT_ROOT,
        output_path=PROVENANCE_PATH,
        variant="A",
        model_name="logistic_regression",
        dataset_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        feature_names=SELECTED_FEATURES,
        training={
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "validation_size_of_train": 0.25,
            "stratified": True,
            "scaler": "StandardScaler",
            "optuna_sampler": "TPESampler",
            "optuna_sampler_seed": RANDOM_STATE,
            "optuna_n_trials": 100,
            "optuna_direction": "maximize",
            "optuna_best_params": best_params,
            "optuna_best_cv_auc": study.best_value,
            "cv_splits": 5,
            "calibration_method": "sigmoid",
            "calibration_cv": 5,
            "threshold_method": "youden_j",
            "selected_threshold": best_threshold,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_alpha": 0.05,
            "artifacts_dir": provenance.relative_path(ARTIFACTS_DIR, PROJECT_ROOT),
        },
        evaluation={
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "confidence_intervals": ci_results,
        },
        artifact_specs=[
            ("model_bundle", MODEL_BUNDLE_PATH, True),
            ("shap_explainer", SHAP_PATH, True),
            ("drift_baseline", DRIFT_BASELINE_PATH, True),
            ("metrics", METRICS_PATH, True),
            ("test_predictions", PREDICTIONS_PATH, False),
        ],
        source_files=[
            Path(__file__).resolve(),
            PROJECT_ROOT / "ml_core" / "evaluation.py",
            PROJECT_ROOT / "ml_core" / "bootstrap.py",
            PROJECT_ROOT / "ml_core" / "thresholds.py",
            PROJECT_ROOT / "ml_core" / "provenance.py",
        ],
        lockfile=PROJECT_ROOT / "requirements.lock",
    )
    print(f"💾 Provenance manifest saved: {PROVENANCE_PATH}")

    print("\n" + "=" * 60)
    print("✅ Logistic Regression pipeline complete!")
    print(f"   - Optuna trials: 100")
    print(f"   - Best threshold: {best_threshold:.4f} (Youden's J)")
    print(f"   - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   - Brier score: {brier_after:.4f} (calibrated)")
    print(f"   - SHAP explainer: saved")
    print(f"   - Drift baseline: saved")
    print("=" * 60)


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    # Rebind the module-level paths that main() reads. Hyperparameters,
    # features, seeds, thresholds and artifact formats are untouched.
    DATA_PATH = _args.data_path
    ARTIFACTS_DIR = _args.artifacts_dir
    MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.pkl"
    METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
    SHAP_PATH = ARTIFACTS_DIR / "shap_explainer.pkl"
    DRIFT_BASELINE_PATH = ARTIFACTS_DIR / "drift_baseline.pkl"
    PROVENANCE_PATH = ARTIFACTS_DIR / "training_manifest.json"
    main()


