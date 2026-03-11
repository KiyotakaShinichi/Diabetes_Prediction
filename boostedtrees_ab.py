# boostedtrees_ab.py
# XGBoost pipeline with Optuna hyperparameter tuning for A/B testing (variant B).
"""
XGBoost model for diabetes prediction - A/B testing variant B.

Uses the same feature set as logistic regression for consistent comparison.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------
# Configuration
# ---------------------------
RANDOM_STATE = 42
DATA_PATH = Path("cleaned_data.csv")
ARTIFACTS_DIR = Path("model_artifacts")
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "boosted_model_bundle.pkl"
METRICS_PATH = ARTIFACTS_DIR / "boosted_metrics.json"
SHAP_PATH = ARTIFACTS_DIR / "boosted_shap_explainer.pkl"
DRIFT_BASELINE_PATH = ARTIFACTS_DIR / "boosted_drift_baseline.pkl"

N_BOOTSTRAP = 200

# Same features as logistic regression (for A/B testing consistency)
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


def compute_youden_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thresholds[best_idx])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics."""
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


def compute_drift_baseline(X_train: pd.DataFrame) -> dict:
    """Compute training-set statistics for drift detection."""
    return {
        "means": X_train.mean().to_dict(),
        "stds": X_train.std().to_dict(),
        "medians": X_train.median().to_dict(),
        "q25": X_train.quantile(0.25).to_dict(),
        "q75": X_train.quantile(0.75).to_dict(),
        "n_train": len(X_train),
        "feature_columns": list(X_train.columns),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> dict:
    """Bootstrap 95% confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_boot: dict[str, list[float]] = {
        "accuracy": [], "precision": [], "recall": [],
        "f1": [], "roc_auc": [], "brier_score": [],
    }
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t, y_p = y_true[idx], y_proba[idx]
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


def main() -> None:
    print("=" * 60)
    print("DIABETES PREDICTION - XGBoost Pipeline (A/B Variant B)")
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

    required_columns = set(SELECTED_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\n📊 Selected Features ({len(SELECTED_FEATURES)}):")
    for feat in SELECTED_FEATURES:
        print(f"   - {feat}: {FEATURE_LABELS.get(feat, feat)}")

    # ---------------------------
    # 2) Split data
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
    print("\n🔄 Running Optuna hyperparameter optimization (50 trials)...")

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
        
        model = XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        )
        
        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    print(f"\n✅ Best Optuna params: {best_params}")
    print(f"   Best CV ROC-AUC: {study.best_value:.4f}")

    # ---------------------------
    # 4) Train final model
    # ---------------------------
    print("\n🏗️ Training final XGBoost model...")

    final_model = XGBClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )
    final_model.fit(X_train, y_train)

    # ---------------------------
    # 5) Youden's J threshold (on validation set)
    # ---------------------------
    print("\n🎯 Computing optimal threshold using Youden's J on validation set...")
    val_proba = final_model.predict_proba(X_val)[:, 1]
    best_threshold = compute_youden_threshold(y_val.values, val_proba)
    print(f"   Best threshold (Youden's J): {best_threshold:.4f}")

    val_pred = (val_proba >= best_threshold).astype(int)
    val_metrics = evaluate_predictions(y_val.values, val_pred, val_proba)
    print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"   Validation F1: {val_metrics['f1']:.4f}")

    # ---------------------------
    # 6) Final evaluation on TEST set
    # ---------------------------
    print("\n🔍 Evaluating on held-out TEST set...")
    test_proba = final_model.predict_proba(X_test)[:, 1]
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
    
    cm = test_metrics["confusion_matrix"]
    print(f"\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:5d}  FP={cm[0][1]:5d}]")
    print(f"    [FN={cm[1][0]:5d}  TP={cm[1][1]:5d}]]")

    print("\n   Classification Report:")
    print(classification_report(y_test, test_pred, digits=4, target_names=["No Diabetes", "Diabetes"]))

    # ---------------------------
    # 7) Probability Calibration
    # ---------------------------
    print("\n🎯 Calibrating probabilities (Platt scaling on validation set)...")
    calibrated_model = CalibratedClassifierCV(
        final_model,
        cv=5,
        method="sigmoid",
    )
    calibrated_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
    )

    cal_proba = calibrated_model.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, test_proba)
    brier_after = brier_score_loss(y_test, cal_proba)
    print(f"   Brier score (before calibration): {brier_before:.4f}")
    print(f"   Brier score (after calibration):  {brier_after:.4f}")

    test_proba_final = cal_proba
    test_pred_final = (test_proba_final >= best_threshold).astype(int)
    test_metrics = evaluate_predictions(y_test.values, test_pred_final, test_proba_final)
    print(f"   Calibrated ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # ---------------------------
    # 8) Bootstrap Confidence Intervals
    # ---------------------------
    print(f"\n📊 Computing {N_BOOTSTRAP}-iteration bootstrap confidence intervals...")
    ci_results = bootstrap_confidence_interval(
        y_test.values, test_proba_final, best_threshold
    )
    print("   95% Confidence Intervals:")
    for metric, vals in ci_results.items():
        print(f"      {metric:12s}: {vals['mean']:.4f}  [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]")

    # ---------------------------
    # 9) SHAP Explainability
    # ---------------------------
    print("\n🔍 Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(final_model)
    shap_values_test = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Mean_SHAP": mean_abs_shap
    }).sort_values("Mean_SHAP", ascending=False)

    print("\n   📈 Feature Importance (mean |SHAP|):")
    for _, row in shap_importance.iterrows():
        print(f"      {row['Feature']:25s}: {row['Mean_SHAP']:.4f}")

    # ---------------------------
    # 10) Feature importance (XGBoost native)
    # ---------------------------
    print("\n📈 Feature Importance (XGBoost gain):")
    importance_df = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    for _, row in importance_df.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    # ---------------------------
    # 11) Save artifacts
    # ---------------------------
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    bundle = {
        "pipeline": calibrated_model,
        "raw_model": final_model,
        "threshold": best_threshold,
        "feature_columns": SELECTED_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "model_name": "xgboost_boosted_trees",
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

    # Save drift baseline
    drift_baseline = compute_drift_baseline(X_train)
    joblib.dump(drift_baseline, DRIFT_BASELINE_PATH)
    print(f"💾 Drift baseline saved: {DRIFT_BASELINE_PATH}")

    joblib.dump({
        "explainer": explainer,
        "expected_value": float(explainer.expected_value),
        "feature_names": SELECTED_FEATURES,
    }, SHAP_PATH)
    print(f"💾 SHAP explainer saved: {SHAP_PATH}")

    metrics_output = {
        "threshold": best_threshold,
        "optuna_params": best_params,
        "optuna_best_cv_auc": study.best_value,
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

    print("\n" + "=" * 60)
    print("✅ XGBoost pipeline complete!")
    print(f"   - Optuna trials: 50")
    print(f"   - Best threshold: {best_threshold:.4f} (Youden's J)")
    print(f"   - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   - Brier score: {brier_after:.4f} (calibrated)")
    print(f"   - SHAP explainer: saved")
    print("=" * 60)


if __name__ == "__main__":
    main()
