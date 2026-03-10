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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


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
    # 7) Feature importance
    # ---------------------------
    print("\n📈 Feature Importance (XGBoost gain):")
    importance_df = pd.DataFrame({
        "Feature": SELECTED_FEATURES,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    for _, row in importance_df.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    # ---------------------------
    # 8) Save artifacts
    # ---------------------------
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Save raw XGBClassifier (no wrapper needed - it has predict_proba natively)
    bundle = {
        "pipeline": final_model,
        "threshold": best_threshold,
        "feature_columns": SELECTED_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "model_name": "xgboost_boosted_trees",
        "optuna_params": best_params,
        "optuna_best_cv_auc": study.best_value,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"\n💾 Model bundle saved: {MODEL_BUNDLE_PATH}")

    metrics_output = {
        "threshold": best_threshold,
        "optuna_params": best_params,
        "optuna_best_cv_auc": study.best_value,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"💾 Metrics saved: {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("✅ XGBoost pipeline complete!")
    print(f"   - Optuna trials: 50")
    print(f"   - Best threshold: {best_threshold:.4f} (Youden's J)")
    print(f"   - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
