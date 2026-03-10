# enhanced_boosted_trees.py
# Enhanced Boosted Trees pipeline with stacking (XGB + RF + LGBM + CatBoost)
# Includes Optuna hyperparameter tuning, Youden's J threshold, CV accuracies, learning curve, full metrics, and plots

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_boostedtrees.csv"

# ---------------------------
# 2) Load & Prepare Data
# ---------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive'
]

df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

categorical_cols = [
    'GeneralHealth', 'HasHighBP', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'EducationLevel', 'IsPhysicallyActive'
]
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[selected_features].values
y = df['DiabetesStatus'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 3) Split Data
# ---------------------------
print("\n✂️ Splitting into train/test (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 4) Optuna Hyperparameter Optimization (XGB)
# ---------------------------
print("\n🔧 Running Optuna for XGBoost hyperparameters...")
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("✅ Best XGBoost params found:", best_params)

# ---------------------------
# 5) Stacking Ensemble
# ---------------------------
print("\n🧩 Training stacking ensemble (XGB + RF + LGBM + CatBoost)...")
estimators = [
    ("xgb", XGBClassifier(**best_params, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ("lgb", LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42)),
    ("cat", CatBoostClassifier(verbose=0, iterations=300, learning_rate=0.1, random_state=42))
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
stack_model.fit(X_train, y_train)
print("✅ Stacking model trained.")

# ---------------------------
# 6) Cross-validation per-fold accuracy
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_acc = []
for train_ix, val_ix in skf.split(X_train, y_train):
    model_fold = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
    model_fold.fit(X_train[train_ix], y_train[train_ix])
    preds_fold = model_fold.predict(X_train[val_ix])
    fold_acc.append(accuracy_score(y_train[val_ix], preds_fold))
print("Per-fold accuracies:", np.round(fold_acc, 4))
print("Mean:", np.mean(fold_acc), "Std:", np.std(fold_acc))

fig_folds = go.Figure([go.Bar(
    x=[f"Fold {i+1}" for i in range(len(fold_acc))],
    y=fold_acc, text=np.round(fold_acc, 4), textposition='auto'
)])
fig_folds.update_layout(title="Cross-validation fold accuracies (Stacked Ensemble)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 7) Learning Curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    stack_model, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - Stacked Ensemble', xaxis_title='Training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 8) Threshold Optimization (Youden's J)
# ---------------------------
print("\n🔍 Optimizing decision threshold (Youden's J)...")
y_prob = stack_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_index = tpr - fpr
optimal_threshold = thresholds[np.argmax(youden_index)]
print("✅ Optimal threshold:", optimal_threshold)

y_pred = (y_prob >= optimal_threshold).astype(int)

# ---------------------------
# 9) Evaluate on Test Set
# ---------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Test set metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", cm)

# Confusion matrix visualization
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Stacked Ensemble')
cm_fig.show()

# ROC curve visualization
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Stacked Ensemble (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Stacked Ensemble', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 10) Save Final CSV
# ---------------------------
final_df = pd.DataFrame(X_test, columns=selected_features)
final_df['Actual'] = y_test
final_df['Predicted'] = y_pred
final_df['Pred_Prob'] = y_prob
final_df.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 11) Done
# ---------------------------
print("\n✅ Enhanced Stacked Trees pipeline complete.")

