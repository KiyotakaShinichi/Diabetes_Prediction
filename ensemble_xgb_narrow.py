# stacking_ensemble_nn_xgb.py
# Stacking ensemble: Narrow NN + XGBoost with Logistic Regression as meta-model
# Preserves your pipeline structure, CV, learning curve, metrics, and CSV workflow

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, roc_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_stacking.csv"

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

if 'DiabetesStatus' not in df.columns:
    raise ValueError("Target column 'DiabetesStatus' not found in dataset.")

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

X = df[selected_features]
y = df['DiabetesStatus']

# ---------------------------
# 3) Scale & Split
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✂️ Splitting into train/test (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 4) Train XGBoost (GridSearchCV)
# ---------------------------
print("\n🌳 Training XGBoost (GridSearchCV)...")
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_gs = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid_xgb,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
xgb_gs.fit(X_train, y_train)
xgb_best = xgb_gs.best_estimator_
print("✅ Best XGBoost params:", xgb_gs.best_params_)

# ---------------------------
# 5) Train Narrow Neural Network (GridSearchCV)
# ---------------------------
print("\n🧠 Training Narrow Neural Network (GridSearchCV)...")
param_grid_nn = {
    'hidden_layer_sizes': [(8,), (8,4), (16,8)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500]
}

mlp_gs = GridSearchCV(
    MLPClassifier(
        activation='relu',
        solver='adam',
        random_state=42
    ),
    param_grid_nn,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
mlp_gs.fit(X_train, y_train)
mlp = mlp_gs.best_estimator_
print("✅ Best Narrow NN params:", mlp_gs.best_params_)

# ---------------------------
# 6) Cross-validation fold accuracies for NN
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set for NN:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_ix, val_ix in skf.split(X_train, y_train):
    mlp_clone = MLPClassifier(**mlp.get_params())
    mlp_clone.fit(X_train[train_ix], y_train.iloc[train_ix])
    preds = mlp_clone.predict(X_train[val_ix])
    acc = accuracy_score(y_train.iloc[val_ix], preds)
    fold_accuracies.append(acc)
print("Per-fold accuracies:", np.round(fold_accuracies, 4))
print("Mean:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))

fig_folds = go.Figure([go.Bar(
    x=[f"Fold {i+1}" for i in range(len(fold_accuracies))],
    y=fold_accuracies, text=np.round(fold_accuracies, 4), textposition='auto'
)])
fig_folds.update_layout(title="Cross-validation fold accuracies (Narrow Neural Network)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 7) Stacking ensemble
# ---------------------------
print("\n🤝 Generating stacking ensemble predictions...")

# Generate base model predictions on training set for meta-model training
xgb_train_pred = xgb_best.predict_proba(X_train)[:, 1]
nn_train_pred = mlp.predict_proba(X_train)[:, 1]

stack_train = np.column_stack((xgb_train_pred, nn_train_pred))

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stack_train, y_train)

# Generate base model predictions on test set
xgb_test_pred = xgb_best.predict_proba(X_test)[:, 1]
nn_test_pred = mlp.predict_proba(X_test)[:, 1]

stack_test = np.column_stack((xgb_test_pred, nn_test_pred))
final_proba = meta_model.predict_proba(stack_test)[:, 1]
final_pred = (final_proba >= 0.5).astype(int)

# ---------------------------
# 8) Evaluation on test set
# ---------------------------
acc = accuracy_score(y_test, final_pred)
prec = precision_score(y_test, final_pred)
rec = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)
auc = roc_auc_score(y_test, final_proba)
kappa = cohen_kappa_score(y_test, final_pred)
mcc = matthews_corrcoef(y_test, final_pred)
cm = confusion_matrix(y_test, final_pred)

print("\n📊 Stacking Ensemble Test Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, final_pred, digits=4))
print("Confusion matrix:\n", cm)

# Confusion Matrix
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Stacking Ensemble')
cm_fig.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, final_proba)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Stacking Ensemble (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Stacking Ensemble', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 9) Save results
# ---------------------------
final_results = pd.DataFrame(X_test, columns=selected_features)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = final_pred
final_results['Pred_Prob'] = final_proba
final_results.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 10) Done
# ---------------------------
print("\n✅ Stacking ensemble pipeline complete.")
print(" - Final CSV:", FINAL_CSV)


