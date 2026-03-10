# logreg_optuna_youden.py
# Logistic Regression pipeline with Optuna hyperparameter tuning, Youden's J threshold, visualizations, and evaluation metrics.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, roc_curve
)
import optuna
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_logreg.csv"

# ---------------------------
# 2) Load & prepare data
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

# Label encode categorical columns
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
# 3) Scale & split
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✂️ Splitting into train/test (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 4) Optuna hyperparameter tuning
# ---------------------------
print("\n🔄 Tuning Logistic Regression (Optuna)...")

def objective(trial):
    C = trial.suggest_loguniform('C', 0.01, 10)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    lr = LogisticRegression(C=C, solver=solver, penalty='l2', max_iter=2000)
    score = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
print("✅ Best params found by Optuna:", best_params)

lr = LogisticRegression(**best_params, max_iter=2000)
lr.fit(X_train, y_train)

# ---------------------------
# 5) Cross-validation fold accuracies
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_ix, val_ix in skf.split(X_train, y_train):
    lr_clone = LogisticRegression(**lr.get_params())
    lr_clone.fit(X_train[train_ix], y_train.iloc[train_ix])
    preds = lr_clone.predict(X_train[val_ix])
    acc = accuracy_score(y_train.iloc[val_ix], preds)
    fold_accuracies.append(acc)
print("Per-fold accuracies:", np.round(fold_accuracies, 4))
print("Mean:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))

fig_folds = go.Figure([go.Bar(
    x=[f"Fold {i+1}" for i in range(len(fold_accuracies))],
    y=fold_accuracies, text=np.round(fold_accuracies, 4), textposition='auto'
)])
fig_folds.update_layout(title="Cross-validation fold accuracies (Logistic Regression)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 6) Learning curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    lr, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(
    title='Learning Curve - Logistic Regression',
    xaxis_title='Training examples',
    yaxis_title='Accuracy'
)
fig_lc.show()


# ---------------------------
# 7) Evaluation on test set with Youden's J
# ---------------------------
print("\n🔍 Evaluating on test set...")

# predicted probabilities
proba_test = lr.predict_proba(X_test)[:, 1]

# Compute Youden's J for best threshold
fpr, tpr, thresholds = roc_curve(y_test, proba_test)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]
print("🔹 Best threshold (Youden's J):", best_threshold)

# Apply threshold
pred_test = (proba_test >= best_threshold).astype(int)

# Metrics
acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)
kappa = cohen_kappa_score(y_test, pred_test)
mcc = matthews_corrcoef(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)

print("\n📊 Test set metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")

print("\nClassification report:\n", classification_report(y_test, pred_test, digits=4))
print("Confusion matrix:\n", cm)

# Confusion matrix visualization
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Logistic Regression')
cm_fig.show()

# ROC curve visualization
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'LR (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Logistic Regression', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# FEATURE IMPORTANCE ("LIFTERS")
# ---------------------------
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': lr.coef_[0]
})
feature_importance['Odds Ratio'] = np.exp(feature_importance['Coefficient'])

# Optional: assume baseline 0.2 (20% probability)
baseline_prob = 0.2
baseline_odds = baseline_prob / (1 - baseline_prob)
feature_importance['New Odds'] = baseline_odds * feature_importance['Odds Ratio']
feature_importance['New Prob'] = feature_importance['New Odds'] / (1 + feature_importance['New Odds'])
feature_importance['Absolute Lift (pp)'] = (feature_importance['New Prob'] - baseline_prob) * 100

feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print("\n🚀 Top Lifters (increase diabetes odds):")
print(feature_importance[['Feature', 'Coefficient', 'Odds Ratio', 'Absolute Lift (pp)']].head(10))

print("\n🧊 Features that Decrease Diabetes Odds:")
print(feature_importance[['Feature', 'Coefficient', 'Odds Ratio', 'Absolute Lift (pp)']].tail(10))

# ---------------------------
# 8) Save results to CSV
# ---------------------------
final_results = pd.DataFrame(X_test, columns=selected_features)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = pred_test
final_results['Pred_Prob'] = proba_test
final_results.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 9) Finish
# ---------------------------
print("\n✅ Logistic Regression pipeline complete with Optuna & Youden's J.")
print(" - Final CSV:", FINAL_CSV)

import joblib
import os

os.makedirs("model_artifacts", exist_ok=True)

# Save everything
joblib.dump(lr, "model_artifacts/diabetes_model.pkl")
joblib.dump(scaler, "model_artifacts/scaler.pkl")
with open("model_artifacts/threshold.txt", 'w') as f:
    f.write(str(best_threshold))

print("✅ Model saved for deployment!")




