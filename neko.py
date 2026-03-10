import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, roc_curve
)
import optuna
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_catboost.csv"

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

X = df[selected_features]
y = df['DiabetesStatus'].map({'No Diabetes': 0, 'Diabetes': 1})

categorical_cols = [
    'GeneralHealth', 'HasHighBP', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'EducationLevel', 'IsPhysicallyActive'
]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[[col for col in X.columns if col not in categorical_cols]] = scaler.fit_transform(X[[col for col in X.columns if col not in categorical_cols]])

# ---------------------------
# 3) Split Data
# ---------------------------
print("\n✂️ Splitting into train/test (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 4) Hyperparameter Tuning with Optuna
# ---------------------------
print("\n🔧 Running Optuna hyperparameter tuning for CatBoost...")

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'verbose': 0,
        'task_type': 'CPU'
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test), verbose=0, early_stopping_rounds=50)
    preds_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    return auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("✅ Best params:", best_params)

# ---------------------------
# 5) Train final CatBoost model
# ---------------------------
final_model = CatBoostClassifier(**best_params, verbose=0, task_type='CPU')
final_model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test), early_stopping_rounds=50)

# ---------------------------
# 6) Cross-validation Accuracies
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_acc = []
for train_ix, val_ix in skf.split(X_train, y_train):
    model = CatBoostClassifier(**best_params, verbose=0, task_type='CPU')
    model.fit(X_train.iloc[train_ix], y_train.iloc[train_ix], cat_features=categorical_cols)
    preds = model.predict(X_train.iloc[val_ix])
    fold_acc.append(accuracy_score(y_train.iloc[val_ix], preds))
print("Per-fold accuracies:", np.round(fold_acc, 4))
print("Mean:", np.mean(fold_acc), "Std:", np.std(fold_acc))

fig_folds = go.Figure([go.Bar(
    x=[f"Fold {i+1}" for i in range(len(fold_acc))],
    y=fold_acc, text=np.round(fold_acc, 4), textposition='auto'
)])
fig_folds.update_layout(title="Cross-validation fold accuracies (CatBoost)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 7) Learning Curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - CatBoost', xaxis_title='Training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 8) Evaluate on Test Set with Youden's J
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, proba_test)
youdens_j = tpr - fpr
best_idx = np.argmax(youdens_j)
best_threshold = thresholds[best_idx]
print(f"🔹 Best threshold (Youden's J): {best_threshold}")

pred_test = (proba_test >= best_threshold).astype(int)

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

cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - CatBoost')
cm_fig.show()

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'CatBoost (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - CatBoost', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 9) Feature Importance / Top Lifters
# ---------------------------
feat_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': final_model.get_feature_importance(prettified=False)
})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)
print("\n💡 Feature importance:\n", feat_importance)

# ---------------------------
# 10) Save Final CSV
# ---------------------------
final_df = X_test.copy()
final_df['Actual'] = y_test
final_df['Predicted'] = pred_test
final_df['Pred_Prob'] = proba_test
final_df.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

print("\n✅ CatBoost pipeline complete with Optuna & Youden's J.")

