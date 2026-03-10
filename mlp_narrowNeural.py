# narrow_nn_pipeline.py
# Narrow Neural Network pipeline (same structure as your LogReg)
# Includes: preprocessing, GridSearchCV, CV, learning curve, metrics, and lifters approximation

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_narrow_nn.csv"

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
# 3) Scale & Split
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✂️ Splitting into train/test (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 4) Narrow Neural Network with GridSearchCV
# ---------------------------
print("\n🧠 Training Narrow Neural Network (GridSearchCV)...")
param_grid = {
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
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
mlp_gs.fit(X_train, y_train)
mlp = mlp_gs.best_estimator_
print("✅ Best params:", mlp_gs.best_params_)

# ---------------------------
# 5) Cross-validation fold accuracies
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set:")
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
# 6) Learning curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    mlp, X_train, y_train, cv=3, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - Narrow Neural Network',
                     xaxis_title='Training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 7) Evaluation on test set
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = mlp.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

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

# Confusion Matrix
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Narrow Neural Network')
cm_fig.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'NN (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Narrow Neural Network', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 8) Save results
# ---------------------------
final_results = pd.DataFrame(X_test, columns=selected_features)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = pred_test
final_results['Pred_Prob'] = proba_test
final_results.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 9) Done
# ---------------------------
print("\n✅ Narrow Neural Network pipeline complete.")
print(" - Final CSV:", FINAL_CSV)
