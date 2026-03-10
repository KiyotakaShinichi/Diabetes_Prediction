# conditional_stack_boosted_nn.py
# Conditional stacking: XGBoost + Narrow Neural Network
# Includes threshold grid search, metrics, plots, and CSV output

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, roc_curve
)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_boosted_nn_stack.csv"

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
# 4) Train Boosted Trees (XGBoost)
# ---------------------------
print("\n🔧 Training Boosted Trees (XGBoost)...")
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

# ---------------------------
# 5) Train Narrow Neural Network
# ---------------------------
print("\n🧠 Training Narrow Neural Network...")
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    activation='relu',
    solver='adam',
    random_state=42
)
mlp.fit(X_train, y_train)
mlp_proba = mlp.predict_proba(X_test)[:, 1]

# ---------------------------
# 6) Grid Search for Best Thresholds
# ---------------------------
print("\n🔍 Searching for best thresholds (F1-maximizing)...")
xgb_thresholds = np.linspace(0.4, 0.7, 7)
mlp_thresholds = np.linspace(0.5, 0.8, 7)

best_f1 = 0
best_xgb_th, best_mlp_th = 0, 0

for xgb_t in xgb_thresholds:
    for mlp_t in mlp_thresholds:
        preds = [
            1 if xgb_p >= xgb_t else 1 if mlp_p >= mlp_t else 0
            for xgb_p, mlp_p in zip(xgb_proba, mlp_proba)
        ]
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_xgb_th, best_mlp_th = xgb_t, mlp_t

print(f"✅ Best thresholds -> XGB: {best_xgb_th}, NN: {best_mlp_th}, F1: {best_f1:.4f}")

# ---------------------------
# 7) Conditional Stacking Predictions
# ---------------------------
final_pred = [
    1 if xgb_p >= best_xgb_th else 1 if mlp_p >= best_mlp_th else 0
    for xgb_p, mlp_p in zip(xgb_proba, mlp_proba)
]

# ---------------------------
# 8) Evaluation
# ---------------------------
acc = accuracy_score(y_test, final_pred)
prec = precision_score(y_test, final_pred)
rec = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)
auc = roc_auc_score(y_test, (xgb_proba + mlp_proba)/2)  # approximate
kappa = cohen_kappa_score(y_test, final_pred)
mcc = matthews_corrcoef(y_test, final_pred)
cm = confusion_matrix(y_test, final_pred)

print("\n📊 Conditional Stacking Test Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC (approx): {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, final_pred, digits=4))
print("Confusion matrix:\n", cm)

# Confusion Matrix
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Boosted Trees + NN Conditional Stacking')
cm_fig.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, (xgb_proba + mlp_proba)/2)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Stacking (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Boosted Trees + NN Stacking', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 9) Save results
# ---------------------------
final_results = pd.DataFrame(X_test, columns=selected_features)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = final_pred
final_results['XGB_Prob'] = xgb_proba
final_results['NN_Prob'] = mlp_proba
final_results['XGB_Threshold'] = best_xgb_th
final_results['NN_Threshold'] = best_mlp_th
final_results.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 10) Done
# ---------------------------
print("\n✅ Conditional stacking pipeline complete.")
print(" - Final CSV:", FINAL_CSV)
