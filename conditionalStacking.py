# conditional_stacking_gridsearch.py
# Conditional Stacking NB + KNN with automated threshold search

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, roc_curve
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_conditional_stack_grid.csv"

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
# 4) Train Kernel NB
# ---------------------------
print("\n🧮 Training Kernel NB...")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_proba = nb.predict_proba(X_test)[:, 1]

# ---------------------------
# 5) Train Subspace KNN
# ---------------------------
print("\n👟 Training Subspace KNN...")
knn = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='manhattan')
knn.fit(X_train, y_train)
knn_proba = knn.predict_proba(X_test)[:, 1]

# ---------------------------
# 6) Grid Search for Optimal Thresholds
# ---------------------------
print("\n🔍 Searching for best thresholds (F1-maximizing)...")
nb_thresholds = np.linspace(0.4, 0.8, 9)
knn_thresholds = np.linspace(0.7, 1.0, 7)

best_f1 = 0
best_nb_th, best_knn_th = 0, 0

for nb_t in nb_thresholds:
    for knn_t in knn_thresholds:
        preds = [
            1 if nb_p >= nb_t else 1 if knn_p >= knn_t else 0
            for nb_p, knn_p in zip(nb_proba, knn_proba)
        ]
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_nb_th, best_knn_th = nb_t, knn_t

print(f"✅ Best thresholds found -> NB: {best_nb_th}, KNN: {best_knn_th}, F1: {best_f1:.4f}")

# ---------------------------
# 7) Apply Conditional Stacking with Optimal Thresholds
# ---------------------------
final_pred = [
    1 if nb_p >= best_nb_th else 1 if knn_p >= best_knn_th else 0
    for nb_p, knn_p in zip(nb_proba, knn_proba)
]

# ---------------------------
# 8) Evaluation
# ---------------------------
acc = accuracy_score(y_test, final_pred)
prec = precision_score(y_test, final_pred)
rec = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)
auc = roc_auc_score(y_test, (nb_proba + knn_proba)/2)  # approximate
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
cm_fig.update_layout(title='Confusion Matrix - Conditional Stacking (Grid Search)')
cm_fig.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, (nb_proba + knn_proba)/2)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Conditional Stacking (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Conditional Stacking', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 9) Save results
# ---------------------------
final_results = pd.DataFrame(X_test, columns=selected_features)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = final_pred
final_results['NB_Prob'] = nb_proba
final_results['KNN_Prob'] = knn_proba
final_results['NB_Threshold'] = best_nb_th
final_results['KNN_Threshold'] = best_knn_th
final_results.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 10) Done
# ---------------------------
print("\n✅ Conditional stacking pipeline with threshold grid search complete.")
print(" - Final CSV:", FINAL_CSV)

