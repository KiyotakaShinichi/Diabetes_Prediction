# ===============================
# nb_bernoulli_fixed.py
# ===============================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Load data
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
]

if 'DiabetesStatus' not in df.columns:
    raise ValueError("❌ Target column 'DiabetesStatus' not found in dataset.")

# Map target
df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

# Define features and target
X = df[selected_features].copy()
y = df['DiabetesStatus']

# Encode categorical features
categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']

for col in categorical_cols:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Create binary version of X (for Bernoulli)
X_bin = X.copy()
for col in X_bin.columns:
    med = X_bin[col].median()
    X_bin[col] = (X_bin[col] > med).astype(int)

# ---------------------------
# Split data (only once)
# ---------------------------
X_bin_train, X_bin_test, y_train, y_test = train_test_split(
    X_bin, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# Train BernoulliNB
# ---------------------------
print("\n🔄 Training Bernoulli Naive Bayes...")
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_bin_train, y_train)

# ---------------------------
# Cross-validation
# ---------------------------
print("\n📊 Cross-validation (5-fold):")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(bnb, X_bin_train, y_train, cv=skf, scoring='accuracy')
print("Per-fold:", np.round(scores, 4))
print("Mean:", np.mean(scores))

# ---------------------------
# Evaluate
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = bnb.predict_proba(X_bin_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, pred_test)
print("\nConfusion Matrix:\n", cm)

# Visualize ROC
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'BNB (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - BernoulliNB', xaxis_title='FPR', yaxis_title='TPR')
roc_fig.show()






