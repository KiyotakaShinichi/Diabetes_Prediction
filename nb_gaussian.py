# nb_gaussian.py
"""
Gaussian Naive Bayes pipeline (mirrors your Logistic Regression pipeline)
Save this block as nb_gaussian.py and run.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef,
    silhouette_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Config / paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_gaussian.csv"
CLUSTER_PROFILE_CSV = r"C:/Users/L/Downloads/patient_cluster_profiles_gaussian.csv"
CENTROIDS_CSV = r"C:/Users/L/Downloads/cluster_centroids_gaussian.csv"

# ---------------------------
# 1) Load & prepare data
# ---------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
]

# ensure target exists
if 'DiabetesStatus' not in df.columns:
    raise ValueError("Target column 'DiabetesStatus' not found in dataset.")

# map target
df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

# Encode categorical features (LabelEncoder)
categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[selected_features].copy()
y = df['DiabetesStatus'].copy()

# ---------------------------
# 2) Scale & split (80/20)
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Keep unscaled test features (original-feature scale) for CSV output & centroids mapping
X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=selected_features).reset_index(drop=True)

# ---------------------------
# 3) Gaussian Naive Bayes (train)
# ---------------------------
print("\n🔄 Training Gaussian Naive Bayes...")
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# ---------------------------
# 4) Cross-validation fold accuracies (per-fold)
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    clf = GaussianNB()
    clf.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
    preds = clf.predict(X_train_scaled[val_idx])
    acc = accuracy_score(y_train.iloc[val_idx], preds)
    fold_accuracies.append(acc)
print("Per-fold accuracies:", np.round(fold_accuracies, 4))
print("Mean:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))

fig_folds = go.Figure([go.Bar(x=[f"Fold {i+1}" for i in range(len(fold_accuracies))],
                             y=fold_accuracies, text=np.round(fold_accuracies,4), textposition='auto')])
fig_folds.update_layout(title="Cross-validation fold accuracies (GaussianNB)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 5) Learning curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    gnb, X_train_scaled, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - GaussianNB', xaxis_title='Number of training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 6) Final evaluation on test set & metrics
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = gnb.predict_proba(X_test_scaled)[:,1]
pred_test = (proba_test >= 0.5).astype(int)

acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)
kappa = cohen_kappa_score(y_test, pred_test)
mcc = matthews_corrcoef(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)

print("\n📊 Test set metrics (GaussianNB):")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Corr Coef (MCC): {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred_test, digits=4))
print("Confusion matrix:\n", cm)

cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Pred: 0', 'Pred: 1'],
    y=['True: 0', 'True: 1'],
    colorscale='Blues',
    text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - GaussianNB (Test set)')
cm_fig.show()

fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'GNB (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - GaussianNB (Test set)', xaxis_title='FPR', yaxis_title='TPR')
roc_fig.show()



