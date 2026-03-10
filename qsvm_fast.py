# qsvm_model_fast.py
# Quadratic SVM classification pipeline (no GridSearchCV)
# Includes: CV accuracies, learning curve, confusion matrix, ROC curve, PCA visualization, and full metrics report.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef, roc_curve
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_qsvm_fast.csv"

# ---------------------------
# 1) Load & Prepare Data
# ---------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
]

if 'DiabetesStatus' not in df.columns:
    raise ValueError("Target column 'DiabetesStatus' not found in dataset.")

df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[selected_features].values
y = df['DiabetesStatus'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 2) Train-Test Split
# ---------------------------
print("\n✂️ Splitting into train/test (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 3) Quadratic SVM (Direct)
# ---------------------------
print("\n⚙️ Training Quadratic SVM (no GridSearch)...")
svm_best = SVC(
    kernel='poly',
    degree=2,
    C=1.0,
    gamma='scale',
    coef0=1.0,
    probability=True,
    random_state=42
)
svm_best.fit(X_train, y_train)
print("✅ Model trained successfully.")

# ---------------------------
# 4) Cross-validation Accuracy (per fold)
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_ix, val_ix in skf.split(X_train, y_train):
    model = SVC(**svm_best.get_params())
    model.fit(X_train[train_ix], y_train[train_ix])
    preds = model.predict(X_train[val_ix])
    acc = accuracy_score(y_train[val_ix], preds)
    fold_accuracies.append(acc)
print("Per-fold accuracies:", np.round(fold_accuracies, 4))
print("Mean:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))

fig_folds = go.Figure([go.Bar(
    x=[f"Fold {i+1}" for i in range(len(fold_accuracies))],
    y=fold_accuracies,
    text=np.round(fold_accuracies,4),
    textposition='auto'
)])
fig_folds.update_layout(title="Cross-validation fold accuracies (Quadratic SVM)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 5) Learning Curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    svm_best, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - Quadratic SVM', xaxis_title='Training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 6) Evaluation on Test Set
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = svm_best.predict_proba(X_test)[:,1]
pred_test = (proba_test >= 0.5).astype(int)

acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)
kappa = cohen_kappa_score(y_test, pred_test)
mcc = matthews_corrcoef(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)

print("\n📊 Test set metrics (Quadratic SVM):")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Corr Coef (MCC): {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred_test, digits=4))
print("Confusion matrix:\n", cm)

# Confusion matrix
cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Pred: 0', 'Pred: 1'],
    y=['True: 0', 'True: 1'],
    colorscale='Blues',
    text=cm,
    texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Quadratic SVM (Test set)')
cm_fig.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Quadratic SVM (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Quadratic SVM (Test set)', xaxis_title='FPR', yaxis_title='TPR')
roc_fig.show()

# ---------------------------
# 7) PCA Visualization
# ---------------------------
pca = PCA(n_components=2)
proj = pca.fit_transform(X_test)
df_viz = pd.DataFrame({
    'PCA1': proj[:,0],
    'PCA2': proj[:,1],
    'True': y_test,
    'Pred': pred_test,
    'Prob': proba_test
})

fig_pca = px.scatter(
    df_viz,
    x='PCA1', y='PCA2',
    color='Pred',
    hover_data=['True', 'Prob'],
    title='PCA Projection - Quadratic SVM (Test set)'
)
fig_pca.show()

# ---------------------------
# 8) Save Final CSV
# ---------------------------
final_df = pd.DataFrame(X_test, columns=selected_features)
final_df['Actual'] = y_test
final_df['Predicted'] = pred_test
final_df['Pred_Prob'] = proba_test
final_df.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

print("\n✅ Pipeline complete.")
