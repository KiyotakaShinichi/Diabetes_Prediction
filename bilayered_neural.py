# bilayer_nn.py
# Bilayer Neural Network classification pipeline
# Includes train/validation split, learning curve, ROC curve, and metrics report.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, roc_curve
)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Config / Paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_bilayer_nn.csv"

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
# 4) Build Bilayer Neural Network
# ---------------------------
def build_bilayer_nn(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_bilayer_nn(X_train.shape[1])

# ---------------------------
# 5) Train with Early Stopping
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
)
print("✅ Training complete.")

# ---------------------------
# 6) Learning Curve Visualization
# ---------------------------
fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(
    x=list(range(1, len(history.history['accuracy'])+1)),
    y=history.history['accuracy'], mode='lines+markers', name='Train accuracy'
))
fig_lc.add_trace(go.Scatter(
    x=list(range(1, len(history.history['val_accuracy'])+1)),
    y=history.history['val_accuracy'], mode='lines+markers', name='Validation accuracy'
))
fig_lc.update_layout(title='Learning Curve - Bilayer NN', xaxis_title='Epoch', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 7) Evaluate on Test Set
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = model.predict(X_test).flatten()
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

# Confusion matrix visualization
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=['Pred: 0', 'Pred: 1'], y=['True: 0', 'True: 1'],
    colorscale='Blues', text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Bilayer NN')
cm_fig.show()

# ROC curve visualization
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Bilayer NN (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Bilayer NN', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
roc_fig.show()

# ---------------------------
# 8) Save Final CSV
# ---------------------------
final_df = pd.DataFrame(X_test, columns=selected_features)
final_df['Actual'] = y_test
final_df['Predicted'] = pred_test
final_df['Pred_Prob'] = proba_test
final_df.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 9) Done
# ---------------------------
print("\n✅ Bilayer NN pipeline complete.")
print(" - Final CSV:", FINAL_CSV)
