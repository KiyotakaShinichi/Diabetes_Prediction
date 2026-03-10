# logreg_with_kprototypes_updated.py
# Clustering-first pipeline using K-Prototypes -> add ClusterLabel as feature -> Logistic Regression
# Produces: CV per-fold accuracies, learning curve, confusion matrix, ROC curve,
# DBI+Elbow+CH voting validation, t-SNE visualization, CSV exports, and LR-based lift chart.
# Lift charts now show coefficient, odds ratio, and absolute lift.

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef,
    davies_bouldin_score, calinski_harabasz_score
)
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# ---------------------------
# Config / paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/cleaned_data_upd.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results_clusters_first_kproto.csv"
CLUSTER_PROFILE_CSV = r"C:/Users/L/Downloads/patient_cluster_profiles_clusters_first_kproto.csv"
CENTROIDS_CSV = r"C:/Users/L/Downloads/cluster_centroids_clusters_first_kproto.csv"
LIFT_CSV = r"C:/Users/L/Downloads/lr_lift_table.csv"

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

if 'DiabetesStatus' not in df.columns:
    raise ValueError("Target column 'DiabetesStatus' not found in dataset.")

df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})
if df['DiabetesStatus'].isna().any():
    raise ValueError("Some DiabetesStatus values couldn't be mapped. Check labels.")

X_orig = df[selected_features].copy()

# Encode categorical features
categorical_cols = ['GeneralHealth', 'HasHighBP', 'HasHighChol', 'AgeCategory',
                    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
                    'EducationLevel', 'IsPhysicallyActive']

le_dict = {}
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ---------------------------
# 2) Subsample 10k rows for testing (balanced)
# ---------------------------
SUBSET_SIZE = 30000
class_counts = df['DiabetesStatus'].value_counts()
max_samples_per_class = min(class_counts.min(), SUBSET_SIZE // 2)

df_balanced_subset = pd.concat([
    df[df['DiabetesStatus'] == 0].sample(max_samples_per_class, random_state=42),
    df[df['DiabetesStatus'] == 1].sample(max_samples_per_class, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

df = df_balanced_subset
X_orig = X_orig.loc[df.index].reset_index(drop=True)
print(f"✅ Balanced subset created: {df.shape[0]} rows")
print(df['DiabetesStatus'].value_counts())

# ---------------------------
# 3) Train/test split
# ---------------------------
print("\n✂️ Splitting into train/test (80/20 stratified)...")
X = df[selected_features].copy()
y = df['DiabetesStatus'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_orig_train = X_orig.loc[X_train.index].reset_index(drop=True)
X_orig_test = X_orig.loc[X_test.index].reset_index(drop=True)

# ---------------------------
# 4) Scale numeric features
# ---------------------------
numeric_cols = ['BMI', 'PoorPhysicalHealthDays']
scaler_num = StandardScaler()
X_train_num_scaled = scaler_num.fit_transform(X_train[numeric_cols].values)
X_test_num_scaled = scaler_num.transform(X_test[numeric_cols].values)

X_train_scaled = X_train.copy().reset_index(drop=True)
X_train_scaled[numeric_cols] = X_train_num_scaled
X_test_scaled = X_test.copy().reset_index(drop=True)
X_test_scaled[numeric_cols] = X_test_num_scaled

# ---------------------------
# 5) K-Prototypes clustering with voting
# ---------------------------
print("\n🔍 Running K-Prototypes clustering with automatic K selection...")
cat_idx = [X_train_scaled.columns.get_loc(c) for c in categorical_cols]

n_samples_train = X_train_scaled.shape[0]
max_k = min(10, max(2, n_samples_train - 1))
K_range = list(range(2, max_k + 1))

costs, dbi_scores, ch_scores = [], [], []
X_train_np = X_train_scaled.values

for k in K_range:
    kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42, n_init=5)
    labels = kproto.fit_predict(X_train_np, categorical=cat_idx)

    costs.append(kproto.cost_)
    X_mixed_for_metrics = np.column_stack(
        [X_train_scaled[numeric_cols].values, X_train_scaled[categorical_cols].values])
    try:
        dbi = davies_bouldin_score(X_mixed_for_metrics, labels)
    except:
        dbi = np.inf
    dbi_scores.append(dbi)
    try:
        ch = calinski_harabasz_score(X_mixed_for_metrics, labels)
    except:
        ch = -1
    ch_scores.append(ch)

cost_diffs = np.diff(costs)
elbow_k = K_range[np.argmin(cost_diffs) + 1] if len(cost_diffs) > 0 else K_range[0]
dbi_k = K_range[np.argmin(dbi_scores)]
ch_k = K_range[np.argmax(ch_scores)]
votes = [elbow_k, dbi_k, ch_k]
best_k = max(set(votes), key=votes.count)

print(f"DBI scores: {dict(zip(K_range, np.round(dbi_scores, 4)))}")
print(f"CH scores: {dict(zip(K_range, np.round(ch_scores, 4)))}")
print(f"Cost (elbow) values: {dict(zip(K_range, np.round(costs, 2)))}")
print(f"\n🎯 Voting results: Elbow={elbow_k}, DBI={dbi_k}, CH={ch_k}")
print(f"✅ Chosen K by majority vote: {best_k}")

kproto_final = KPrototypes(n_clusters=best_k, init='Cao', random_state=42, n_init=10)
train_clusters = kproto_final.fit_predict(X_train_np, categorical=cat_idx)
X_train_scaled['ClusterLabel'] = train_clusters

# Assign test clusters
X_test_scaled = X_test_scaled.reset_index(drop=True)
X_test_num = X_test_scaled[numeric_cols].values
X_test_cat = X_test_scaled[categorical_cols].values
centroids_arr = np.array(kproto_final.cluster_centroids_, dtype=object)
num_idx = [X_train_scaled.columns.get_loc(c) for c in numeric_cols]
cat_idx_local = [X_train_scaled.columns.get_loc(c) for c in categorical_cols]
centroids_num = centroids_arr[:, num_idx].astype(float)
centroids_cat = centroids_arr[:, cat_idx_local].astype(int)
num_distances = cdist(X_test_num, centroids_num, metric='euclidean')
cat_distances = np.zeros((X_test_cat.shape[0], centroids_cat.shape[0]))
for i, centroid in enumerate(centroids_cat):
    cat_distances[:, i] = np.sum(X_test_cat != centroid, axis=1)
total_distances = num_distances + cat_distances
X_test_clusters = np.argmin(total_distances, axis=1)
X_test_scaled['ClusterLabel'] = X_test_clusters
print(f"✅ Cluster labels added to training and test sets (k={best_k})")

# ---------------------------
# 6) Save centroids & cluster profiles
# ---------------------------
centroids_df = pd.DataFrame(centroids_arr, columns=selected_features)
centroids_df.index.name = 'Cluster'
centroids_df.to_csv(CENTROIDS_CSV, index=True)
print(f"💾 Saved centroids to: {CENTROIDS_CSV}")

profile = X_train_scaled.copy()
profile['DiabetesStatus'] = y_train.values
profile = profile.groupby('ClusterLabel')[selected_features + ['DiabetesStatus']].mean().round(4)
profile.to_csv(CLUSTER_PROFILE_CSV)
print(f"💾 Saved cluster profiles to: {CLUSTER_PROFILE_CSV}")

# ---------------------------
# 7) Logistic Regression
# ---------------------------
print("\n🔄 Training Logistic Regression (with cluster feature)...")
X_train_final = X_train_scaled.values
X_test_final = X_test_scaled.values

param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'penalty': ['l2']}
lr_gs = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42),
                     param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
lr_gs.fit(X_train_final, y_train)
lr = lr_gs.best_estimator_
print("✅ Best params:", lr_gs.best_params_)

# ---------------------------
# 8) Cross-validation fold accuracies
# ---------------------------
print("\n📊 Cross-validation accuracies:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_ix, val_ix in skf.split(X_train_final, y_train):
    lr_clone = LogisticRegression(**lr.get_params())
    lr_clone.fit(X_train_final[train_ix], y_train.iloc[train_ix])
    preds = lr_clone.predict(X_train_final[val_ix])
    fold_accuracies.append(accuracy_score(y_train.iloc[val_ix], preds))
print("Per-fold:", np.round(fold_accuracies, 4), "Mean:", np.mean(fold_accuracies))

# ---------------------------
# 9) Learning curve
# ---------------------------
train_sizes, train_scores, val_scores = learning_curve(
    lr, X_train_final, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - Logistic Regression', xaxis_title='Number of training examples',
                     yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 10) Evaluation on test set
# ---------------------------
proba_test = lr.predict_proba(X_test_final)[:, 1]
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
print(
    f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Kappa: {kappa:.4f}, MCC: {mcc:.4f}")
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, pred_test, digits=4))

# ---------------------------
# 11) t-SNE visualization
# ---------------------------
print("\n🌀 Running t-SNE for 2D visualization...")
X_all_for_tsne = np.vstack([X_train_final, X_test_final])
tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, max_iter=1000, metric="euclidean", random_state=42)
tsne_result = tsne.fit_transform(X_all_for_tsne)

df_all = pd.concat([X_train_scaled, X_test_scaled], axis=0).reset_index(drop=True)
df_all['TSNE1'] = tsne_result[:, 0]
df_all['TSNE2'] = tsne_result[:, 1]
proba_train = lr.predict_proba(X_train_final)[:, 1]
df_all['LR_Prob'] = np.concatenate([proba_train, proba_test])
df_all['ClusterLabel'] = df_all['ClusterLabel'].astype(int)

fig = px.scatter(
    df_all, x='TSNE1', y='TSNE2',
    color='ClusterLabel',
    hover_data=selected_features + ['LR_Prob'],
    title=f"t-SNE projection colored by ClusterLabel (k={best_k})",
)
fig.update_layout(height=700)
fig.show()

# ---------------------------
# 12) Final CSV export
# ---------------------------
final_results = X_test_scaled.copy().reset_index(drop=True)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = pred_test
final_results['Pred_Prob'] = proba_test
final_results['ClusterLabel'] = X_test_clusters
final_results['BMI_orig'] = X_orig_test['BMI'].reset_index(drop=True)

final_results.to_csv(FINAL_CSV, index=False)
print(f"💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 13) LR-based lift table (coefficients + odds ratio + absolute lift)
# ---------------------------
print("\n📈 Computing logistic regression-based lift table...")

import statsmodels.api as sm

def lr_lift_table(lr_model, X_df, y_series, feature_name, categorical_cols=[]):
    """
    Returns a DataFrame with:
    Feature, Category, Coefficient, OddsRatio, AbsoluteLiftMean
    """
    overall_mean = y_series.mean()
    df_feat = X_df[[feature_name]].copy()

    # One-hot encode categorical feature or keep numeric
    if feature_name in categorical_cols or df_feat[feature_name].dtype == 'object':
        df_feat = pd.get_dummies(df_feat[feature_name].astype(str), prefix=feature_name)
    else:
        df_feat = df_feat.astype(float)

    # Add constant column safely
    X_sm = sm.add_constant(df_feat, has_constant='add')
    X_sm = X_sm.astype(float)  # force float

    y_series_float = y_series.astype(float)

    logit_model = sm.Logit(y_series_float, X_sm)
    result = logit_model.fit(disp=0)

    lift_records = []
    for col in df_feat.columns:
        coef = result.params[col]
        odds_ratio = np.exp(coef)
        lift_abs = result.predict(X_sm) / overall_mean  # absolute lift per row
        lift_records.append({
            'Feature': feature_name,
            'Category': col,
            'Coefficient': coef,
            'OddsRatio': odds_ratio,
            'AbsoluteLiftMean': lift_abs.mean()
        })
    return pd.DataFrame(lift_records)

# Collect lift tables for all features
lift_tables = []
for col in selected_features:
    lift_tables.append(
        lr_lift_table(lr, final_results, final_results['Actual'], col, categorical_cols=categorical_cols)
    )

# Add ClusterLabel
lift_tables.append(
    lr_lift_table(lr, final_results, final_results['Actual'], 'ClusterLabel', categorical_cols=['ClusterLabel'])
)

# Combine all into one table
lift_table_full = pd.concat(lift_tables, ignore_index=True)
lift_table_full.to_csv(LIFT_CSV, index=False)

print(f"💾 LR-based lift table saved → {LIFT_CSV}")
print(lift_table_full.head(20))

print("\n✅ Pipeline complete.")
print(" - Final CSV:", FINAL_CSV)
print(" - Cluster profiles:", CLUSTER_PROFILE_CSV)
print(" - Cluster centroids:", CENTROIDS_CSV)
print(" - LR-based Lift CSV:", LIFT_CSV)



