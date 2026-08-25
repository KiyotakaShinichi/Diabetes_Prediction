# enc1_smoteenn_balanced.py
# Path configuration is resolved before the heavy third-party imports below so
# that `python <script> --help` works without plotly/imblearn/kmodes/statsmodels
# installed. See experiment_config.py.
import experiment_config

experiment_config.require_direct_execution(__name__, 'enc1_smoteenn.py')
_ARGS = experiment_config.parse_args(
    'enc1_smoteenn.py',
    default_data_path=experiment_config.PROJECT_ROOT / 'enc1.csv',
)
DATA_PATH = _ARGS.data_path
OUTPUT_PATH = experiment_config.result_path(_ARGS, 'enc1_smoteenn.csv')

import pandas as pd

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from collections import Counter

# ---------------------------
# 1) Config / Paths
# ---------------------------

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

# Encode target
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

# Encode categorical features
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

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nOriginal class distribution:", Counter(y))

# ---------------------------
# 3) Apply Balanced SMOTEENN
# ---------------------------
print("\n🔄 Applying balanced SMOTEENN...")
smoteenn = SMOTEENN(sampling_strategy='auto', random_state=42)
X_res, y_res = smoteenn.fit_resample(X_scaled, y)
print("After SMOTEENN distribution:", Counter(y_res))

# ---------------------------
# 4) Save Resampled Dataset
# ---------------------------
df_res = pd.DataFrame(X_res, columns=selected_features)
df_res['DiabetesStatus'] = y_res
df_res.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Resampled dataset saved to: {OUTPUT_PATH}")
