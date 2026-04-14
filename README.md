# Diabetes Prediction System

Clinical decision support project for diabetes risk prediction using multiple machine learning approaches, with:

- A FastAPI backend for inference and analytics
- A Streamlit public-facing assessment app
- A separate login-protected admin dashboard
- Model training scripts and saved artifacts

## What This Project Includes

- Inference API with A/B model routing and logging
- Logistic Regression baseline model with Optuna tuning
- Optional boosted-tree model variant for A/B testing
- SHAP-based local explanations
- Drift baseline generation and drift checks
- SQLite by default, PostgreSQL support via environment variable

## Architecture

- Public frontend: Streamlit app on port 8501
- API backend: FastAPI app on port 8000
- Admin dashboard: Streamlit app on internal-only port 8502

## Repository Overview

### Core Applications

- `app.py`: FastAPI inference API (`/predict`, `/explain`, drift and analytics endpoints)
- `streamlit_app.py`: Public clinical assessment UI
- `admin_app.py`: Auth-protected analytics dashboard
- `inference_db.py`: Inference logging and database access
- `admin_auth.py`: Admin authentication helpers
- `create_admin_user.py`: Utility to add admin users

### Model Training Scripts

- `logisticregression_only.py`: Logistic Regression training pipeline (Optuna + Youden threshold)
- `boostedtrees_ab.py`: Boosted-tree training pipeline for variant B
- Other training experiments: additional `*.py` model files in the repository

### Data and Artifacts

- `cleaned_data.csv`: Main training dataset
- `model_artifacts/`: Saved model bundles, metrics, thresholds, test predictions
- `data/`: Runtime data storage (for example local SQLite DB files)

## Quick Start (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train models

```powershell
# Variant A (required)
python logisticregression_only.py

# Variant B (optional, enables full A/B testing)
python boostedtrees_ab.py
```

Expected key outputs in `model_artifacts/`:

- `model_bundle.pkl`
- `metrics.json`
- `shap_explainer.pkl`
- `drift_baseline.pkl`

Optional variant B outputs:

- `boosted_model_bundle.pkl`
- `boosted_metrics.json`
- `boosted_shap_explainer.pkl`
- `boosted_drift_baseline.pkl`

### 4. Run services

Use separate terminals:

```powershell
# API
uvicorn app:app --host 0.0.0.0 --port 8000

# Public UI
streamlit run streamlit_app.py --server.port 8501

# Admin dashboard (internal only)
streamlit run admin_app.py --server.port 8502
```

## API Endpoints

Base URL: `http://localhost:8000`

- `GET /health`: Service and model availability status
- `POST /predict`: Risk prediction
- `POST /explain`: SHAP feature contributions for one prediction
- `POST /drift-check`: Per-feature z-score drift check for one payload
- `GET /drift-baseline`: Drift reference statistics
- `GET /inference-logs`: Recent logs (admin-oriented)
- `GET /analytics-summary`: Aggregate analytics (admin-oriented)

Interactive docs:

- `http://localhost:8000/docs`

### Example prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "GenHlth": 3,
    "HighBP": 1,
    "BMI": 30.0,
    "HighChol": 1,
    "Age": 9,
    "DiffWalk": 0,
    "HeartDiseaseorAttack": 0,
    "PhysHlth": 5,
    "Education": 4,
    "PhysActivity": 1
  }'
```

## Input Features

The default models use 10 features:

- `GenHlth`: 1 (Excellent) to 5 (Poor)
- `HighBP`: 0/1
- `BMI`: numeric, 10-80
- `HighChol`: 0/1
- `Age`: 1 (18-24) to 13 (80+)
- `DiffWalk`: 0/1
- `HeartDiseaseorAttack`: 0/1
- `PhysHlth`: 0-30
- `Education`: 1-6
- `PhysActivity`: 0/1

## Admin Access

- Admin credentials are stored in `admin_users.json`
- A default admin may be auto-created by auth utilities
- To create additional admin users:

```powershell
python create_admin_user.py
```

Important:

- Do not expose the admin dashboard publicly
- Restrict access to internal networks/VPN

## Database Configuration

- Local development: SQLite (automatic)
- Production: set `DATABASE_URL` to PostgreSQL connection string

PowerShell example:

```powershell
$env:DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

## Deployment

For deployment architecture and platform notes, see:

- `README_DEPLOY.md`

## Development Notes

- If API startup fails due to missing artifacts, retrain variant A first.
- If variant B artifacts are missing, `/predict` can fall back to variant A.
- Keep sensitive analytics endpoints behind authentication/network controls in production.
