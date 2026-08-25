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

Canonical tested Python version: **3.11**. CI and the `Dockerfile` both use it,
and the `Dockerfile` installs from `requirements.lock` rather than the
loose ranges, so the container matches the tested runtime exactly.

The model bundles in `model_artifacts/` are pickles that record the library
version that wrote them, so `requirements.txt` pins `scikit-learn==1.8.0` and
`xgboost==3.0.4` exactly. Installing other versions still works but raises
`InconsistentVersionWarning` and an XGBoost model-format warning, and would
require retraining to clear.

### 1. Create and activate virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Everyday development - resolves fresh versions within the ranges declared in
`requirements.txt`:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Reproducible install - exact, fully pinned transitive dependency set:

```powershell
python -m pip install -r requirements.lock -r requirements-dev.lock
```

`requirements.lock` is generated, not hand-edited. It is a *universal*
lock: it carries environment markers, so one file installs correctly on
Windows, on Linux (the Dockerfile target) and on macOS. Regenerate it after
any change to `requirements.txt`:

```powershell
python -m pip install uv
uv pip compile --universal --python-version 3.11 --output-file requirements.lock requirements.txt
```

Development and test tooling is locked the same way, constrained by the
production lock so every shared transitive package resolves to the identical
version. This is what CI installs. Regenerate it after changing
`requirements-dev.txt`:

```powershell
uv pip compile --universal --python-version 3.11 --constraint requirements.lock --output-file requirements-dev.lock requirements-dev.txt
```

### 3. Run the test suite and linter

The suite is self-contained: no network, no PostgreSQL, and no retraining. It
uses a temporary SQLite database, so it never touches `data/`.

```powershell
python -m pytest -q
ruff check .
```

`tests/test_artifact_compatibility.py` additionally proves the committed model
bundles still load under the locked dependency set without raising
`InconsistentVersionWarning` or XGBoost's model-format warning. It fails if a
dependency bump makes an artifact incompatible, so the choice becomes explicit:
revert the bump, or retrain and recommit the artifacts.

`ruff check .` lints a **governed subset** of the repository - `app.py`,
`inference_db.py`, `conftest.py` and `tests/`. The unmaintained training
experiments are out of scope; `ruff.toml` records the exact boundary and why.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`,
against the same canonical Python 3.11 and the same committed lockfiles:

| Job | What it proves |
| --- | --- |
| Lint, tests and artifact compatibility | `ruff check .`, the full pytest suite, model artifacts load warning-free, `compileall` plus an import smoke, and the working tree is still clean afterwards |
| Docker build and health smoke | the image builds, reports Python 3.11, answers `/health`, serves a real `/predict`, and logs no serialization-version warning |
| Dependency vulnerability audit | `pip-audit` over both lockfiles |

CI never contacts PostgreSQL or Neon: `DATABASE_URL` is forced empty so both the
test suite and the container fall back to local SQLite.

Dependency updates are proposed weekly by Dependabot
(`.github/dependabot.yml`). Nothing auto-merges, and `scikit-learn` / `xgboost`
are deliberately excluded because bumping either invalidates the committed
model artifacts.

### 4. Train models

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

### 5. Run services

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
