# Diabetes Prediction System

[![CI](https://github.com/KiyotakaShinichi/Diabetes_Prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/KiyotakaShinichi/Diabetes_Prediction/actions/workflows/ci.yml)

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

### Shared evaluation core

`ml_core/` holds the model-agnostic evaluation layer that **both** maintained
pipelines use, so there is one owner per behaviour rather than two drifting
copies:

| Module | Owns |
| --- | --- |
| `ml_core/evaluation.py` | the metric set and its zero-division policy |
| `ml_core/bootstrap.py` | percentile bootstrap confidence intervals |
| `ml_core/thresholds.py` | Youden's J threshold selection |

It is pure and import-safe: nothing in it reads a dataset, fits a model, writes
a file, prints, plots, or mutates global state (including warnings filters and
the global NumPy RNG). Training orchestration - data loading, the Optuna study,
model fitting, calibration, SHAP and artifact writing - stays in the pipeline
scripts. `compute_drift_baseline` is deliberately *not* shared: the two
pipelines emit different baseline schemas and `/drift-check` branches on that
difference.

`compute_youden_threshold` guarantees a **finite** serving threshold within
`[0, 1]`. `sklearn.metrics.roc_curve` prepends a synthetic infinite threshold,
and the previous implementation could select it whenever no real cut-point beat
random - yielding a threshold that classifies every patient as negative. Only
finite candidates are considered now; a single-class target is rejected rather
than silently given a neutral value. Where the old code already returned a
finite threshold, the selected value is unchanged, which the test suite proves
against a pinned copy of the old implementation.

Full training remains offline and reproducible via the commands below. It is
**not** run in normal CI - CI exercises the shared utilities and verifies that
the committed artifacts still load, rather than retraining models.

### Data and Artifacts

- `cleaned_data.csv`: Main training dataset
- `model_artifacts/`: Saved model bundles, metrics, thresholds, test predictions
- `data/`: Runtime data storage (for example local SQLite DB files)

## Quick Start (Windows PowerShell)

Canonical tested Python version: **3.11** (verified on **3.11.16**). CI, the
`Dockerfile`, `render.yaml` and the devcontainer all use it,
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

### 3. Configure the environment (optional for local use)

```powershell
Copy-Item .env.example .env
```

`.env.example` documents every variable the code actually reads:
`DATABASE_URL` (leave empty for local SQLite), plus `ADMIN_USERNAME` and
`ADMIN_PASSWORD` for seeding the admin dashboard. Nothing auto-loads `.env` -
export the variables or pass them to Docker/Render. `.env` is gitignored;
`.env.example` contains placeholders only and no real credentials.

Admin authentication has **not** been hardened yet - if the admin variables are
unset, the code still falls back to a built-in default account. Set them.

### 4. Run the test suite and linter

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

`render.yaml` pins `PYTHON_VERSION: "3.11.16"` and builds with
`pip install -r requirements.lock`, and the devcontainer installs the same
locks, so every deployment surface matches the tested runtime.
`tests/test_config_contracts.py` parses those files and fails if one drifts.

Dependency updates are proposed weekly by Dependabot
(`.github/dependabot.yml`). Nothing auto-merges, and `scikit-learn` / `xgboost`
are deliberately excluded because bumping either invalidates the committed
model artifacts.

### 5. Train models

```powershell
# Variant A (required)
python logisticregression_only.py

# Variant B (optional, enables full A/B testing)
python boostedtrees_ab.py
```

Both read the committed `cleaned_data.csv` and write to `model_artifacts/`,
resolved from the project directory - so these commands work from any working
directory, on any machine. Override either location explicitly:

```powershell
python logisticregression_only.py --data-path D:/data/my_cohort.csv --artifacts-dir D:/artifacts
python logisticregression_only.py --help
```

The archived single-model experiments (`qsvm.py`, `xgboost_only.py`,
`logreg+clustering.py` and others) take the same `--data-path` and a
`--results-dir`, and `--help` works without installing their optional
plotting/clustering dependencies:

```powershell
python qsvm.py --help
python qsvm.py --data-path C:/path/to/cleaned_data_upd.csv --results-dir ./out
```

They expect `cleaned_data_upd.csv`, a renamed variant with the target column
`DiabetesStatus`, which is **not committed** - the repository ships
`cleaned_data.csv` (target `Diabetes_binary`) instead. The two schemas are not
interchangeable, so supply your own copy with `--data-path`. Results default to
the gitignored `experiment_results/`.

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

### 6. Run services

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

- `GET /health`: Liveness - is the process up?
- `GET /ready`: Readiness - can this instance actually serve predictions?
- `POST /predict`: Risk prediction
- `POST /explain`: SHAP feature contributions for one prediction
- `POST /drift-check`: Per-feature z-score drift check for one payload
- `GET /drift-baseline`: Drift reference statistics
- `GET /inference-logs`: Recent logs (admin-oriented)
- `GET /analytics-summary`: Aggregate analytics (admin-oriented)

Interactive docs:

- `http://localhost:8000/docs`

### Probes

`/health` is a **liveness** probe: cheap, always `200` while the process can
answer, mutates nothing and loads nothing. It never fails because a dependency
is down, so it is not a signal about whether to send traffic. Its response
schema is unchanged.

`/ready` is a **readiness** probe: it confirms the primary model bundle actually
deserializes and that the configured inference log is reachable, then returns
`200` with `{"status": "ready", "checks": [...]}` or `503` with
`{"status": "not_ready", ...}`.

- Variant B is optional - `/predict` already falls back to variant A when the
  boosted bundle is absent - so a missing variant B does not make the service
  unready.
- The database check runs against whatever is configured. With no
  `DATABASE_URL` that is local SQLite, so readiness never depends on an
  external PostgreSQL when local storage is a valid runtime.
- Bundles are cached by `(path, mtime, size)`, so probes reuse the loaded model
  rather than re-deserializing the pickle on every call, and a replaced
  artifact is still picked up without a restart.

### Errors and logging

Failures are mapped deliberately rather than collapsing into a generic 500:

| Situation | Status |
| --- | --- |
| Invalid feature values or missing fields | `422` (Pydantic) |
| Unknown `model_variant` selector | `400` |
| A required model artifact is missing or unreadable | `503` |
| The inference log cannot be read | `503` |
| Scoring itself fails unexpectedly | `500` |

Error bodies are sanitized: they carry a fixed message and a `request_id`,
never `repr(exc)`, a traceback, a filesystem path, a database credential or raw
SQL text. The real cause - including the traceback - is logged server-side
against the same `request_id`, so an operator can correlate a client report
with the exact failure.

Operational logging uses the Python standard library only; **no external
error-tracking provider is required or integrated**. Events carry structured
context via `extra=` (`event`, `request_id`, `model_variant`, `model_name`), so
a prediction can be correlated across its response, its stored row and the log
line. Clinical feature payloads are deliberately not logged.

**Telemetry failure policy**: inference logging is analytics, not an audit
record. If the model scores successfully but the log write fails, the
prediction is still returned and the failure is logged at `WARNING`. This
matches the behaviour `streamlit_app.py` already documented - logging must
never break the user-facing result - and the store is disposable, gitignored
runtime state.

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

**There is no default administrator.** The dashboard fails closed: if no
credential provider is configured, every login is refused and the login page
says so. No account is ever created implicitly, and importing any module never
writes credentials.

Configure exactly one of the two providers.

**Environment provider** - for Render, containers, any stateless deployment.
Both variables are required; setting only one is a configuration error and
still fails closed rather than falling back to the other provider.

```powershell
$env:ADMIN_USERNAME = "alice"
$env:ADMIN_PASSWORD = "<a real secret, managed as a platform secret>"
```

**Runtime user store** - for local use. Create an account explicitly:

```powershell
python create_admin_user.py --username alice
```

The password is prompted for with `getpass`; it is never accepted as a
command-line argument, which would leak it into shell history and the process
list. It is stored as a salted PBKDF2-HMAC-SHA256 hash (600,000 iterations) in
`data/admin_users.json`. Minimum length is 8 characters.

That file is **gitignored and not committed**, and is created only by the
command above. A malformed or missing store means "no file-backed users" - it
never means "create a default".

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
