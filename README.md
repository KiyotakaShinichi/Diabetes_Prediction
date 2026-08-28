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

One serving path. The FastAPI service is the only component that loads a model
and scores a request; the public UI is presentation over an HTTP client.

```
Public Streamlit (8501)
        |
        |  POST /predict, POST /explain      DIABETES_API_BASE_URL
        v
FastAPI inference API (8000)
        |
        +-- Pydantic validation of every field
        +-- canonical feature ordering, refusing a bundle that disagrees
        +-- model bundle loading and caching
        +-- deterministic A/B variant routing
        +-- thresholding and risk classification
        +-- request correlation ids and sanitised errors
        +-- inference telemetry (the single writer)
        v
model_artifacts/  ->  inference log (SQLite, or PostgreSQL via DATABASE_URL)

Admin Streamlit (8502, login-protected) reads the inference log and the
committed artifacts directly. It is a monitoring surface, not an API client.
```

The public app does **not** load `model_bundle.pkl`, call `predict_proba`,
compare against a threshold, choose a model variant, or write to the inference
log. Those decisions belong to the API and are read back off its response, so
the two cannot drift apart. `tests/test_serving_convergence.py` enforces this
against the AST rather than by convention.

### Configuring the public UI

| Variable | Default | Purpose |
| --- | --- | --- |
| `DIABETES_API_BASE_URL` | `http://127.0.0.1:8000` | Where the public UI sends predictions |
| `DIABETES_API_TIMEOUT_SECONDS` | `15` | Per-request timeout |

`DIABETES_API_BASE_URL` accepts a full `http://` or `https://` URL, or a bare
`host:port` with no scheme. The scheme-less form exists because Render's
Blueprint resolves `fromService` / `property: hostport` to `diabetes-api:10000`
over its private network, where there is no TLS terminator; the client
normalises that to `http://`. A malformed value raises rather than falling back
to loopback, so a misconfigured deployment fails visibly instead of starting up
and failing every submission.

On Render the wiring is a service reference, not a literal hostname, so it
survives a Blueprint redeploy:

```yaml
      - key: DIABETES_API_BASE_URL
        fromService:
          name: diabetes-api
          type: web
          property: hostport
```

## Frontend

Two Streamlit applications share one presentation layer. Neither renders styling
of its own beyond what `ui/theme.py` owns, and the global theme is pinned in
`.streamlit/config.toml` rather than left to the visitor's browser.

### Public assessment

An educational risk estimator, not a diagnostic tool. It asks the ten questions
named by the canonical feature contract, and nothing is filled in on the
visitor's behalf: every choice starts on a placeholder and both numeric fields
start empty, so an untouched form cannot produce a result. Submitting an
incomplete form names the questions still outstanding.

The result leads with a single quantity - the estimated risk - followed by what
that estimate does and does not mean. The classification threshold is presented
as model metadata in a details panel rather than as a second number competing
with the estimate. Held-out precision, recall and ROC-AUC are shown from the
committed metrics, labelled as dataset-level behaviour rather than as confidence
in any individual estimate.

### Explainability

Each answer's SHAP contribution is listed with the question as the visitor read
it, the answer they gave, the direction of the effect stated in words, and its
relative strength. Direction never depends on interpreting a colour. When no
SHAP explainer is deployed, the panel says so instead of disappearing.

### Admin monitoring

A separate, login-protected dashboard covering inference volume, the A/B split
across model variants, committed evaluation metrics with bootstrap confidence
intervals, and a drift comparison against the training baseline for either
variant. It fails closed: with no authentication provider configured, no login
can succeed and the dashboard says so.

### Model provenance

The details panel reports the serving variant, model family, feature count and
threshold, along with how many artifacts are inventoried by SHA-256 checksum.
It also states plainly that the training lineage of the committed models is
unknown - they predate this project's provenance system, and reconstructing a
training run for them would be fabrication. Integrity is verified in CI; history
is not claimed.

### Non-diagnostic purpose

The tool estimates statistical risk from survey-style inputs. It uses no
laboratory values, it does not diagnose any condition, and it does not instruct
anyone to obtain a specific medical test. The boundary is stated above the form,
before any information is entered, and restated with the result.

## Repository Overview

### Core Applications

- `app.py`: FastAPI inference API (`/predict`, `/explain`, drift and analytics endpoints)
- `streamlit_app.py`: Public risk-assessment UI (page orchestration)
- `admin_app.py`: Auth-protected analytics dashboard (page orchestration)
- `ui/`: Shared presentation layer for both Streamlit apps
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

### Training lifecycle

Both maintained pipelines are composed of independently callable stages rather
than one long script, so any stage can be exercised on its own:

```
prepare -> optimize -> fit -> evaluate -> calibrate -> explain -> persist -> attest
```

`main()` in each pipeline is orchestration only. Concretely:

| Stage | Logistic regression | Boosted trees |
| --- | --- | --- |
| prepare | `prepare_training_data` - loads, validates against the feature contract, splits 60/20/20 stratified | same |
| optimize | `optimize_hyperparameters(splits, n_trials=...)` | same |
| fit | `fit_final_pipeline` (StandardScaler + LogisticRegression) | `fit_final_model` (XGBClassifier, no scaler) |
| threshold | `select_threshold` - Youden's J on the validation set | same |
| calibrate | `calibrate_pipeline` (Platt, cv=5) | `calibrate_model` |
| explain | `build_shap_explainer` (LinearExplainer) | `build_shap_explainer` (TreeExplainer) |
| drift | `build_logistic_drift_baseline` (schema A) | `build_boosted_drift_baseline` (schema B) |
| persist | `write_training_outputs(artifacts_dir, ...)` | same |
| attest | `emit_provenance(...)` - written **last** | same |

The two are deliberately not merged: the boosted variant has no scaler, uses a
different SHAP explainer and emits a different drift schema.

`ml_core/training.py` owns the genuinely shared parts - dataset loading, schema
validation, the split, threshold-based evaluation and atomic JSON writing.
`validate_training_dataset` checks the data actually provides what the feature
contract promises: target present, binary, both classes; every canonical feature
present, not duplicated, not entirely null, numeric or coercible.

**Full production optimization is not run in CI.** The real runs use 100 trials
(logistic) and 50 (boosted) over the full dataset and stay an offline operation.
What CI does run is a set of **small real model fits** on a ~240-row synthetic
fixture, which exercise the whole lifecycle end to end in seconds and verify the
resulting manifest with the provenance verifier.

> Those smoke runs verify **plumbing, not model quality**. They fit a few
> hundred synthetic rows with a two-trial budget; their metrics are not evidence
> about the committed models and are never compared against them.

A real training run emits a full provenance manifest automatically as its last
step - see the provenance section above.

### Provenance and artifact integrity

**Artifact integrity is not proof of historical training provenance.** The two
are tracked separately and deliberately never conflated.

**Current committed artifacts - integrity-attested, lineage incomplete.**
`provenance/legacy_artifact_attestation.json` is an inventory of the artifacts
in `model_artifacts/` as they exist today: SHA256, byte size, the library
versions read out of each file, and which ones the serving code actually loads.
Those artifacts predate this system, so no run record was kept for them. Every
historical field - `producer_git_sha`, `training_run_id`,
`training_dataset_sha256`, `training_started_at`, `training_configuration`,
`training_environment` - is an explicit `null`. Reconstructing them from file
timestamps or from what the current code happens to do would be fabrication, so
the repository does not claim that HEAD produced these models, and the current
dataset hash is labelled as *today's* dataset rather than the training input.

The attestation also separates the eight artifacts serving code loads from four
that nothing loads - including `diabetes_model.pkl` and `scaler.pkl`, which
carry scikit-learn **1.7.1**, older than the served bundles. A test cross-checks
every `required_for_serving` flag against the actual serving source, so a dead
artifact cannot quietly become a dependency.

**Future training runs - full provenance, automatically.** Both maintained
pipelines now emit a `training_run` manifest
(`model_artifacts/training_manifest.json` and `boosted_training_manifest.json`)
as their **last** step, after every artifact and the metrics file are on disk,
written atomically via a temp file and `os.replace`. A run that fails part way
leaves no manifest at all. Each records the dataset fingerprint, the ordered
feature schema and its hash, the git commit **and whether the tree was dirty**,
hashes of the exact training source files, the installed package versions and
lockfile hash, the full training configuration (seed, splits, Optuna sampler and
trial count, best hyperparameters, calibration, threshold method and selected
threshold, bootstrap settings), the evaluation metrics, and a hashed inventory
of every artifact produced.

Verify any manifest, or all of them:

```powershell
python tools/verify_provenance.py
python tools/build_artifact_attestation.py --check
```

Both exit non-zero on any mismatch, and CI runs them on every push. Regenerate
the attestation - only when the artifacts legitimately change - with:

```powershell
python tools/build_artifact_attestation.py
```

Manifests contain no absolute developer paths, no environment variables and no
secrets; the verifier rejects a manifest that carries an absolute path.

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

Dependency ownership is intentionally split between human-maintained inputs and
generated locks; there is no second package manifest:

| File | Owner and purpose |
| --- | --- |
| `requirements.txt` | human-maintained direct runtime requirements |
| `requirements-dev.txt` | human-maintained direct development/CI requirements |
| `requirements.lock` | generated, exact universal runtime resolution |
| `requirements-dev.lock` | generated, exact universal development resolution constrained by the runtime lock |

For a clean or CI-equivalent environment, install only the committed locks:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -r requirements-dev.lock
python -m pip check
```

The direct source files are for intentional dependency changes and local
resolution experiments:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
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
python tools/verify_dependency_contract.py
```

The verifier checks that all four files are tracked and non-empty, every direct
dependency (including the public client's `requests` transport) is pinned within
its declared range, shared runtime/dev pins agree, and each lock identifies its
canonical input. A separate `uv.lock` or Poetry/Pipenv manifest is deliberately
not generated: pip requirements plus `uv pip compile` are the operational
contract, and adding another lock would create redundant dependency truth.

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

### 4. Run quality gates

The suite is self-contained: no network, no PostgreSQL, and no retraining. It
uses a temporary SQLite database, so it never touches `data/`.

```powershell
python tools/verify_dependency_contract.py
python -m pytest -q
ruff check .
python -m mypy
coverage run -m pytest -q
coverage report
python -m pytest -q tests/test_training_smoke.py
python tools/verify_provenance.py
python tools/build_artifact_attestation.py --check
```

`python -m mypy` checks `ml_core/`, `experiment_config.py`, and `tools/` as
declared in `pyproject.toml`. Coverage uses branch measurement over the same
stable owned module boundary and enforces a conservative 90% combined ratchet;
historical experiments are excluded instead of being counted as artificial 0%.

`tests/test_artifact_compatibility.py` additionally proves the committed model
bundles still load under the locked dependency set without raising
`InconsistentVersionWarning` or XGBoost's model-format warning. It fails if a
dependency bump makes an artifact incompatible, so the choice becomes explicit:
revert the bump, or retrain and recommit the artifacts.

`ruff check .` governs maintained runtime modules, shared UI and ML modules,
tools, tests, and both maintained training pipelines. Historical root-level
studies remain narrowly excluded; [experiments/README.md](experiments/README.md)
records every script's status, expected dataset, output location, and known
reproducibility hazards.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`,
against the same canonical Python 3.11 and the same committed lockfiles:

| Job | What it proves |
| --- | --- |
| Lint and typecheck | dependency inputs and locks agree, `ruff check .` governs maintained code, and mypy checks the owned stable scope |
| Tests, artifacts, and training smoke | artifact compatibility and attestation, the complete pytest suite partitioned to expose deterministic mini-training, the 90% coverage ratchet, a CPU-only deep-learning training smoke, `compileall`, import safety, and a clean tree |
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
the gitignored `experiment_results/`. Their maintained/historical/dead
classification and known hazards are inventoried in
[experiments/README.md](experiments/README.md).

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
# API - start this first; the public UI depends on it
uvicorn app:app --host 0.0.0.0 --port 8000

# Public UI - reaches the API at http://127.0.0.1:8000 with no configuration
streamlit run streamlit_app.py --server.port 8501

# Admin dashboard (internal only)
streamlit run admin_app.py --server.port 8502
```

Or build and smoke the API container from the same production lock:

```powershell
docker build --tag diabetes-api:local .
docker run --rm --publish 8000:8000 --env DATABASE_URL= diabetes-api:local
```

The public UI cannot produce an estimate without the API running: it performs
no local inference. To point it somewhere else, set `DIABETES_API_BASE_URL`
before launching it.

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

## Project History

`CHANGELOG.md` records the engineering milestones, newest first, each citing the
commit that completed it. No version has been released and no tag exists, so
every entry sits under `Unreleased`. `experiments/README.md` is the companion
inventory for the archival research scripts that sit outside the lint and
typecheck boundary.

## Development Notes

- If API startup fails due to missing artifacts, retrain variant A first.
- If variant B artifacts are missing, `/predict` can fall back to variant A.
- Keep sensitive analytics endpoints behind authentication/network controls in production.
