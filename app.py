"""
Diabetes Risk Assessment API
FastAPI-based inference service with A/B testing, SHAP explainability,
confidence intervals, and drift monitoring.

Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
import hashlib
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_db import fetch_recent_logs, init_db, log_inference
from ml_core import feature_contract

logger = logging.getLogger("diabetes_api")

# Resolve packaged resources from the project directory, never from the caller's
# working directory, so the service behaves identically wherever it is launched.
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.pkl"
BOOSTED_BUNDLE_PATH = ARTIFACTS_DIR / "boosted_model_bundle.pkl"
SHAP_PATH_A = ARTIFACTS_DIR / "shap_explainer.pkl"
SHAP_PATH_B = ARTIFACTS_DIR / "boosted_shap_explainer.pkl"
DRIFT_BASELINE_A = ARTIFACTS_DIR / "drift_baseline.pkl"
DRIFT_BASELINE_B = ARTIFACTS_DIR / "boosted_drift_baseline.pkl"


def choose_variant(user_id: str) -> str:
    """Deterministic A/B assignment based on user_id hash.

    MD5 is used purely as a fast, stable bucketing function - it carries no
    security property here, and usedforsecurity=False states that explicitly
    (it also keeps this working under a FIPS-restricted build). The digest is
    unchanged, so existing A/B assignments are stable.
    """
    digest = hashlib.md5(user_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return "A" if int(digest[-2:], 16) % 2 == 0 else "B"


class DiabetesFeatures(BaseModel):
    """
    Clinical input features for diabetes risk assessment.

    Features mapped from BRFSS survey data:
    - GenHlth: General health (1=Excellent to 5=Poor)
    - HighBP: High blood pressure (0=No, 1=Yes)
    - BMI: Body Mass Index (10-80)
    - HighChol: High cholesterol (0=No, 1=Yes)
    - Age: Age category (1=18-24 to 13=80+)
    - DiffWalk: Difficulty walking (0=No, 1=Yes)
    - HeartDiseaseorAttack: Heart disease/MI history (0=No, 1=Yes)
    - PhysHlth: Poor physical health days in past 30 days (0-30)
    - Education: Education level (1-6)
    - PhysActivity: Physical activity in past 30 days (0=No, 1=Yes)
    """
    GenHlth: int = Field(..., ge=1, le=5, description="General health (1=Excellent to 5=Poor)")
    HighBP: int = Field(..., ge=0, le=1, description="High blood pressure (0=No, 1=Yes)")
    BMI: float = Field(..., ge=10, le=80, description="Body Mass Index")
    HighChol: int = Field(..., ge=0, le=1, description="High cholesterol (0=No, 1=Yes)")
    Age: int = Field(..., ge=1, le=13, description="Age category (1=18-24 to 13=80+)")
    DiffWalk: int = Field(..., ge=0, le=1, description="Difficulty walking (0=No, 1=Yes)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Heart disease/MI history")
    PhysHlth: int = Field(..., ge=0, le=30, description="Poor physical health days (0-30)")
    Education: int = Field(..., ge=1, le=6, description="Education level (1-6)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity (0=No, 1=Yes)")


class ServiceError(Exception):
    """Base for deliberate serving failures with a client-safe mapping."""

    status_code = 500
    client_detail = "Internal error. Contact the service operator with the request id."


class ArtifactUnavailableError(ServiceError):
    """A required serving artifact is missing or cannot be loaded.

    Mapped to 503: the process is alive but cannot serve predictions, which is
    what a load balancer or orchestrator needs to know.
    """

    status_code = 503
    client_detail = "Model artifact unavailable. The service cannot serve predictions."


class InferenceError(ServiceError):
    """Scoring failed after the model was successfully loaded."""

    status_code = 500
    client_detail = "Inference failed. Contact the service operator with the request id."


@lru_cache(maxsize=8)
def _load_bundle_cached(path_str: str, _mtime_ns: int, _size: int) -> dict:
    """Deserialize a bundle once per (path, mtime, size).

    Keyed on the file's stat so a replaced artifact is picked up without a
    restart, while readiness probes and repeated predictions do not
    re-deserialize the pickle every call.
    """
    return joblib.load(Path(path_str))


def load_model_bundle(path: Path) -> dict:
    """Load a model bundle, raising a deliberate ArtifactUnavailableError."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ArtifactUnavailableError(f"artifact not readable at {path}") from exc
    try:
        return _load_bundle_cached(str(path), stat.st_mtime_ns, stat.st_size)
    except Exception as exc:
        # joblib/pickle raise a wide range of types for a truncated or
        # version-incompatible artifact; all of them mean the same thing here.
        raise ArtifactUnavailableError(f"artifact at {path} could not be deserialized") from exc


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: prepare the inference log schema on startup.

    Replaces the deprecated startup event hook. Importing this module
    performs no database work; initialisation happens only when the application
    actually starts.
    """
    logger.info("Starting API.", extra={"event": "startup.begin"})
    init_db()
    logger.info(
        "Inference log schema ready.",
        extra={"event": "startup.complete", "artifacts_dir": str(ARTIFACTS_DIR)},
    )
    yield
    logger.info("Shutting down API.", extra={"event": "shutdown"})


app = FastAPI(
    title="Diabetes Risk Assessment API",
    description="Clinical decision support API for diabetes risk prediction",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ServiceError)
async def _service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Log the real cause server-side; return a sanitized body to the client.

    The client never receives repr(exc), a traceback, a filesystem path, a
    database credential or raw SQL error text - only a stable message and the
    request id needed to correlate with the server log.
    """
    request_id = str(uuid.uuid4())
    logger.exception(
        "Request failed: %s", type(exc).__name__,
        extra={
            "event": "request.failed",
            "request_id": request_id,
            "path": request.url.path,
            "error_type": type(exc).__name__,
            "status_code": exc.status_code,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.client_detail, "request_id": request_id},
    )


@app.get("/health")
def health() -> dict:
    """Liveness probe: is the process up?

    Deliberately cheap and always 200 while the process can answer. It mutates
    nothing, loads nothing and never fails on a dependency - use /ready to
    decide whether to send traffic. The response schema is unchanged.
    """
    return {
        "status": "ok",
        "service": "Diabetes Risk Assessment API",
        "model_bundle_exists": MODEL_BUNDLE_PATH.exists(),
        "boosted_bundle_exists": BOOSTED_BUNDLE_PATH.exists(),
    }


def _check_artifact(name: str, path: Path) -> dict:
    """Confirm one artifact actually deserializes. Cached, so probes are cheap.

    Reports a logical name and a coarse reason - never the artifact path, which
    a readiness probe has no business exposing.
    """
    try:
        load_model_bundle(path)
    except ArtifactUnavailableError:
        return {"name": name, "ok": False, "reason": "unavailable"}
    return {"name": name, "ok": True}


@app.get("/ready")
def ready(response: Response) -> dict:
    """Readiness probe: can this instance actually serve predictions?

    Checks that the primary model bundle deserializes and that the configured
    inference log is reachable. The database check runs against whatever is
    configured - with no DATABASE_URL that is local SQLite, so readiness never
    depends on an external PostgreSQL when local storage is a valid runtime.

    200 when predictions can be served, 503 otherwise. Reasons are coarse
    labels, never internal paths or driver error text.
    """
    checks: list[dict] = [_check_artifact("primary_model", MODEL_BUNDLE_PATH)]

    boosted = _check_artifact("boosted_model", BOOSTED_BUNDLE_PATH)
    # Variant B is optional: /predict already falls back to A when it is absent,
    # so it must not gate readiness.
    boosted["required"] = False
    checks.append(boosted)

    try:
        init_db()
        checks.append({"name": "inference_log", "ok": True})
    except Exception:
        logger.warning(
            "Readiness check could not reach the inference log.",
            exc_info=True,
            extra={"event": "ready.dependency_unavailable", "dependency": "inference_log"},
        )
        checks.append({"name": "inference_log", "ok": False, "reason": "unavailable"})

    required_failed = [c for c in checks if not c["ok"] and c.get("required", True) is not False]
    is_ready = not required_failed

    if not is_ready:
        response.status_code = 503
        logger.error(
            "Service is not ready.",
            extra={
                "event": "ready.not_ready",
                "failed": [c["name"] for c in required_failed],
            },
        )

    return {"status": "ready" if is_ready else "not_ready", "checks": checks}


@app.post("/predict")
def predict(
    payload: DiabetesFeatures,
    user_id: str = "anonymous",
    model_variant: str = "auto"
) -> dict:
    """
    Predict diabetes risk from clinical features.

    Parameters:
    - payload: Clinical feature values
    - user_id: Optional user identifier for A/B assignment
    - model_variant: 'auto' (A/B testing), 'A' (logistic regression), or 'B' (boosted trees)

    Returns:
    - Prediction result with probability and risk classification
    """
    # A/B variant selection
    selected_variant = choose_variant(user_id) if model_variant == "auto" else model_variant.upper()
    if selected_variant not in {"A", "B"}:
        raise HTTPException(status_code=400, detail="model_variant must be: auto, A, or B")

    # Select model bundle
    bundle_path = MODEL_BUNDLE_PATH if selected_variant == "A" else BOOSTED_BUNDLE_PATH
    fallback_used = False
    if selected_variant == "B" and not bundle_path.exists():
        bundle_path = MODEL_BUNDLE_PATH
        fallback_used = True

    request_id = str(uuid.uuid4())

    # Load model. load_model_bundle raises ArtifactUnavailableError (-> 503) and
    # the handler logs the real cause; the client never sees the artifact path.
    bundle = load_model_bundle(bundle_path)
    try:
        pipeline = bundle["pipeline"]
        threshold = float(bundle["threshold"])
        feature_columns = bundle["feature_columns"]
        model_name = bundle.get("model_name", "logistic_regression")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactUnavailableError(f"bundle at {bundle_path} is missing required keys") from exc

    # The bundle records the feature order it was trained on. Serving orders
    # columns from the canonical contract, so a bundle that disagrees would be
    # scored on silently misaligned columns - refuse instead.
    if list(feature_columns) != list(feature_contract.FEATURE_NAMES):
        raise ArtifactUnavailableError(
            f"bundle at {bundle_path} declares a feature order that differs from the contract"
        )

    # Score. Anything that goes wrong here is an inference failure, not a
    # validation problem: the payload already passed Pydantic.
    payload_dict = payload.model_dump()
    try:
        # Order the columns from the canonical contract, never from JSON key
        # order or Pydantic field order.
        input_df = feature_contract.order_columns(pd.DataFrame([payload_dict]))
        probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    except Exception as exc:
        raise InferenceError(f"scoring failed for variant {selected_variant}") from exc

    prediction = int(probability >= threshold)

    # Persist telemetry. This is analytics, not an audit record: streamlit_app.py
    # already states that logging must never break the user-facing result, and
    # the store is disposable gitignored runtime state. A write failure is
    # therefore degraded and logged, never allowed to discard a valid score.
    try:
        log_inference(
            request_id=request_id,
            model_variant=selected_variant,
            model_name=model_name,
            probability=probability,
            prediction=prediction,
            threshold=threshold,
            payload=payload_dict,
            # The value A/B bucketing was computed from. Only its digest is
            # stored, and it identifies an experiment assignment rather than a
            # person - the public UI sends a random per-session value and the
            # default here is the literal "anonymous".
            assignment_key=user_id,
        )
    except Exception:
        logger.warning(
            "Inference telemetry could not be persisted; returning the prediction anyway.",
            exc_info=True,
            extra={
                "event": "inference.persist_failed",
                "request_id": request_id,
                "model_variant": selected_variant,
                "model_name": model_name,
            },
        )
    else:
        logger.info(
            "Inference served.",
            extra={
                "event": "inference.served",
                "request_id": request_id,
                "model_variant": selected_variant,
                "model_name": model_name,
                "prediction": prediction,
                "fallback_to_a": fallback_used,
            },
        )

    return {
        "request_id": request_id,
        "model_variant": selected_variant,
        "model_name": model_name,
        "fallback_to_A": fallback_used,
        "prediction": prediction,
        "risk_category": "HIGH" if prediction == 1 else "LOW",
        "probability": round(probability, 6),
        "threshold": round(threshold, 6),
        "confidence_intervals": bundle.get("confidence_intervals"),
        "calibration": bundle.get("calibration"),
    }


@app.post("/explain")
def explain(
    payload: DiabetesFeatures,
    model_variant: str = "A",
) -> dict:
    """
    Get SHAP-based feature contribution explanation for a prediction.

    Returns per-feature SHAP values explaining why the model produced its prediction.
    """
    variant = model_variant.upper()
    if variant not in {"A", "B"}:
        raise HTTPException(status_code=400, detail="model_variant must be A or B")

    shap_path = SHAP_PATH_A if variant == "A" else SHAP_PATH_B
    bundle_path = MODEL_BUNDLE_PATH if variant == "A" else BOOSTED_BUNDLE_PATH

    if not shap_path.exists():
        raise HTTPException(status_code=404, detail=f"SHAP explainer not found for variant {variant}")

    try:
        shap_bundle = joblib.load(shap_path)
    except Exception as exc:
        raise ArtifactUnavailableError(f"SHAP explainer at {shap_path} is unreadable") from exc

    model_bundle = load_model_bundle(bundle_path)
    try:
        pipeline = model_bundle["pipeline"]
        threshold = float(model_bundle["threshold"])
        feature_columns = model_bundle["feature_columns"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactUnavailableError(f"bundle at {bundle_path} is missing required keys") from exc

    payload_dict = payload.model_dump()
    input_df = feature_contract.order_columns(pd.DataFrame([payload_dict]))

    probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= threshold)

    explainer = shap_bundle["explainer"]
    expected_value = shap_bundle["expected_value"]

    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 for binary

    feature_contributions = []
    for i, feat in enumerate(feature_columns):
        feature_contributions.append({
            "feature": feat,
            "value": float(input_df.iloc[0][feat]),
            "shap_value": float(shap_values[0][i]),
        })

    feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "model_variant": variant,
        "probability": round(probability, 6),
        "prediction": prediction,
        "risk_category": "HIGH" if prediction == 1 else "LOW",
        "expected_value": round(float(expected_value), 6),
        "feature_contributions": feature_contributions,
    }


@app.post("/drift-check")
def drift_check(
    payload: DiabetesFeatures,
    model_variant: str = "A",
) -> dict:
    """
    Check if a single input shows signs of data drift relative to training distribution.

    Returns per-feature z-scores and overall drift flag.
    """
    variant = model_variant.upper()
    if variant not in {"A", "B"}:
        raise HTTPException(status_code=400, detail="model_variant must be A or B")

    drift_path = DRIFT_BASELINE_A if variant == "A" else DRIFT_BASELINE_B
    if not drift_path.exists():
        raise HTTPException(status_code=404, detail=f"Drift baseline not found for variant {variant}")

    try:
        baseline = joblib.load(drift_path)
    except Exception as exc:
        raise ArtifactUnavailableError(f"drift baseline at {drift_path} is unreadable") from exc

    payload_dict = payload.model_dump()

    # Handle two baseline formats:
    # Format A (LR): {feature_name: {mean, std, ...}, ...}
    # Format B (XGB): {feature_columns: [...], means: {...}, stds: {...}, ...}
    if "feature_columns" in baseline:
        feature_cols = list(baseline["feature_columns"])
        means = baseline["means"]
        stds = baseline["stds"]
    else:
        feature_cols = list(baseline.keys())
        means = {feat: baseline[feat]["mean"] for feat in feature_cols}
        stds = {feat: baseline[feat]["std"] for feat in feature_cols}

    drift_details = []
    drift_flags = 0

    for feat in feature_cols:
        val = float(payload_dict.get(feat, 0))
        mean = means[feat]
        std = stds[feat]
        z_score = (val - mean) / std if std > 0 else 0.0
        is_outlier = abs(z_score) > 3.0

        drift_details.append({
            "feature": feat,
            "value": val,
            "training_mean": round(mean, 4),
            "training_std": round(std, 4),
            "z_score": round(z_score, 4),
            "is_outlier": is_outlier,
        })

        if is_outlier:
            drift_flags += 1

    return {
        "model_variant": variant,
        "drift_detected": drift_flags > 0,
        "outlier_count": drift_flags,
        "total_features": len(feature_cols),
        "feature_drift": drift_details,
    }


@app.get("/drift-baseline")
def get_drift_baseline(model_variant: str = "A") -> dict:
    """Return training-set statistics used for drift detection."""
    variant = model_variant.upper()
    drift_path = DRIFT_BASELINE_A if variant == "A" else DRIFT_BASELINE_B
    if not drift_path.exists():
        raise HTTPException(status_code=404, detail="Drift baseline not found")
    try:
        baseline = joblib.load(drift_path)
    except Exception as exc:
        raise ArtifactUnavailableError(f"drift baseline at {drift_path} is unreadable") from exc

    return {"model_variant": variant, "baseline": baseline}


class LogStoreUnavailableError(ServiceError):
    """The inference log could not be read."""

    status_code = 503
    client_detail = "Inference log unavailable. Analytics cannot be served right now."


def _read_logs(limit: int) -> list[dict]:
    """Read recent inference logs, mapping a store failure to a safe 503.

    Raw driver text - which can carry a connection string, host or SQL - must
    never reach the client, so the real error is logged by the ServiceError
    handler and the client gets a fixed message plus a request id.
    """
    try:
        return fetch_recent_logs(limit=limit)
    except Exception as exc:
        raise LogStoreUnavailableError("inference log read failed") from exc


@app.get("/inference-logs")
def inference_logs(limit: int = 100) -> dict:
    """
    Retrieve recent inference logs (admin endpoint).

    Parameters:
    - limit: Maximum number of logs to return (1-1000)
    """
    safe_limit = max(1, min(limit, 1000))
    rows = _read_logs(safe_limit)
    return {"count": len(rows), "logs": rows}


@app.get("/analytics-summary")
def analytics_summary(limit: int = 1000) -> dict:
    """
    Get aggregated analytics summary (admin endpoint).

    Parameters:
    - limit: Number of recent logs to aggregate (1-10000)
    """
    safe_limit = max(1, min(limit, 10000))
    rows = _read_logs(safe_limit)

    if not rows:
        return {
            "count": 0,
            "positive_rate": 0.0,
            "average_probability": 0.0,
            "by_variant": {},
            "by_model": {},
        }

    total = len(rows)
    positive_count = sum(int(item["prediction"]) for item in rows)
    avg_probability = sum(float(item["probability"]) for item in rows) / total

    by_variant: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for item in rows:
        variant = item.get("model_variant", "unknown")
        model = item.get("model_name", "unknown")
        by_variant[variant] = by_variant.get(variant, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1

    return {
        "count": total,
        "positive_rate": positive_count / total,
        "average_probability": avg_probability,
        "by_variant": by_variant,
        "by_model": by_model,
    }
