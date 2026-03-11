"""
Diabetes Risk Assessment API
FastAPI-based inference service with A/B testing, SHAP explainability,
confidence intervals, and drift monitoring.

Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from pathlib import Path
import hashlib
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference_db import fetch_recent_logs, init_db, log_inference


MODEL_BUNDLE_PATH = Path("model_artifacts/model_bundle.pkl")
BOOSTED_BUNDLE_PATH = Path("model_artifacts/boosted_model_bundle.pkl")
SHAP_PATH_A = Path("model_artifacts/shap_explainer.pkl")
SHAP_PATH_B = Path("model_artifacts/boosted_shap_explainer.pkl")
DRIFT_BASELINE_A = Path("model_artifacts/drift_baseline.pkl")
DRIFT_BASELINE_B = Path("model_artifacts/boosted_drift_baseline.pkl")


def choose_variant(user_id: str) -> str:
    """Deterministic A/B assignment based on user_id hash."""
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
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


app = FastAPI(
    title="Diabetes Risk Assessment API",
    description="Clinical decision support API for diabetes risk prediction",
    version="2.0.0",
)


def load_model_bundle(path: Path) -> dict:
    """Load model bundle from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}")
    return joblib.load(path)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize database on startup."""
    init_db()


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Diabetes Risk Assessment API",
        "model_bundle_exists": MODEL_BUNDLE_PATH.exists(),
        "boosted_bundle_exists": BOOSTED_BUNDLE_PATH.exists(),
    }


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

    # Load model
    try:
        bundle = load_model_bundle(bundle_path)
        pipeline = bundle["pipeline"]
        threshold = float(bundle["threshold"])
        feature_columns = bundle["feature_columns"]
        model_name = bundle.get("model_name", "logistic_regression")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    # Prepare input
    payload_dict = payload.model_dump()
    input_df = pd.DataFrame([payload_dict])
    input_df = input_df[feature_columns]

    # Predict
    probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= threshold)
    request_id = str(uuid.uuid4())

    # Log inference
    log_inference(
        request_id=request_id,
        model_variant=selected_variant,
        model_name=model_name,
        probability=probability,
        prediction=prediction,
        threshold=threshold,
        payload=payload_dict,
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
        model_bundle = load_model_bundle(bundle_path)
        pipeline = model_bundle["pipeline"]
        threshold = float(model_bundle["threshold"])
        feature_columns = model_bundle["feature_columns"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Load error: {exc}") from exc

    payload_dict = payload.model_dump()
    input_df = pd.DataFrame([payload_dict])[feature_columns]

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

    baseline = joblib.load(drift_path)
    payload_dict = payload.model_dump()

    # Handle two baseline formats:
    # Format A (LR): {feature_name: {mean, std, ...}, ...}
    # Format B (XGB): {feature_columns: [...], means: {...}, stds: {...}, ...}
    if "feature_columns" in baseline:
        feature_cols = baseline["feature_columns"]
        get_mean = lambda f: baseline["means"][f]
        get_std = lambda f: baseline["stds"][f]
    else:
        feature_cols = list(baseline.keys())
        get_mean = lambda f: baseline[f]["mean"]
        get_std = lambda f: baseline[f]["std"]

    drift_details = []
    drift_flags = 0

    for feat in feature_cols:
        val = float(payload_dict.get(feat, 0))
        mean = get_mean(feat)
        std = get_std(feat)
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
    baseline = joblib.load(drift_path)
    return {"model_variant": variant, "baseline": baseline}


@app.get("/inference-logs")
def inference_logs(limit: int = 100) -> dict:
    """
    Retrieve recent inference logs (admin endpoint).
    
    Parameters:
    - limit: Maximum number of logs to return (1-1000)
    """
    safe_limit = max(1, min(limit, 1000))
    rows = fetch_recent_logs(limit=safe_limit)
    return {"count": len(rows), "logs": rows}


@app.get("/analytics-summary")
def analytics_summary(limit: int = 1000) -> dict:
    """
    Get aggregated analytics summary (admin endpoint).
    
    Parameters:
    - limit: Number of recent logs to aggregate (1-10000)
    """
    safe_limit = max(1, min(limit, 10000))
    rows = fetch_recent_logs(limit=safe_limit)

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
