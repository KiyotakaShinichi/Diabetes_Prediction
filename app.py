"""
Diabetes Risk Assessment API
FastAPI-based inference service with A/B testing support.

Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from pathlib import Path
import hashlib
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference_db import fetch_recent_logs, init_db, log_inference


MODEL_BUNDLE_PATH = Path("model_artifacts/model_bundle.pkl")
BOOSTED_BUNDLE_PATH = Path("model_artifacts/boosted_model_bundle.pkl")


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
    }


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
