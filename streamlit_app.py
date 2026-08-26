"""
Diabetes Risk Assessment
Educational risk-estimation tool. Not a diagnostic device.

Page orchestration only: page config, artifact loading, scoring, monitoring and
the order the sections appear in. Every section is rendered by ui.public_components,
which never loads a model or scores a request.

Run with: streamlit run streamlit_app.py
"""

import json
import uuid
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from inference_db import log_inference
from ml_core import feature_contract
from ui import public_components, theme

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.pkl"

SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.pkl"

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

ATTESTATION_PATH = PROJECT_ROOT / "provenance" / "legacy_artifact_attestation.json"

#: The served variant behind this UI. Variant B is reachable through the API.
MODEL_VARIANT = "A"

#: Where the completed assessment lives between reruns.
RESULT_KEY = "assessment_result"


@st.cache_resource
def load_model():
    if not MODEL_BUNDLE_PATH.exists():
        return None, None, None, None, None
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    return (
        bundle["pipeline"],
        float(bundle["threshold"]),
        bundle["feature_columns"],
        bundle.get("model_name", "unknown"),
        bundle.get("confidence_intervals"),
    )


@st.cache_resource
def load_shap_explainer():
    if not SHAP_EXPLAINER_PATH.exists():
        return None, None, None
    shap_bundle = joblib.load(SHAP_EXPLAINER_PATH)
    return (
        shap_bundle["explainer"],
        shap_bundle["expected_value"],
        shap_bundle["feature_names"],
    )


@st.cache_data
def load_evaluation_metrics() -> dict:
    """Held-out test metrics as committed by training. Never recomputed here."""
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_artifact_attestation() -> dict:
    """Integrity record for the committed artifacts, or {} when absent.

    Deliberately read-only and deliberately incomplete: the committed models
    predate the provenance system, so this file states what is observable now
    and records the rest as explicitly unknown. The UI must not present more
    than it says.
    """
    if not ATTESTATION_PATH.exists():
        return {}

    with open(ATTESTATION_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def score(pipeline, threshold: float, payload: dict) -> tuple[pd.DataFrame, float, int]:
    """Canonical column order, then one scoring call. No UI, no logging."""
    input_df = feature_contract.order_columns(pd.DataFrame([payload]))
    probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    return input_df, probability, int(probability >= threshold)


def explain(explainer, feature_columns, input_df: pd.DataFrame) -> dict:
    """SHAP contribution per feature, or {} when no explainer is deployed."""
    if explainer is None:
        return {}
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return {
        name: float(value) for name, value in zip(feature_columns, shap_values[0])
    }


def main() -> None:
    """Render the assessment UI.

    Streamlit executes the script with __name__ == "__main__" (verified
    against streamlit.testing AppTest), so this runs under
    `streamlit run streamlit_app.py` while a plain import stays free of UI
    side effects - no page config, no CSS injection, no model loading.
    """
    st.set_page_config(
        page_title="Diabetes Risk Assessment",
        page_icon=":material/health_metrics:",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    theme.inject_css()

    pipeline, threshold, feature_columns, model_name, confidence_intervals = load_model()
    shap_explainer, _shap_expected, _shap_features = load_shap_explainer()
    metrics = load_evaluation_metrics()
    attestation = load_artifact_attestation()

    public_components.render_header()

    if pipeline is None:
        st.error(
            "The risk model is not available, so no assessment can be made. "
            "If you are running this locally, train the model first with "
            "`python logisticregression_only.py`."
        )
        st.stop()

    tab_assess, tab_about = st.tabs(["Risk assessment", "About this tool"])

    with tab_assess:
        public_components.render_scope_notice()

        st.header("Your health information")
        st.caption("All fields are required. Nothing is filled in for you.")

        with st.form("assessment_form"):
            payload, answers = public_components.render_feature_form()
            submitted = st.form_submit_button(
                "Estimate my risk", use_container_width=True, type="primary"
            )

        if submitted:
            outstanding = public_components.missing_features(payload)
            if outstanding:
                st.warning(
                    "Please answer every question before requesting an estimate. "
                    "Still needed: " + ", ".join(outstanding),
                    icon=":material/warning:",
                )
            else:
                with st.spinner("Calculating your risk estimate..."):
                    input_df, probability, prediction = score(pipeline, threshold, payload)
                    shap_by_feature = explain(shap_explainer, feature_columns, input_df)

                    try:
                        log_inference(
                            request_id=str(uuid.uuid4()),
                            model_variant=MODEL_VARIANT,
                            model_name=model_name,
                            probability=probability,
                            prediction=prediction,
                            threshold=threshold,
                            payload=payload,
                        )
                    except Exception:
                        # Monitoring must never break the assessment for a visitor.
                        pass

                st.session_state[RESULT_KEY] = {
                    "probability": probability,
                    "prediction": prediction,
                    "threshold": threshold,
                    "answers": answers,
                    "shap": shap_by_feature,
                    "model_name": model_name,
                    "model_variant": MODEL_VARIANT,
                }

        # Rendered from session state, so a rerun that is not a submission
        # (switching tabs, resizing) does not erase the visitor's result.
        result = st.session_state.get(RESULT_KEY)
        if result is not None:
            st.header("Your result")
            public_components.render_result(result)
            public_components.render_answers(result)
            public_components.render_explanation_panel(result)
            public_components.render_performance_panel(metrics, confidence_intervals)
            public_components.render_model_details(result, attestation)

    with tab_about:
        public_components.render_about(metrics)


if __name__ == "__main__":
    main()
