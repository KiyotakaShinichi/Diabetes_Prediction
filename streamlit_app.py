"""
Diabetes Risk Assessment
Educational risk-estimation tool. Not a diagnostic device.

Page orchestration only. This app renders and collects; it does not score.
Every prediction comes from the inference API through ui.api_client, which is
the single authoritative serving path - the same one that enforces canonical
feature order, validates the model bundle, routes A/B, correlates requests and
sanitises errors. No model artifact is loaded here for inference.

Run with: streamlit run streamlit_app.py
"""

import json
import uuid
from pathlib import Path

import streamlit as st

from ui import api_client, public_components, theme

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

#: Static evaluation metadata, not inference. Read from disk because it
#: describes the training run rather than any individual request.
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

ATTESTATION_PATH = PROJECT_ROOT / "provenance" / "legacy_artifact_attestation.json"

#: Where the completed assessment lives between reruns.
RESULT_KEY = "assessment_result"

#: Stable per-session identifier, sent to the API so its deterministic A/B
#: bucketing gives one visitor a consistent variant across submissions. The UI
#: never derives a variant from it - the API decides and reports back.
VISITOR_KEY = "visitor_id"


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


def build_client() -> api_client.DiabetesApiClient:
    """One client per rerun, so configuration changes take effect immediately."""
    return api_client.DiabetesApiClient()


def visitor_id() -> str:
    """A stable id for this browser session, created on first use."""
    if VISITOR_KEY not in st.session_state:
        st.session_state[VISITOR_KEY] = str(uuid.uuid4())
    return st.session_state[VISITOR_KEY]


def request_assessment(client: api_client.DiabetesApiClient, payload: dict) -> dict:
    """Score, then explain, and shape the result the UI renders.

    The explanation is a secondary concern: if it fails the estimate still
    stands, and the UI shows its explicit "unavailable" state rather than
    discarding a valid result. A scoring failure, by contrast, propagates.
    """
    prediction = client.predict(payload, user_id=visitor_id())

    try:
        explanation = client.explain(payload, model_variant=prediction.model_variant)
        contributions = explanation.by_feature()
    except api_client.ApiError:
        contributions = {}

    return {
        "probability": prediction.probability,
        "prediction": prediction.prediction,
        "risk_category": prediction.risk_category,
        "threshold": prediction.threshold,
        "model_name": prediction.model_name,
        "model_variant": prediction.model_variant,
        "request_id": prediction.request_id,
        "confidence_intervals": prediction.confidence_intervals,
        "shap": contributions,
    }


def report_failure(error: api_client.ApiError) -> None:
    """Show a visitor-safe message, with the correlation id when there is one."""
    st.error(error.user_message, icon=":material/error:")
    if error.request_id:
        st.caption(
            "If you report this, quote reference "
            f"`{error.request_id}` so it can be traced in the service log."
        )


def main() -> None:
    """Render the assessment UI.

    Streamlit executes the script with __name__ == "__main__" (verified
    against streamlit.testing AppTest), so this runs under
    `streamlit run streamlit_app.py` while a plain import stays free of UI
    side effects - no page config, no CSS injection, no network calls.
    """
    st.set_page_config(
        page_title="Diabetes Risk Assessment",
        page_icon=":material/health_metrics:",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    theme.inject_css()

    metrics = load_evaluation_metrics()
    attestation = load_artifact_attestation()
    client = build_client()

    public_components.render_header()

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
                    try:
                        result = request_assessment(client, payload)
                    except api_client.ApiError as error:
                        report_failure(error)
                    else:
                        result["answers"] = answers
                        st.session_state[RESULT_KEY] = result

        # Rendered from session state, so a rerun that is not a submission
        # (switching tabs, resizing) does not erase the visitor's result.
        result = st.session_state.get(RESULT_KEY)
        if result is not None:
            st.header("Your result")
            public_components.render_result(result)
            public_components.render_answers(result)
            public_components.render_explanation_panel(result)
            public_components.render_performance_panel(
                metrics, result.get("confidence_intervals")
            )
            public_components.render_model_details(result, attestation)

    with tab_about:
        public_components.render_about(metrics)


if __name__ == "__main__":
    main()
