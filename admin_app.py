"""
Admin Dashboard - Separate Application (Port 8504)
Login-protected analytics dashboard with inference monitoring,
model performance, and drift detection.

Page orchestration only: authentication, tab layout, artifact and log loading.
Every section is rendered by ui.admin_components.

Run with: streamlit run admin_app.py --server.port 8504
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from admin_auth import authenticate_user, authentication_status, is_configured
from inference_db import fetch_recent_logs
from ui import admin_components

# Resolve packaged resources from the project directory, never from the caller's
# working directory, so the service behaves identically wherever it is launched.
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
MODEL_BUNDLE_A = ARTIFACTS_DIR / "model_bundle.pkl"
MODEL_BUNDLE_B = ARTIFACTS_DIR / "boosted_model_bundle.pkl"
METRICS_A = ARTIFACTS_DIR / "metrics.json"
METRICS_B = ARTIFACTS_DIR / "boosted_metrics.json"
DRIFT_BASELINE_A = ARTIFACTS_DIR / "drift_baseline.pkl"
DRIFT_BASELINE_B = ARTIFACTS_DIR / "boosted_drift_baseline.pkl"

#: Rows pulled for the drift comparison. Unchanged.
DRIFT_SAMPLE_LIMIT = 200


def _configure_page() -> None:
    """Streamlit page setup. Called from main(), never at import time."""
    st.set_page_config(
        page_title="Admin Dashboard - Diabetes Prediction System",
        page_icon=":material/lock:",
        layout="wide",
    )
    st.session_state.setdefault("admin_authenticated", False)
    st.session_state.setdefault("admin_username", "")


def login_page():
    """Render login form."""
    st.title("Admin Login")
    st.write("This dashboard is restricted to authorized personnel.")

    if not is_configured():
        st.error(
            "Admin authentication is not configured, so no login can succeed. "
            "Set ADMIN_USERNAME and ADMIN_PASSWORD, or create an account with "
            "`python create_admin_user.py --username <name>`."
        )
        st.caption(f"Status: {authentication_status()}")
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("admin_login_form"):
            st.subheader("Enter Credentials")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login", use_container_width=True)

        if login_submitted:
            if authenticate_user(username=username, password=password):
                st.session_state.admin_authenticated = True
                st.session_state.admin_username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")


def render_logs_tab() -> None:
    """Inference volume, A/B split and the raw log listing."""
    max_rows = st.slider("Records to load", min_value=50, max_value=5000, value=500, step=50)
    logs = fetch_recent_logs(limit=max_rows)

    if not logs:
        st.info("No inference logs yet. Predictions will appear here once an assessment is run.")
        return

    logs_df = pd.DataFrame(logs)
    logs_df["created_at"] = pd.to_datetime(logs_df["created_at"], errors="coerce")

    st.subheader("Summary Metrics")
    admin_components.render_kpi_row(logs_df)

    st.divider()
    st.subheader("A/B Testing Breakdown")
    admin_components.render_variant_breakdown(logs_df)

    st.divider()
    st.subheader("Request Timeline")
    admin_components.render_timeline(logs_df)

    st.divider()
    st.subheader("Inference Log Records")
    admin_components.render_logs_table(logs_df)


def render_performance_tab() -> None:
    """Committed evaluation metrics for both variants."""
    st.subheader("Model Performance & Confidence Intervals")

    for label, metrics_path in [
        ("Variant A - Logistic Regression", METRICS_A),
        ("Variant B - XGBoost Boosted Trees", METRICS_B),
    ]:
        if not metrics_path.exists():
            st.markdown(f"### {label}")
            st.warning(f"Metrics file not found: {metrics_path}")
            continue

        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)

        admin_components.render_performance_section(label, metrics)
        st.divider()


def render_drift_tab() -> None:
    """Training baseline against recent inference inputs."""
    st.subheader("Data Drift Monitoring")
    st.caption(
        "Compare recent inference inputs against training distribution. "
        f"Features with |z-score| > {admin_components.DRIFT_Z_THRESHOLD} are "
        "flagged as potential drift."
    )

    variant_choice = st.radio(
        "Select model variant", ["A (Logistic Regression)", "B (XGBoost)"], horizontal=True
    )
    drift_path = DRIFT_BASELINE_A if variant_choice.startswith("A") else DRIFT_BASELINE_B

    if not drift_path.exists():
        st.warning("Drift baseline not found. Retrain the model to generate it.")
        return

    baseline_raw = joblib.load(drift_path)
    feature_cols, n_train, get_stat = admin_components.drift_baseline_accessor(baseline_raw)

    st.write(
        f"**Training set size:** {n_train:,} samples"
        if isinstance(n_train, int)
        else f"**Features:** {len(feature_cols)}"
    )
    admin_components.render_baseline_distribution(feature_cols, get_stat)

    st.divider()
    st.write("**Drift Analysis on Recent Inferences:**")
    recent_logs = fetch_recent_logs(limit=DRIFT_SAMPLE_LIMIT)

    if not recent_logs:
        st.info("No inference logs to analyze. Make some predictions first.")
        return

    input_rows = admin_components.parse_logged_payloads(recent_logs)
    if not input_rows:
        st.warning("Could not parse inference payloads.")
        return

    admin_components.render_drift_analysis(feature_cols, get_stat, input_rows)


def dashboard_page():
    """Render the admin analytics dashboard."""
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    header_col1.title("Inference Analytics Dashboard")
    header_col2.success(f"Signed in: {st.session_state.admin_username}")
    if header_col3.button("Log out", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.session_state.admin_username = ""
        st.rerun()

    st.divider()

    tab_logs, tab_perf, tab_drift = st.tabs(
        ["Inference Logs", "Model Performance", "Drift Monitoring"]
    )

    with tab_logs:
        render_logs_tab()
    with tab_perf:
        render_performance_tab()
    with tab_drift:
        render_drift_tab()


def main() -> None:
    """Application entrypoint.

    Streamlit executes the script with __name__ == "__main__" (verified against
    streamlit.testing AppTest), so the guard below runs under
    `streamlit run admin_app.py` while a plain import stays side-effect free.
    """
    _configure_page()
    if st.session_state.admin_authenticated:
        dashboard_page()
    else:
        login_page()


if __name__ == "__main__":
    main()
