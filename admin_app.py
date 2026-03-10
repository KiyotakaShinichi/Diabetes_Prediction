"""
Admin Dashboard - Separate Application (Port 8502)
Login-protected analytics dashboard for inference monitoring.
Run with: streamlit run admin_app.py --server.port 8502
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from admin_auth import authenticate_user, ensure_default_admin
from inference_db import fetch_recent_logs


st.set_page_config(
    page_title="Admin Dashboard - Diabetes Prediction System",
    page_icon="🔒",
    layout="wide",
)

ensure_default_admin()
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "admin_username" not in st.session_state:
    st.session_state.admin_username = ""


def login_page():
    """Render login form."""
    st.title("🔒 Admin Login")
    st.write("This dashboard is restricted to authorized personnel.")

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


def dashboard_page():
    """Render the admin analytics dashboard."""
    # Header with logout
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    header_col1.title("📊 Inference Analytics Dashboard")
    header_col2.success(f"👤 {st.session_state.admin_username}")
    if header_col3.button("🚪 Logout", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.session_state.admin_username = ""
        st.rerun()

    st.divider()

    # Controls
    max_rows = st.slider("Records to load", min_value=50, max_value=5000, value=500, step=50)
    logs = fetch_recent_logs(limit=max_rows)

    if not logs:
        st.info("No inference logs yet. Predictions will appear here once the API is used.")
        return

    logs_df = pd.DataFrame(logs)
    logs_df["created_at"] = pd.to_datetime(logs_df["created_at"], errors="coerce")

    # Summary Metrics
    st.subheader("📈 Summary Metrics")
    total_requests = len(logs_df)
    positive_rate = float(logs_df["prediction"].mean()) if total_requests > 0 else 0.0
    avg_probability = float(logs_df["probability"].mean()) if total_requests > 0 else 0.0
    unique_users = int(logs_df["request_id"].nunique())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Inferences", f"{total_requests:,}")
    metric_cols[1].metric("Positive Rate", f"{positive_rate:.1%}")
    metric_cols[2].metric("Avg Risk Score", f"{avg_probability:.1%}")
    metric_cols[3].metric("Unique Requests", f"{unique_users:,}")

    st.divider()

    # A/B Testing Analysis
    st.subheader("🔬 A/B Testing Breakdown")
    ab_col1, ab_col2 = st.columns(2)

    with ab_col1:
        st.write("**Requests by Variant**")
        variant_counts = logs_df["model_variant"].value_counts().rename_axis("Variant").reset_index(name="Count")
        st.dataframe(variant_counts, use_container_width=True, hide_index=True)

    with ab_col2:
        st.write("**Requests by Model**")
        model_counts = logs_df["model_name"].value_counts().rename_axis("Model").reset_index(name="Count")
        st.dataframe(model_counts, use_container_width=True, hide_index=True)

    # Performance comparison by model
    if logs_df["model_name"].nunique() > 1:
        st.write("**Model Performance Comparison**")
        model_perf = logs_df.groupby("model_name").agg({
            "prediction": ["count", "mean"],
            "probability": "mean"
        }).round(4)
        model_perf.columns = ["Total", "Positive Rate", "Avg Probability"]
        st.dataframe(model_perf, use_container_width=True)

    st.divider()

    # Timeline Analysis
    st.subheader("📅 Request Timeline")
    valid_dates = logs_df.dropna(subset=["created_at"])
    if not valid_dates.empty:
        # Hourly breakdown
        hourly_df = (
            valid_dates
            .set_index("created_at")
            .resample("h")
            .size()
            .reset_index(name="Requests")
        )
        hourly_df = hourly_df[hourly_df["Requests"] > 0]
        if not hourly_df.empty:
            st.write("**Hourly Request Volume**")
            st.dataframe(hourly_df.tail(24), use_container_width=True, hide_index=True)

    st.divider()

    # Full Logs Table
    st.subheader("📋 Inference Log Records")
    display_df = logs_df.drop(columns=["payload"], errors="ignore").copy()
    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Export
    csv_data = logs_df.drop(columns=["payload"], errors="ignore").to_csv(index=False)
    st.download_button(
        label="📥 Download Logs as CSV",
        data=csv_data,
        file_name="inference_logs_export.csv",
        mime="text/csv",
        use_container_width=True,
    )


# Main routing
if st.session_state.admin_authenticated:
    dashboard_page()
else:
    login_page()
