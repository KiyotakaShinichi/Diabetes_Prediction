"""
Admin Dashboard - Separate Application (Port 8504)
Login-protected analytics dashboard with inference monitoring,
model performance, and drift detection.
Run with: streamlit run admin_app.py --server.port 8504
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from admin_auth import authenticate_user, ensure_default_admin
from inference_db import fetch_recent_logs

ARTIFACTS_DIR = Path("model_artifacts")
MODEL_BUNDLE_A = ARTIFACTS_DIR / "model_bundle.pkl"
MODEL_BUNDLE_B = ARTIFACTS_DIR / "boosted_model_bundle.pkl"
METRICS_A = ARTIFACTS_DIR / "metrics.json"
METRICS_B = ARTIFACTS_DIR / "boosted_metrics.json"
DRIFT_BASELINE_A = ARTIFACTS_DIR / "drift_baseline.pkl"
DRIFT_BASELINE_B = ARTIFACTS_DIR / "boosted_drift_baseline.pkl"


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

    tab_logs, tab_perf, tab_drift = st.tabs(["📋 Inference Logs", "📈 Model Performance", "🔍 Drift Monitoring"])

    # ===========================
    # TAB 1: Inference Logs
    # ===========================
    with tab_logs:
        max_rows = st.slider("Records to load", min_value=50, max_value=5000, value=500, step=50)
        logs = fetch_recent_logs(limit=max_rows)

        if not logs:
            st.info("No inference logs yet. Predictions will appear here once the API is used.")
        else:
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

            if logs_df["model_name"].nunique() > 1:
                st.write("**Model Performance Comparison**")
                model_perf = logs_df.groupby("model_name").agg({
                    "prediction": ["count", "mean"],
                    "probability": "mean"
                }).round(4)
                model_perf.columns = ["Total", "Positive Rate", "Avg Probability"]
                st.dataframe(model_perf, use_container_width=True)

            st.divider()

            # Timeline
            st.subheader("📅 Request Timeline")
            valid_dates = logs_df.dropna(subset=["created_at"])
            if not valid_dates.empty:
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

            # Full Logs
            st.subheader("📋 Inference Log Records")
            display_df = logs_df.drop(columns=["payload"], errors="ignore").copy()
            if "created_at" in display_df.columns:
                display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv_data = logs_df.drop(columns=["payload"], errors="ignore").to_csv(index=False)
            st.download_button(
                label="📥 Download Logs as CSV",
                data=csv_data,
                file_name="inference_logs_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ===========================
    # TAB 2: Model Performance
    # ===========================
    with tab_perf:
        st.subheader("📈 Model Performance & Confidence Intervals")

        for label, metrics_path, bundle_path in [
            ("Variant A — Logistic Regression", METRICS_A, MODEL_BUNDLE_A),
            ("Variant B — XGBoost Boosted Trees", METRICS_B, MODEL_BUNDLE_B),
        ]:
            st.markdown(f"### {label}")

            if not metrics_path.exists():
                st.warning(f"Metrics file not found: {metrics_path}")
                continue

            with open(metrics_path) as f:
                metrics = json.load(f)

            # Key metrics
            test = metrics.get("test_metrics", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ROC-AUC", f"{test.get('roc_auc', 0):.4f}")
            col2.metric("F1 Score", f"{test.get('f1', 0):.4f}")
            col3.metric("Threshold", f"{metrics.get('threshold', 0):.4f}")
            brier = metrics.get("calibration", {}).get("brier_after", test.get("brier_score", 0))
            col4.metric("Brier Score", f"{brier:.4f}")

            # Calibration info
            calib = metrics.get("calibration", {})
            if calib:
                c1, c2 = st.columns(2)
                c1.metric("Brier (Before Calibration)", f"{calib.get('brier_before', 0):.4f}")
                c2.metric("Brier (After Calibration)", f"{calib.get('brier_after', 0):.4f}")

            # Confidence intervals
            ci = metrics.get("confidence_intervals", {})
            if ci:
                st.write("**95% Bootstrap Confidence Intervals**")
                ci_rows = []
                for metric_name, vals in ci.items():
                    ci_rows.append({
                        "Metric": metric_name.upper(),
                        "Mean": f"{vals['mean']:.4f}",
                        "CI Lower": f"{vals['ci_lower']:.4f}",
                        "CI Upper": f"{vals['ci_upper']:.4f}",
                    })
                st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

            # Optuna params
            opt = metrics.get("optuna_params", {})
            if opt:
                with st.expander("Optuna Best Hyperparameters"):
                    st.json(opt)

            st.divider()

    # ===========================
    # TAB 3: Drift Monitoring
    # ===========================
    with tab_drift:
        st.subheader("🔍 Data Drift Monitoring")
        st.caption(
            "Compare recent inference inputs against training distribution. "
            "Features with |z-score| > 3 are flagged as potential drift."
        )

        variant_choice = st.radio("Select model variant", ["A (Logistic Regression)", "B (XGBoost)"], horizontal=True)
        drift_path = DRIFT_BASELINE_A if variant_choice.startswith("A") else DRIFT_BASELINE_B

        if not drift_path.exists():
            st.warning("Drift baseline not found. Retrain the model to generate it.")
        else:
            baseline_raw = joblib.load(drift_path)

            # Normalize two baseline formats
            if "feature_columns" in baseline_raw:
                # XGBoost format: {feature_columns, means, stds, medians, q25, q75, n_train}
                feature_cols = baseline_raw["feature_columns"]
                n_train = baseline_raw.get("n_train", "N/A")
                get_stat = lambda f, s: baseline_raw[s][f]
            else:
                # LR format: {feat: {mean, std, median, q25, q75, ...}, ...}
                feature_cols = list(baseline_raw.keys())
                n_train = "N/A"
                stat_map = {"means": "mean", "stds": "std", "medians": "median", "q25": "q25", "q75": "q75"}
                get_stat = lambda f, s: baseline_raw[f][stat_map[s]]

            st.write(f"**Training set size:** {n_train:,} samples" if isinstance(n_train, int) else f"**Features:** {len(feature_cols)}")

            # Show training distribution
            st.write("**Training Distribution (baseline):**")
            dist_rows = []
            for feat in feature_cols:
                dist_rows.append({
                    "Feature": feat,
                    "Mean": f"{get_stat(feat, 'means'):.4f}",
                    "Std": f"{get_stat(feat, 'stds'):.4f}",
                    "Median": f"{get_stat(feat, 'medians'):.4f}",
                    "Q25": f"{get_stat(feat, 'q25'):.4f}",
                    "Q75": f"{get_stat(feat, 'q75'):.4f}",
                })
            st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

            # Check recent inferences for drift
            st.divider()
            st.write("**Drift Analysis on Recent Inferences:**")
            recent_logs = fetch_recent_logs(limit=200)

            if not recent_logs:
                st.info("No inference logs to analyze. Make some predictions first.")
            else:
                # Parse payloads
                input_rows = []
                for log_entry in recent_logs:
                    try:
                        p = log_entry.get("payload", "{}")
                        if isinstance(p, str):
                            p = json.loads(p)
                        input_rows.append(p)
                    except Exception:
                        continue

                if input_rows:
                    infer_df = pd.DataFrame(input_rows)
                    avail_cols = [c for c in feature_cols if c in infer_df.columns]

                    drift_results = []
                    for feat in avail_cols:
                        infer_mean = float(infer_df[feat].mean())
                        train_mean = get_stat(feat, "means")
                        train_std = get_stat(feat, "stds")
                        z = (infer_mean - train_mean) / train_std if train_std > 0 else 0
                        drift_results.append({
                            "Feature": feat,
                            "Inference Mean": f"{infer_mean:.4f}",
                            "Training Mean": f"{train_mean:.4f}",
                            "Training Std": f"{train_std:.4f}",
                            "Z-Score": f"{z:.4f}",
                            "Drift?": "⚠️ YES" if abs(z) > 3 else "✅ No",
                        })

                    drift_df = pd.DataFrame(drift_results)
                    n_drifted = sum(1 for r in drift_results if "YES" in r["Drift?"])

                    if n_drifted > 0:
                        st.error(f"⚠️ Drift detected in {n_drifted}/{len(avail_cols)} features!")
                    else:
                        st.success("✅ No significant drift detected across all features.")

                    st.dataframe(drift_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Could not parse inference payloads.")


# Main routing
if st.session_state.admin_authenticated:
    dashboard_page()
else:
    login_page()
