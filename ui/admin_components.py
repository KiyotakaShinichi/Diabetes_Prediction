"""Sections of the admin monitoring dashboard.

Presentation only, and deliberately behaviour-preserving. Every figure below is
computed exactly as it was before this module existed: the KPI definitions, the
A/B breakdown, the timeline resampling and the drift z-score are unchanged, and
so are the database queries behind them. Known issues with some of those
definitions are recorded in the Track H0 audit and belong to a later track -
moving code is not the moment to change what it reports.

``admin_app.py`` keeps authentication, tab layout and data loading.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import streamlit as st

from ui import formatting

#: |z| above which a feature is reported as drifted. Unchanged.
DRIFT_Z_THRESHOLD = 3

#: Baseline statistic key -> per-feature key, for the logistic-regression
#: baseline schema. The boosted schema stores the same statistics the other way
#: round; both are read through the accessor built in drift_baseline_accessor().
_LR_STAT_KEYS = {
    "means": "mean",
    "stds": "std",
    "medians": "median",
    "q25": "q25",
    "q75": "q75",
}


def render_kpi_row(logs_df: pd.DataFrame) -> None:
    """Top-level counters. Definitions unchanged from the original dashboard."""
    total_requests = len(logs_df)
    positive_rate = float(logs_df["prediction"].mean()) if total_requests > 0 else 0.0
    avg_probability = float(logs_df["probability"].mean()) if total_requests > 0 else 0.0
    unique_users = int(logs_df["request_id"].nunique())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Inferences", formatting.count(total_requests))
    metric_cols[1].metric("Positive Rate", formatting.percent(positive_rate, 1))
    metric_cols[2].metric("Avg Risk Score", formatting.percent(avg_probability, 1))
    metric_cols[3].metric("Unique Requests", formatting.count(unique_users))


def render_variant_breakdown(logs_df: pd.DataFrame) -> None:
    """Counts by variant and by model, plus the comparison table when both ran."""
    ab_col1, ab_col2 = st.columns(2)

    with ab_col1:
        st.write("**Requests by Variant**")
        variant_counts = (
            logs_df["model_variant"].value_counts().rename_axis("Variant").reset_index(name="Count")
        )
        st.dataframe(variant_counts, use_container_width=True, hide_index=True)

    with ab_col2:
        st.write("**Requests by Model**")
        model_counts = (
            logs_df["model_name"].value_counts().rename_axis("Model").reset_index(name="Count")
        )
        st.dataframe(model_counts, use_container_width=True, hide_index=True)

    if logs_df["model_name"].nunique() > 1:
        st.write("**Model Performance Comparison**")
        model_perf = logs_df.groupby("model_name").agg({
            "prediction": ["count", "mean"],
            "probability": "mean",
        }).round(4)
        model_perf.columns = ["Total", "Positive Rate", "Avg Probability"]
        st.dataframe(model_perf, use_container_width=True)


def render_timeline(logs_df: pd.DataFrame) -> None:
    """Hourly request volume over the loaded window."""
    valid_dates = logs_df.dropna(subset=["created_at"])
    if valid_dates.empty:
        return

    hourly_df = (
        valid_dates.set_index("created_at").resample("h").size().reset_index(name="Requests")
    )
    hourly_df = hourly_df[hourly_df["Requests"] > 0]
    if not hourly_df.empty:
        st.write("**Hourly Request Volume**")
        st.dataframe(hourly_df.tail(24), use_container_width=True, hide_index=True)


def render_logs_table(logs_df: pd.DataFrame) -> None:
    """The raw log listing and its CSV export. Payloads are never displayed."""
    display_df = logs_df.drop(columns=["payload"], errors="ignore").copy()
    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = logs_df.drop(columns=["payload"], errors="ignore").to_csv(index=False)
    st.download_button(
        label="Download Logs as CSV",
        data=csv_data,
        file_name="inference_logs_export.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_performance_section(label: str, metrics: Mapping[str, Any]) -> None:
    """One variant's committed evaluation metrics."""
    st.markdown(f"### {label}")

    test = metrics.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", formatting.decimal(test.get("roc_auc", 0)))
    col2.metric("F1 Score", formatting.decimal(test.get("f1", 0)))
    col3.metric("Threshold", formatting.decimal(metrics.get("threshold", 0)))
    calibration = metrics.get("calibration", {})
    brier = calibration.get("brier_after", test.get("brier_score", 0))
    col4.metric("Brier Score", formatting.decimal(brier))

    if calibration:
        c1, c2 = st.columns(2)
        c1.metric("Brier (Before Calibration)", formatting.decimal(calibration.get("brier_before", 0)))
        c2.metric("Brier (After Calibration)", formatting.decimal(calibration.get("brier_after", 0)))

    intervals = metrics.get("confidence_intervals", {})
    if intervals:
        st.write("**95% Bootstrap Confidence Intervals**")
        st.dataframe(
            formatting.confidence_interval_table(intervals),
            use_container_width=True,
            hide_index=True,
        )

    optuna_params = metrics.get("optuna_params", {})
    if optuna_params:
        with st.expander("Optuna Best Hyperparameters"):
            st.json(optuna_params)


def drift_baseline_accessor(baseline: Mapping[str, Any]) -> tuple[list[str], Any, Callable[[str, str], float]]:
    """Read either drift-baseline schema through one accessor.

    Variant B stores ``{feature_columns, means, stds, ...}``; variant A stores
    ``{feature: {mean, std, ...}}``. Both schemas and both readings are exactly
    as they were - this only replaces two inline lambdas with one named function
    so the two apps cannot drift apart on how a baseline is read.
    """
    if "feature_columns" in baseline:
        feature_cols = list(baseline["feature_columns"])
        n_train = baseline.get("n_train", "N/A")

        def get_stat(feature: str, statistic: str) -> float:
            return baseline[statistic][feature]
    else:
        feature_cols = list(baseline.keys())
        n_train = "N/A"

        def get_stat(feature: str, statistic: str) -> float:
            return baseline[feature][_LR_STAT_KEYS[statistic]]

    return feature_cols, n_train, get_stat


def render_baseline_distribution(feature_cols: list[str], get_stat: Callable[[str, str], float]) -> None:
    """Training-set statistics the drift check compares against."""
    st.write("**Training Distribution (baseline):**")
    dist_rows = [
        {
            "Feature": feature,
            "Mean": formatting.decimal(get_stat(feature, "means")),
            "Std": formatting.decimal(get_stat(feature, "stds")),
            "Median": formatting.decimal(get_stat(feature, "medians")),
            "Q25": formatting.decimal(get_stat(feature, "q25")),
            "Q75": formatting.decimal(get_stat(feature, "q75")),
        }
        for feature in feature_cols
    ]
    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)


def parse_logged_payloads(recent_logs: list[Mapping[str, Any]]) -> list[dict]:
    """Payload dictionaries from log rows, skipping any row that will not parse."""
    parsed = []
    for entry in recent_logs:
        raw = entry.get("payload", "{}")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                continue
        if isinstance(raw, dict):
            parsed.append(raw)
    return parsed


def render_drift_analysis(
    feature_cols: list[str],
    get_stat: Callable[[str, str], float],
    input_rows: list[dict],
) -> None:
    """Per-feature drift verdict.

    The statistic is unchanged: z = (inference mean - training mean) / training
    std, flagged above |3|. The H0 audit records that this uses the population
    standard deviation where the standard error of the mean would be the
    conventional choice; correcting it is deferred so that this refactor changes
    no reported number.
    """
    infer_df = pd.DataFrame(input_rows)
    avail_cols = [column for column in feature_cols if column in infer_df.columns]

    drift_results = []
    for feature in avail_cols:
        infer_mean = float(infer_df[feature].mean())
        train_mean = get_stat(feature, "means")
        train_std = get_stat(feature, "stds")
        z = (infer_mean - train_mean) / train_std if train_std > 0 else 0
        drift_results.append({
            "Feature": feature,
            "Inference Mean": formatting.decimal(infer_mean),
            "Training Mean": formatting.decimal(train_mean),
            "Training Std": formatting.decimal(train_std),
            "Z-Score": formatting.decimal(z),
            "Drift?": "YES" if abs(z) > DRIFT_Z_THRESHOLD else "No",
        })

    n_drifted = sum(1 for row in drift_results if row["Drift?"] == "YES")
    if n_drifted > 0:
        st.error(f"Drift detected in {n_drifted}/{len(avail_cols)} features.", icon=":material/warning:")
    else:
        st.success("No significant drift detected across all features.", icon=":material/check_circle:")

    st.dataframe(pd.DataFrame(drift_results), use_container_width=True, hide_index=True)
