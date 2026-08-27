"""Sections of the admin monitoring dashboard.

Presentation, plus the small amount of arithmetic a monitoring view needs. The
H0 audit recorded several figures here that could not mean what their labels
claimed; those are corrected rather than carried forward, and each correction is
documented at the function that makes it.

The governing rule is that no figure may claim more than the stored data can
support. Live traffic carries no outcome labels, so nothing here reports live
accuracy; request_id is unique per request, so nothing here counts people; and
the drift comparison states which statistic it computed instead of implying a
test it cannot run.

``admin_app.py`` keeps authentication, tab layout and data loading.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from inference_db import ASSIGNMENT_COLUMN
from ui import formatting

#: Significance level for the per-feature drift test, before correction.
DRIFT_ALPHA = 0.05

#: Below this many production rows the test is not run at all. The statistic
#: relies on the central limit theorem to treat the sampling distribution of a
#: mean as normal; on eight rows of ordinal data that assumption is not
#: available, and reporting "no drift" from it would be the same false comfort
#: the previous implementation gave.
MIN_DRIFT_SAMPLE = 30

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
    """Top-level counters, each one measurable from what is actually stored.

    The previous fourth counter was "Unique Requests", computed as
    ``request_id.nunique()``. Because request_id is a fresh uuid4 written once
    per request, it was arithmetically identical to Total Inferences and could
    never say anything - the underlying variable was even named unique_users,
    implying a person count the schema could not support.

    It is replaced by a count of distinct experiment assignments, which is
    genuinely observable now that the assignment digest is persisted. Rows
    written before that column existed hold NULL and are excluded rather than
    counted as one shared subject, so the figure is reported alongside how many
    rows could actually be attributed.
    """
    total_requests = len(logs_df)
    high_risk_share = float(logs_df["prediction"].mean()) if total_requests > 0 else 0.0
    avg_probability = float(logs_df["probability"].mean()) if total_requests > 0 else 0.0

    if ASSIGNMENT_COLUMN in logs_df.columns:
        attributed = logs_df[ASSIGNMENT_COLUMN].dropna()
        subjects = int(attributed.nunique())
        unattributed = total_requests - len(attributed)
    else:
        subjects = 0
        unattributed = total_requests

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total inferences", formatting.count(total_requests))
    metric_cols[1].metric(
        "High-risk share", formatting.percent(high_risk_share, 1),
        help="Share of assessments the model flagged as higher risk. Not an "
             "accuracy figure - no outcome labels exist for live traffic.",
    )
    metric_cols[2].metric(
        "Mean risk score", formatting.percent(avg_probability, 1),
        help="Average predicted probability across the selected rows.",
    )
    metric_cols[3].metric(
        "Experiment subjects", formatting.count(subjects),
        help="Distinct A/B assignment identifiers. This is an experiment "
             "assignment count, not a count of people."
             + (f" {unattributed} older row(s) predate this field." if unattributed else ""),
    )


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


def hourly_volume(logs_df: pd.DataFrame) -> pd.DataFrame:
    """Requests per hour, gaps included.

    Empty hours are kept rather than dropped: a quiet period is information, and
    removing those rows made the old listing imply continuous traffic.
    """
    valid_dates = logs_df.dropna(subset=["created_at"])
    if valid_dates.empty:
        return pd.DataFrame(columns=["created_at", "Requests"])

    return (
        valid_dates.set_index("created_at").resample("h").size().reset_index(name="Requests")
    )


def render_timeline(logs_df: pd.DataFrame) -> None:
    """Request volume over time, as a chart.

    This was a 24-row table, which is the least legible way to present the one
    genuine time series on the dashboard. st.bar_chart is native and needs no
    new dependency; the table stays behind an expander for anyone reading exact
    counts.
    """
    hourly_df = hourly_volume(logs_df)
    if hourly_df.empty:
        st.info("No timestamped rows in the selected window.", icon=":material/info:")
        return

    st.write("**Requests per hour**")
    # st.vega_lite_chart rather than st.bar_chart: both are native Streamlit and
    # neither adds a dependency, but bar_chart imports altair, which fails to
    # import on Python 3.14 (it takes TypedDict from the standard library there,
    # where PEP 728's closed= does not exist). The canonical runtime is 3.11.16
    # and altair is fine on it, but a chart that cannot render on a developer's
    # interpreter cannot be tested on it either. The Vega-Lite spec is handled
    # entirely by Streamlit's frontend, so it works on both.
    st.vega_lite_chart(
        hourly_df,
        {
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "created_at", "type": "temporal", "title": "Hour"},
                "y": {"field": "Requests", "type": "quantitative", "title": "Requests"},
            },
            "height": 220,
        },
        use_container_width=True,
    )

    with st.expander("Hourly counts as a table"):
        st.dataframe(hourly_df.tail(48), use_container_width=True, hide_index=True)


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
    """One variant's committed held-out metrics.

    Everything here is measured on the held-out test set at training time and
    read from the committed metrics file. None of it describes live traffic:
    served requests carry no outcome label, so live precision, recall or
    accuracy are not computable and are deliberately absent rather than
    approximated.

    The Brier score used to appear twice - once on its own in the top row and
    again as the "after calibration" half of the pair below, both reading
    brier_after. The standalone copy is gone; the before/after pair remains
    because the comparison is the informative part, and the freed slot now
    shows recall, which the threshold choice trades against.
    """
    st.markdown(f"### {label}")

    test = metrics.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", formatting.decimal(test.get("roc_auc", 0)))
    col2.metric("F1 score", formatting.decimal(test.get("f1", 0)))
    col3.metric("Precision", formatting.decimal(test.get("precision", 0)))
    col4.metric("Recall", formatting.decimal(test.get("recall", 0)))

    calibration = metrics.get("calibration", {})
    if calibration:
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision threshold", formatting.decimal(metrics.get("threshold", 0)))
        c2.metric("Brier (before calibration)", formatting.decimal(calibration.get("brier_before", 0)))
        c3.metric("Brier (after calibration)", formatting.decimal(calibration.get("brier_after", 0)))
    else:
        st.metric("Decision threshold", formatting.decimal(metrics.get("threshold", 0)))

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
            return float(baseline[statistic][feature])
    else:
        feature_cols = list(baseline.keys())
        n_train = "N/A"

        def get_stat(feature: str, statistic: str) -> float:
            return float(baseline[feature][_LR_STAT_KEYS[statistic]])

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


def parse_logged_payloads(recent_logs: Sequence[Mapping[str, Any]]) -> list[dict]:
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


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """One feature's comparison against its training baseline."""

    feature: str
    production_mean: float
    training_mean: float
    training_std: float
    #: Effect size: how many training standard deviations the mean has moved.
    #: This is the quantity the old dashboard printed and labelled "Z-Score".
    standardized_shift: float
    #: The actual test statistic, None when the baseline std is zero.
    z_statistic: float | None
    p_value: float | None
    drifted: bool


@dataclass(frozen=True, slots=True)
class DriftReport:
    """The verdict, separated from any rendering so it can be tested directly."""

    status: str  # "insufficient_data" | "drifted" | "stable"
    sample_size: int
    alpha: float
    corrected_alpha: float
    features: tuple[FeatureDrift, ...]

    @property
    def drifted_features(self) -> tuple[FeatureDrift, ...]:
        return tuple(item for item in self.features if item.drifted)


def two_sided_normal_p(z: float) -> float:
    """P(|Z| >= |z|) for a standard normal, without pulling in SciPy."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def assess_drift(
    feature_cols: list[str],
    get_stat: Callable[[str, str], float],
    input_rows: list[dict],
    *,
    alpha: float = DRIFT_ALPHA,
    min_sample: int = MIN_DRIFT_SAMPLE,
) -> DriftReport:
    """Test whether production feature means have moved away from training.

    What the data supports, and what it does not
    -------------------------------------------
    The committed baselines carry mean, standard deviation and quartiles per
    feature - no histograms and no raw training values. Production rows do carry
    raw feature values. That combination supports a comparison of means; it does
    not support PSI, KS or a chi-square goodness-of-fit test, all of which need a
    reference distribution this project does not store. None is forced here.

    The statistic
    -------------
    A one-sample z-test of the production mean against the training mean,
    treating the training standard deviation as the population sigma:

        standardized shift  d = (mean_prod - mean_train) / sd_train
        test statistic      z = d * sqrt(n_prod)

    The previous implementation reported d and compared it to 3, calling the
    result significant. d is an effect size, not a test statistic: it does not
    grow with evidence, so on any realistic traffic volume the check could
    essentially never fire - a mean would have to move three full training
    standard deviations. Multiplying by sqrt(n) supplies the standard error the
    old form omitted, which at n=200 makes the test roughly fourteen times more
    sensitive. Both numbers are reported, because the effect size is what tells
    an operator whether a statistically detectable shift is also a large one.

    Assumptions, stated rather than buried: sd_train is treated as known, which
    is reasonable given the training sample behind it; and the sampling
    distribution of the mean is treated as normal, which is why min_sample
    exists. Ten features are tested at once, so alpha is Bonferroni-corrected -
    without it, one false alarm in twenty per feature would make a stable system
    look permanently drifted.
    """
    frame = pd.DataFrame(input_rows)
    available = [column for column in feature_cols if column in frame.columns]
    sample_size = len(frame)

    if sample_size < min_sample or not available:
        return DriftReport(
            status="insufficient_data",
            sample_size=sample_size,
            alpha=alpha,
            corrected_alpha=alpha,
            features=(),
        )

    corrected_alpha = alpha / len(available)
    results: list[FeatureDrift] = []
    for feature in available:
        production_mean = float(frame[feature].mean())
        training_mean = float(get_stat(feature, "means"))
        training_std = float(get_stat(feature, "stds"))

        if training_std <= 0:
            # A constant training feature has no scale to standardise against.
            results.append(FeatureDrift(
                feature=feature, production_mean=production_mean,
                training_mean=training_mean, training_std=training_std,
                standardized_shift=0.0, z_statistic=None, p_value=None, drifted=False,
            ))
            continue

        shift = (production_mean - training_mean) / training_std
        z_statistic = shift * math.sqrt(sample_size)
        p_value = two_sided_normal_p(z_statistic)
        results.append(FeatureDrift(
            feature=feature, production_mean=production_mean,
            training_mean=training_mean, training_std=training_std,
            standardized_shift=shift, z_statistic=z_statistic,
            p_value=p_value, drifted=p_value < corrected_alpha,
        ))

    drifted = any(item.drifted for item in results)
    return DriftReport(
        status="drifted" if drifted else "stable",
        sample_size=sample_size,
        alpha=alpha,
        corrected_alpha=corrected_alpha,
        features=tuple(results),
    )


def render_drift_analysis(
    feature_cols: list[str],
    get_stat: Callable[[str, str], float],
    input_rows: list[dict],
) -> None:
    """Present the drift verdict, naming the statistic it actually computed."""
    report = assess_drift(feature_cols, get_stat, input_rows)

    if report.status == "insufficient_data":
        st.info(
            f"Not enough recent inferences to test for drift: {report.sample_size} "
            f"row(s) against a minimum of {MIN_DRIFT_SAMPLE}. Reporting no drift "
            "from this little data would not be meaningful.",
            icon=":material/info:",
        )
        return

    st.caption(
        "One-sample z-test of each production feature mean against its training "
        f"mean, using the training standard deviation as the population sigma. "
        f"n = {report.sample_size}. Significance is Bonferroni-corrected across "
        f"{len(report.features)} features, so a feature is flagged at "
        f"p < {report.corrected_alpha:.4f}."
    )

    if report.status == "drifted":
        st.error(
            f"Mean shift detected in {len(report.drifted_features)} of "
            f"{len(report.features)} features.",
            icon=":material/warning:",
        )
    else:
        st.success(
            "No feature mean differs significantly from its training baseline.",
            icon=":material/check_circle:",
        )

    st.dataframe(
        pd.DataFrame([
            {
                "Feature": item.feature,
                "Production mean": formatting.decimal(item.production_mean),
                "Training mean": formatting.decimal(item.training_mean),
                "Training std": formatting.decimal(item.training_std),
                "Standardized shift": formatting.decimal(item.standardized_shift),
                "z": "-" if item.z_statistic is None else formatting.decimal(item.z_statistic),
                "p-value": "-" if item.p_value is None else f"{item.p_value:.2e}",
                "Shifted?": "YES" if item.drifted else "No",
            }
            for item in report.features
        ]),
        use_container_width=True,
        hide_index=True,
    )
