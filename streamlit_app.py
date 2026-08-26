"""
Diabetes Risk Assessment
Educational risk-estimation tool. Not a diagnostic device.

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

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.pkl"

SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.pkl"

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

ATTESTATION_PATH = PROJECT_ROOT / "provenance" / "legacy_artifact_attestation.json"

#: Shown as the first option of every categorical input. Selecting nothing is a
#: valid state that blocks scoring, so an untouched form cannot produce a result.
PLACEHOLDER = "Select..."

#: The served variant behind this UI. Variant B is reachable through the API.
MODEL_VARIANT = "A"


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


genhlth_options = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}

age_options = {
    "18-24 years": 1,
    "25-29 years": 2,
    "30-34 years": 3,
    "35-39 years": 4,
    "40-44 years": 5,
    "45-49 years": 6,
    "50-54 years": 7,
    "55-59 years": 8,
    "60-64 years": 9,
    "65-69 years": 10,
    "70-74 years": 11,
    "75-79 years": 12,
    "80+ years": 13,
}

education_options = {
    "Never attended school": 1,
    "Elementary (Grades 1-8)": 2,
    "Some high school (Grades 9-11)": 3,
    "High school graduate / GED": 4,
    "Some college / Technical school": 5,
    "College graduate or higher": 6,
}

binary_yes_no = {"No": 0, "Yes": 1}

#: Only what Streamlit has no primitive for: the page banner and the result
#: hero. Colours are stated explicitly rather than inherited, and no
#: Streamlit-generated class is targeted, so a Streamlit upgrade cannot silently
#: restyle the page. The app background is NOT overridden - that belongs to
#: .streamlit/config.toml.
CUSTOM_CSS = """
<style>
    .app-header {
        background: linear-gradient(135deg, #015f8f, #00697f);
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        color: #ffffff;
        margin-bottom: 1.25rem;
    }

    .app-header h1 {
        margin: 0;
        font-size: 1.6rem;
        color: #ffffff;
    }

    .app-header p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #ffffff;
    }

    .risk-hero {
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 6px solid;
        color: #1f2933;
    }

    .risk-hero .risk-label {
        font-size: 0.95rem;
        margin: 0;
    }

    .risk-hero .risk-value {
        font-size: 2.75rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.2rem 0;
    }

    .risk-hero .risk-band {
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }

    .risk-hero .risk-note {
        font-size: 0.9rem;
        margin: 0.6rem 0 0 0;
    }

    .risk-hero.is-elevated {
        background-color: #fee2e2;
        border-left-color: #b91c1c;
    }

    .risk-hero.is-lower {
        background-color: #dcfce7;
        border-left-color: #15803d;
    }
</style>
"""


def render_header() -> None:
    """Page banner. The only <h1> on the page; sections below use st.header."""
    st.markdown(
        '<div class="app-header">'
        "<h1>Diabetes Risk Assessment</h1>"
        "<p>An educational machine-learning tool for estimating diabetes risk "
        "from general health and lifestyle information.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_scope_notice() -> None:
    """The non-diagnostic boundary, stated before any input is collected."""
    st.info(
        "**This tool does not diagnose diabetes.** It estimates a statistical "
        "risk score from survey-style health information, using a model trained "
        "on population data. It uses no blood tests or laboratory values. "
        "Discuss any health concerns with a qualified healthcare professional - "
        "a clinical diagnosis requires appropriate medical evaluation.",
        icon=":material/info:",
    )


def collect_inputs() -> tuple[dict, dict]:
    """Render the input form; return (numeric payload, human-readable answers).

    Every categorical starts on a placeholder and BMI/PhysHlth start empty, so
    nothing is pre-filled on behalf of the visitor. Missing entries are reported
    to the caller as absent keys rather than substituted with a default.
    """
    payload: dict[str, float] = {}
    answers: dict[str, str] = {}

    def categorical(name: str, label: str, mapping: dict[str, int], help_text: str) -> None:
        choice = st.selectbox(
            label, options=[PLACEHOLDER, *mapping], index=0, help=help_text
        )
        if choice != PLACEHOLDER:
            payload[name] = mapping[choice]
            answers[name] = choice

    st.subheader("About you")
    col1, col2 = st.columns(2)
    with col1:
        categorical("Age", "Age group", age_options, "Your age range.")
    with col2:
        categorical(
            "Education", "Education level", education_options,
            "The highest level of education you completed.",
        )

    st.subheader("General health")
    col1, col2, col3 = st.columns(3)
    with col1:
        categorical(
            "GenHlth", "How is your general health?", genhlth_options,
            "Your own view of your overall health.",
        )
    with col2:
        bmi_spec = feature_contract.spec_for("BMI")
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=float(bmi_spec.minimum),
            max_value=float(bmi_spec.maximum),
            value=None,
            step=0.1,
            placeholder=f"{bmi_spec.minimum:g}-{bmi_spec.maximum:g}",
            help="Weight in kilograms divided by height in metres squared.",
        )
        if bmi is not None:
            payload["BMI"] = float(bmi)
            answers["BMI"] = f"{float(bmi):.1f}"
    with col3:
        phys_spec = feature_contract.spec_for("PhysHlth")
        phys_hlth = st.number_input(
            "Days of poor physical health",
            min_value=int(phys_spec.minimum),
            max_value=int(phys_spec.maximum),
            value=None,
            step=1,
            placeholder=f"{phys_spec.minimum:g}-{phys_spec.maximum:g}",
            help="In the past 30 days, how many days was your physical health not good?",
        )
        if phys_hlth is not None:
            payload["PhysHlth"] = int(phys_hlth)
            answers["PhysHlth"] = f"{int(phys_hlth)} of the last 30 days"

    st.subheader("Medical history")
    col1, col2, col3 = st.columns(3)
    with col1:
        categorical(
            "HighBP", "Told you have high blood pressure?", binary_yes_no,
            "Whether a health professional has told you that you have high blood pressure.",
        )
    with col2:
        categorical(
            "HighChol", "Told you have high cholesterol?", binary_yes_no,
            "Whether a health professional has told you that you have high cholesterol.",
        )
    with col3:
        categorical(
            "HeartDiseaseorAttack", "Heart disease or heart attack?", binary_yes_no,
            "Any history of coronary heart disease or a heart attack.",
        )

    st.subheader("Daily activity")
    col1, col2 = st.columns(2)
    with col1:
        categorical(
            "PhysActivity", "Physically active in the past 30 days?", binary_yes_no,
            "Any physical activity outside of work in the past 30 days.",
        )
    with col2:
        categorical(
            "DiffWalk", "Difficulty walking or climbing stairs?", binary_yes_no,
            "Whether you have serious difficulty walking or climbing stairs.",
        )

    return payload, answers


def missing_features(payload: dict) -> list[str]:
    """Contract features the visitor has not answered yet, in contract order."""
    return [
        spec.display_label
        for spec in feature_contract.FEATURE_SPECS
        if spec.name not in payload
    ]


def score(pipeline, threshold: float, payload: dict) -> tuple[pd.DataFrame, float, int]:
    """Canonical ordering, then a single scoring call. No UI here."""
    input_df = feature_contract.order_columns(pd.DataFrame([payload]))
    probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    return input_df, probability, int(probability >= threshold)


def render_result(result: dict) -> None:
    """The result hero: one primary quantity, then its plain-language meaning."""
    probability = result["probability"]
    elevated = result["prediction"] == 1
    band = (
        "Above the model's alert level"
        if elevated
        else "Below the model's alert level"
    )
    note = (
        "This estimate is high enough that the model would flag it for a closer "
        "look. It does not mean you have diabetes."
        if elevated
        else "This estimate is below the level at which the model would flag a "
        "profile for a closer look. It does not rule out diabetes."
    )

    st.markdown(
        f'<div class="risk-hero {"is-elevated" if elevated else "is-lower"}" '
        'role="status" aria-live="polite">'
        '<p class="risk-label">Estimated diabetes risk</p>'
        f'<p class="risk-value">{probability:.0%}</p>'
        f'<p class="risk-band">{band}</p>'
        f'<p class="risk-note">{note}</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "This is a statistical risk estimate, not a diagnosis. Only a qualified "
        "healthcare professional can diagnose diabetes."
    )


def render_answers(result: dict) -> None:
    """What the visitor entered, in their words rather than in model codes."""
    with st.expander("Your answers"):
        rows = [
            {
                "Question": feature_contract.spec_for(name).display_label,
                "Your answer": result["answers"].get(name, "-"),
            }
            for name in feature_contract.FEATURE_NAMES
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_explanation(result: dict, explainer_available: bool) -> None:
    """Which answers moved the score, described rather than colour-coded."""
    with st.expander("What influenced this estimate", expanded=True):
        if not explainer_available:
            st.info(
                "The explanation model is not available in this deployment, so "
                "the per-answer breakdown cannot be shown. The risk estimate "
                "above is unaffected.",
                icon=":material/info:",
            )
            return

        st.caption(
            "Each answer either pushed the estimate up or pulled it down, "
            "relative to an average profile. These are associations the model "
            "learned in the data, not causes."
        )

        contributions = pd.DataFrame(
            [
                {
                    "Answer": feature_contract.spec_for(name).display_label,
                    "You entered": result["answers"].get(name, "-"),
                    "Effect": "Increased the estimate" if value > 0 else "Reduced the estimate",
                    "Strength": abs(float(value)),
                }
                for name, value in result["shap"].items()
            ]
        ).sort_values("Strength", ascending=False)

        st.dataframe(
            contributions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strength": st.column_config.ProgressColumn(
                    "Strength",
                    help="How strongly this answer moved the estimate.",
                    min_value=0.0,
                    max_value=float(max(contributions["Strength"].max(), 1e-9)),
                    format="%.3f",
                )
            },
        )


def render_performance(metrics: dict) -> None:
    """Dataset-level evaluation numbers, labelled as such."""
    test = metrics.get("test_metrics", {})
    if not test:
        return

    with st.expander("How well the model performed on held-out data"):
        st.caption(
            "Measured once on a held-out test set that the model never saw "
            "during training. These describe the model's behaviour across that "
            "whole dataset - they are not a statement about how certain this "
            "particular estimate is."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{test['precision']:.0%}")
        col2.metric("Recall", f"{test['recall']:.0%}")
        col3.metric("ROC-AUC", f"{test['roc_auc']:.3f}")

        st.markdown(
            f"- Of the profiles the model flagged as higher-risk, about "
            f"**{test['precision']:.0%}** did have diabetes in the evaluation data.\n"
            f"- Of the people who did have diabetes, the model flagged about "
            f"**{test['recall']:.0%}** of them.\n"
            f"- Its overall ability to rank higher-risk profiles above lower-risk "
            f"ones (ROC-AUC) was **{test['roc_auc']:.3f}**, where 0.5 is no better "
            "than chance and 1.0 is perfect."
        )


def render_model_details(result: dict, attestation: dict) -> None:
    """Model identity, limited to what the committed artifacts can support."""
    with st.expander("Model details"):
        col1, col2 = st.columns(2)
        col1.metric("Model variant", result["model_variant"])
        col2.metric("Features used", len(feature_contract.FEATURE_NAMES))

        st.markdown(
            f"- **Model family:** {result['model_name']}\n"
            f"- **Alert level (classification threshold):** {result['threshold']:.1%} - "
            "estimates at or above this are flagged as higher-risk. It was chosen "
            "during training to balance catching true cases against false alarms, "
            "and is model metadata rather than a number about you.\n"
            f"- **Your estimate:** {result['probability']:.1%}"
        )

        if attestation:
            unknown = attestation.get("unknown_history", {})
            st.markdown(
                f"- **Artifact integrity:** {len(attestation.get('artifacts', []))} "
                "model files are inventoried by SHA-256 checksum and verified in CI."
            )
            if unknown.get("producer_git_sha") is None:
                st.caption(
                    "Training lineage for these artifacts is recorded as unknown: "
                    "they predate this project's provenance system, and "
                    "reconstructing a training run for them would be fabrication. "
                    "Integrity of the files is verified; their history is not claimed."
                )


def render_about_tab(metrics: dict) -> None:
    """Longer-form background, kept out of the assessment flow."""
    st.header("About this tool")
    st.markdown(
        "This tool estimates the statistical likelihood of diabetes from ten "
        "general health and lifestyle answers. It is built as an educational and "
        "portfolio demonstration of an end-to-end machine-learning system, and "
        "it is **not** a medical device."
    )

    st.subheader("What it does not do")
    st.markdown(
        "- It does not diagnose diabetes, or any other condition.\n"
        "- It does not use blood glucose, HbA1c, or any laboratory result.\n"
        "- It does not replace evaluation by a healthcare professional.\n"
        "- It does not account for everything that affects an individual's risk."
    )

    st.subheader("How it works")
    st.markdown(
        "- **Model:** logistic regression with L2 regularisation.\n"
        "- **Tuning:** Optuna hyperparameter search with 5-fold cross-validation.\n"
        "- **Alert level:** chosen with Youden's J statistic, balancing sensitivity "
        "and specificity.\n"
        "- **Calibration:** Platt scaling, so the percentage behaves like a "
        "probability rather than an arbitrary score.\n"
        "- **Explanations:** SHAP values, showing how each answer moved the estimate.\n"
        "- **Evaluation:** a held-out test set kept separate from training and tuning."
    )

    st.subheader("What it asks about")
    st.dataframe(
        pd.DataFrame(
            [
                {"Question": spec.display_label, "Accepted values": _domain_text(spec)}
                for spec in feature_contract.FEATURE_SPECS
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    test = metrics.get("test_metrics", {})
    if test:
        st.subheader("Measured performance")
        st.markdown(
            f"On the held-out test set: precision **{test['precision']:.0%}**, "
            f"recall **{test['recall']:.0%}**, ROC-AUC **{test['roc_auc']:.3f}**. "
            "These are dataset-level figures, not a confidence level for any "
            "individual estimate."
        )

    st.subheader("Data source")
    st.markdown(
        "Trained on the CDC Behavioral Risk Factor Surveillance System (BRFSS) "
        "health-indicator dataset. BRFSS is a self-reported telephone survey, so "
        "the model reflects the patterns and the limitations of survey data."
    )


def _domain_text(spec) -> str:
    """Human-readable accepted range for one contract feature."""
    if spec.kind == "binary":
        return "Yes or No"
    if spec.kind == "continuous":
        return f"{spec.minimum:g} to {spec.maximum:g}"
    return f"{spec.minimum:g} to {spec.maximum:g} (scale)"


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

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    pipeline, threshold, feature_columns, model_name, _ci_bounds = load_model()
    shap_explainer, _shap_expected, _shap_features = load_shap_explainer()
    metrics = load_evaluation_metrics()
    attestation = load_artifact_attestation()

    render_header()

    if pipeline is None:
        st.error(
            "The risk model is not available, so no assessment can be made. "
            "If you are running this locally, train the model first with "
            "`python logisticregression_only.py`."
        )
        st.stop()

    tab_assess, tab_about = st.tabs(["Risk assessment", "About this tool"])

    with tab_assess:
        render_scope_notice()

        st.header("Your health information")
        st.caption("All fields are required. Nothing is filled in for you.")

        with st.form("assessment_form"):
            payload, answers = collect_inputs()
            submitted = st.form_submit_button(
                "Estimate my risk", use_container_width=True, type="primary"
            )

        if submitted:
            outstanding = missing_features(payload)
            if outstanding:
                st.warning(
                    "Please answer every question before requesting an estimate. "
                    "Still needed: " + ", ".join(outstanding),
                    icon=":material/warning:",
                )
            else:
                with st.spinner("Calculating your risk estimate..."):
                    input_df, probability, prediction = score(pipeline, threshold, payload)

                    shap_by_feature = {}
                    if shap_explainer is not None:
                        shap_values = shap_explainer.shap_values(input_df)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        shap_by_feature = {
                            name: float(value)
                            for name, value in zip(feature_columns, shap_values[0])
                        }

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

                st.session_state["assessment_result"] = {
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
        result = st.session_state.get("assessment_result")
        if result is not None:
            st.header("Your result")
            render_result(result)
            render_answers(result)
            render_explanation(result, explainer_available=bool(result["shap"]))
            render_performance(metrics)
            render_model_details(result, attestation)

    with tab_about:
        render_about_tab(metrics)


if __name__ == "__main__":
    main()
