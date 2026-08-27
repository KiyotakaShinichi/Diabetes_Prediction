"""Sections of the public risk-assessment page.

Presentation only. Nothing here loads a bundle, calls ``predict_proba`` or
writes to the inference log - ``streamlit_app.py`` keeps all of that, and passes
finished values in.

On the display mappings below: the wording shown to a visitor ("55-59 years") is
UI copy, not part of the served feature contract, which accepts the integer code
and nothing else. What *is* contract is the set of codes each mapping may
produce, so ``ml_core.feature_contract`` stays the single authority for the
domain and the tests assert every mapping's values equal ``allowed_values``
exactly. That keeps the labels here from becoming a second, untested source of
truth for the model's input space.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd
import streamlit as st

from ml_core import feature_contract
from ui import formatting, theme

#: First option of every choice list. Selecting nothing is a legitimate state
#: that blocks scoring, so an untouched form cannot yield a result.
PLACEHOLDER = "Select..."

YES_NO = MappingProxyType({"No": 0, "Yes": 1})

GENERAL_HEALTH_CHOICES = MappingProxyType({
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
})

AGE_CHOICES = MappingProxyType({
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
})

EDUCATION_CHOICES = MappingProxyType({
    "Never attended school": 1,
    "Elementary (Grades 1-8)": 2,
    "Some high school (Grades 9-11)": 3,
    "High school graduate / GED": 4,
    "Some college / Technical school": 5,
    "College graduate or higher": 6,
})

#: Contract feature -> display wording for its codes.
CATEGORICAL_CHOICES = MappingProxyType({
    "GenHlth": GENERAL_HEALTH_CHOICES,
    "Age": AGE_CHOICES,
    "Education": EDUCATION_CHOICES,
    "HighBP": YES_NO,
    "HighChol": YES_NO,
    "HeartDiseaseorAttack": YES_NO,
    "PhysActivity": YES_NO,
    "DiffWalk": YES_NO,
})

#: Features entered as a number rather than chosen from a list. Their bounds come
#: from the contract at render time, never from a literal here.
NUMERIC_FEATURES = ("BMI", "PhysHlth")


@dataclass(frozen=True, slots=True)
class Question:
    """One contract feature as it is put to the visitor."""

    feature: str
    label: str
    help: str
    placeholder: str = ""
    suffix: str = ""


#: The form, in the order it is rendered. Every contract feature appears exactly
#: once; the tests assert that against FEATURE_NAMES rather than trusting it.
SECTIONS: tuple[tuple[str, tuple[Question, ...]], ...] = (
    (
        "About you",
        (
            Question("Age", "Age group", "Your age range."),
            Question("Education", "Education level",
                     "The highest level of education you completed."),
        ),
    ),
    (
        "General health",
        (
            Question("GenHlth", "How is your general health?",
                     "Your own view of your overall health."),
            Question("BMI", "Body Mass Index (BMI)",
                     "Weight in kilograms divided by height in metres squared."),
            Question("PhysHlth", "Days of poor physical health",
                     "In the past 30 days, how many days was your physical health not good?",
                     suffix=" of the last 30 days"),
        ),
    ),
    (
        "Medical history",
        (
            Question("HighBP", "Told you have high blood pressure?",
                     "Whether a health professional has told you that you have high blood pressure."),
            Question("HighChol", "Told you have high cholesterol?",
                     "Whether a health professional has told you that you have high cholesterol."),
            Question("HeartDiseaseorAttack", "Heart disease or heart attack?",
                     "Any history of coronary heart disease or a heart attack."),
        ),
    ),
    (
        "Daily activity",
        (
            Question("PhysActivity", "Physically active in the past 30 days?",
                     "Any physical activity outside of work in the past 30 days."),
            Question("DiffWalk", "Difficulty walking or climbing stairs?",
                     "Whether you have serious difficulty walking or climbing stairs."),
        ),
    ),
)


def render_header() -> None:
    """Page banner."""
    theme.banner(
        "Diabetes Risk Assessment",
        "An educational machine-learning tool for estimating diabetes risk "
        "from general health and lifestyle information.",
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


def _ask_categorical(question: Question, payload: dict, answers: dict) -> None:
    choices = CATEGORICAL_CHOICES[question.feature]
    chosen = st.selectbox(
        question.label, options=[PLACEHOLDER, *choices], index=0, help=question.help
    )
    if chosen != PLACEHOLDER:
        payload[question.feature] = choices[chosen]
        answers[question.feature] = chosen


def _ask_numeric(question: Question, payload: dict, answers: dict) -> None:
    spec = feature_contract.spec_for(question.feature)
    is_integer = spec.dtype is int
    entered = st.number_input(
        question.label,
        min_value=spec.dtype(spec.minimum),
        max_value=spec.dtype(spec.maximum),
        value=None,
        step=1 if is_integer else 0.1,
        placeholder=f"{spec.minimum:g}-{spec.maximum:g}",
        help=question.help,
    )
    if entered is not None:
        value = int(entered) if is_integer else float(entered)
        payload[question.feature] = value
        answers[question.feature] = (
            f"{value}{question.suffix}" if is_integer else f"{value:.1f}{question.suffix}"
        )


def render_feature_form() -> tuple[dict, dict]:
    """Render every question; return (numeric payload, human-readable answers).

    A feature the visitor has not answered is simply absent from the payload -
    never silently defaulted - so the caller can refuse to score.
    """
    payload: dict[str, float] = {}
    answers: dict[str, str] = {}

    for heading, questions in SECTIONS:
        st.subheader(heading)
        columns = st.columns(len(questions))
        for column, question in zip(columns, questions, strict=True):
            with column:
                if question.feature in CATEGORICAL_CHOICES:
                    _ask_categorical(question, payload, answers)
                else:
                    _ask_numeric(question, payload, answers)

    return payload, answers


def missing_features(payload: dict) -> list[str]:
    """Unanswered contract features, named as the visitor saw them."""
    asked = {q.feature: q.label for _, questions in SECTIONS for q in questions}
    return [
        asked.get(spec.name, spec.display_label)
        for spec in feature_contract.FEATURE_SPECS
        if spec.name not in payload
    ]


def render_result(result: dict) -> None:
    """One primary quantity, then what it means in plain language."""
    elevated = result["prediction"] == 1
    theme.risk_hero(
        label="Estimated diabetes risk",
        value=formatting.percent(result["probability"]),
        band="Above the model's alert level" if elevated else "Below the model's alert level",
        note=(
            "This estimate is high enough that the model would flag it for a "
            "closer look. It does not mean you have diabetes."
            if elevated
            else "This estimate is below the level at which the model would flag "
            "a profile for a closer look. It does not rule out diabetes."
        ),
        elevated=elevated,
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


def render_explanation_panel(result: dict) -> None:
    """Which answers moved the estimate, described in words rather than colour."""
    with st.expander("What influenced this estimate", expanded=True):
        if not result.get("shap"):
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


def render_performance_panel(metrics: dict, confidence_intervals: dict | None = None) -> None:
    """Held-out evaluation figures, labelled as dataset-level behaviour."""
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
        col1.metric("Precision", formatting.percent(test["precision"]))
        col2.metric("Recall", formatting.percent(test["recall"]))
        col3.metric("ROC-AUC", formatting.decimal(test["roc_auc"], 3))

        st.markdown(
            "- Of the profiles the model flagged as higher-risk, about "
            f"**{formatting.percent(test['precision'])}** did have diabetes in "
            "the evaluation data.\n"
            "- Of the people who did have diabetes, the model flagged about "
            f"**{formatting.percent(test['recall'])}** of them.\n"
            "- Its overall ability to rank higher-risk profiles above lower-risk "
            f"ones (ROC-AUC) was **{formatting.decimal(test['roc_auc'], 3)}**, "
            "where 0.5 is no better than chance and 1.0 is perfect."
        )

        if confidence_intervals:
            st.markdown(
                "**Uncertainty around those figures**  \n"
                "Bootstrap resampling of the test set, 95% intervals."
            )
            st.dataframe(
                formatting.confidence_interval_table(confidence_intervals),
                use_container_width=True,
                hide_index=True,
            )


def render_model_details(result: dict, attestation: dict) -> None:
    """Model identity, limited to what the committed artifacts can support."""
    with st.expander("Model details"):
        col1, col2 = st.columns(2)
        col1.metric("Model variant", result["model_variant"])
        col2.metric("Features used", len(feature_contract.FEATURE_NAMES))

        st.markdown(
            f"- **Model family:** {result['model_name']}\n"
            "- **Alert level (classification threshold):** "
            f"{formatting.percent(result['threshold'], 1)} - estimates at or above "
            "this are flagged as higher-risk. It was chosen during training to "
            "balance catching true cases against false alarms, and is model "
            "metadata rather than a number about you.\n"
            f"- **Your estimate:** {formatting.percent(result['probability'], 1)}"
        )

        if result.get("request_id"):
            st.markdown(
                "- **Reference for this estimate:** "
                f"`{result['request_id']}` - quote it if you report a problem, so "
                "the service log for this request can be found."
            )

        if attestation:
            st.markdown(
                f"- **Artifact integrity:** {len(attestation.get('artifacts', []))} "
                "model files are inventoried by SHA-256 checksum and verified in CI."
            )
            if attestation.get("unknown_history", {}).get("producer_git_sha") is None:
                st.caption(
                    "Training lineage for these artifacts is recorded as unknown: "
                    "they predate this project's provenance system, and "
                    "reconstructing a training run for them would be fabrication. "
                    "Integrity of the files is verified; their history is not claimed."
                )


def _domain_text(spec: feature_contract.FeatureSpec) -> str:
    """Human-readable accepted range for one contract feature."""
    if spec.kind == "binary":
        return "Yes or No"
    return f"{spec.minimum:g} to {spec.maximum:g}"


def render_about(metrics: dict) -> None:
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
            f"On the held-out test set: precision **{formatting.percent(test['precision'])}**, "
            f"recall **{formatting.percent(test['recall'])}**, "
            f"ROC-AUC **{formatting.decimal(test['roc_auc'], 3)}**. "
            "These are dataset-level figures, not a confidence level for any "
            "individual estimate."
        )

    st.subheader("Data source")
    st.markdown(
        "Trained on the CDC Behavioral Risk Factor Surveillance System (BRFSS) "
        "health-indicator dataset. BRFSS is a self-reported telephone survey, so "
        "the model reflects the patterns and the limitations of survey data."
    )
