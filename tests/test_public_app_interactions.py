"""What the public assessment page does once someone actually uses it.

The existing entrypoint tests prove the script is import-safe and that it
launches. Everything here happens after a click: filling the form, submitting
it, and reading the result. AppTest runs the real script in-process, so these
stay deterministic - no browser, no ports, no screenshots to re-baseline.

Two isolation rules apply to every test in this module:

* the inference log is redirected to tmp_path, so running the suite never writes
  to data/inference_logs.db or to a configured DATABASE_URL;
* nothing is asserted about the committed artifacts other than reading them.
"""
import shutil

import pytest
import streamlit as st

import inference_db
from conftest import ARTIFACTS_DIR, REPO_ROOT
from ml_core import feature_contract
from ui import public_components

pytest.importorskip("streamlit.testing.v1", reason="streamlit testing API unavailable")
from streamlit.testing.v1 import AppTest

APP = "streamlit_app.py"


@pytest.fixture(autouse=True)
def isolated_inference_log(tmp_path, monkeypatch):
    """No test here may reach the real inference database."""
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setattr(inference_db, "DB_PATH", tmp_path / "inference_logs.db")
    return tmp_path / "inference_logs.db"


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    """Each test gets cold loaders.

    st.cache_resource keys on the function's qualified name and code, not on the
    file it came from, so a copied script is a cache HIT for the original app's
    entry. Without this, a deployment fixture that omits an artifact would still
    be handed the real model cached by an earlier test, and the test would pass
    while proving nothing.
    """
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def run_app() -> AppTest:
    app = AppTest.from_file(str(REPO_ROOT / APP), default_timeout=180)
    app.run()
    return app


def fill_every_field(app: AppTest, *, bmi: float = 28.0, phys_hlth: int = 4) -> AppTest:
    """Answer every question with its first real option (index 0 is the placeholder)."""
    for widget in app.selectbox:
        widget.select_index(1)
    app.number_input[0].set_value(bmi)
    app.number_input[1].set_value(phys_hlth)
    return app


def submit(app: AppTest) -> AppTest:
    app.button[0].click().run()
    return app


def all_frames(app: AppTest) -> str:
    """Every dataframe rendered in full - repr() elides middle rows."""
    return "\n".join(frame.value.to_string() for frame in app.dataframe)


def all_text(app: AppTest) -> str:
    """Every rendered string, for asserting that a message reached the page."""
    parts = []
    for group in (app.markdown, app.caption, app.info, app.warning, app.error, app.success):
        parts.extend(str(element.value) for element in group)
    return "\n".join(parts)


# ============================================ the form covers the contract

def test_the_form_asks_about_every_served_feature():
    asked = [q.feature for _heading, questions in public_components.SECTIONS for q in questions]

    assert sorted(asked) == sorted(feature_contract.FEATURE_NAMES)


def test_every_question_names_a_real_contract_feature():
    for _heading, questions in public_components.SECTIONS:
        for question in questions:
            assert feature_contract.spec_for(question.feature) is not None
            assert question.label and question.help


def test_a_submitted_payload_has_exactly_the_contract_keys():
    app = submit(fill_every_field(run_app()))

    assert set(app.session_state["assessment_result"]["answers"]) == set(
        feature_contract.FEATURE_NAMES
    )


@pytest.mark.parametrize("feature", sorted(public_components.CATEGORICAL_CHOICES))
def test_categorical_choices_cover_the_canonical_domain_exactly(feature):
    """Display wording is UI copy; the codes behind it must be contract."""
    choices = public_components.CATEGORICAL_CHOICES[feature]
    spec = feature_contract.spec_for(feature)

    assert tuple(sorted(choices.values())) == spec.allowed_values
    assert len(set(choices)) == len(choices), "duplicate display label"


@pytest.mark.parametrize(
    ("index", "feature"),
    [(0, "BMI"), (1, "PhysHlth")],
    ids=["bmi", "phys_hlth"],
)
def test_numeric_widget_bounds_match_the_contract(index, feature):
    """Asserted on the rendered widget, not on the source that built it."""
    app = run_app()
    spec = feature_contract.spec_for(feature)
    widget = app.number_input[index]

    assert widget.min == spec.dtype(spec.minimum)
    assert widget.max == spec.dtype(spec.maximum)


# ==================================================== nothing is pre-filled

def test_no_field_is_answered_before_the_visitor_answers_it():
    app = run_app()

    assert all(widget.value == public_components.PLACEHOLDER for widget in app.selectbox)
    assert all(widget.value is None for widget in app.number_input)


def test_an_untouched_form_cannot_produce_an_assessment():
    """The phantom-patient defect: one click used to yield a full result."""
    app = submit(run_app())

    assert "assessment_result" not in app.session_state
    assert app.warning, "expected the page to say what is still needed"


def test_a_partially_completed_form_names_what_is_missing():
    app = run_app()
    app.selectbox[0].select_index(1)
    app.number_input[0].set_value(25.0)
    submit(app)

    assert "assessment_result" not in app.session_state
    message = " ".join(str(w.value) for w in app.warning)
    unanswered = public_components.missing_features({"Age": 1, "BMI": 25.0})
    for label in unanswered:
        assert label in message


# ======================================================== the result itself

def test_a_complete_submission_renders_a_result():
    app = submit(fill_every_field(run_app()))

    assert not app.exception, app.exception
    result = app.session_state["assessment_result"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)


def test_the_classification_follows_the_threshold():
    app = submit(fill_every_field(run_app()))
    result = app.session_state["assessment_result"]

    assert result["prediction"] == int(result["probability"] >= result["threshold"])


def test_the_result_states_that_it_is_not_a_diagnosis():
    app = submit(fill_every_field(run_app()))

    text = all_text(app).lower()
    assert "not a diagnosis" in text


def test_the_scope_notice_renders_before_any_submission():
    app = run_app()

    assert "does not diagnose diabetes" in all_text(app)


def test_the_result_does_not_prescribe_diagnostic_tests():
    """A survey-trained model must not issue specific test instructions."""
    app = submit(fill_every_field(run_app()))

    text = all_text(app).lower()
    # Naming a test to say the model does NOT use it is fine; instructing the
    # visitor to obtain one is not. These are the phrasings that were removed.
    for prescription in (
        "further diagnostic evaluation",
        "confirmatory testing",
        "oral glucose tolerance test",
        "recommend further",
        "as appropriate",
    ):
        assert prescription not in text, f"result prescribes: {prescription}"
    assert "cannot diagnose" in text or "not a diagnosis" in text


def test_the_result_survives_a_rerun_that_is_not_a_submission():
    app = submit(fill_every_field(run_app()))
    before = app.session_state["assessment_result"]["probability"]

    app.run()

    assert app.session_state["assessment_result"]["probability"] == before


# ================================================== explanation presentation

def test_the_explanation_uses_display_labels_not_raw_feature_names():
    app = submit(fill_every_field(run_app()))

    rendered = all_frames(app)
    assert "HeartDiseaseorAttack" not in rendered
    assert feature_contract.spec_for("HeartDiseaseorAttack").display_label in rendered


def test_the_explanation_states_direction_in_words():
    """Meaning must not depend on reading a colour."""
    app = submit(fill_every_field(run_app()))

    rendered = all_frames(app)
    assert "Increased the estimate" in rendered or "Reduced the estimate" in rendered


# ============================================== deployments missing artifacts

@pytest.fixture
def deployment(tmp_path):
    """A real copy of the script whose project root holds only the named artifacts.

    AppTest executes the script as its own module, so monkeypatching the
    imported ``streamlit_app`` does NOT reach the running app - a patched
    loader would be silently ignored and the test would pass without proving
    anything. Copying the script into a directory with a chosen subset of
    artifacts drives the real loaders instead: the script resolves
    PROJECT_ROOT from its own __file__, while ui/ and inference_db still
    import from the repository on sys.path.
    """

    def build(*artifacts: str) -> AppTest:
        project = tmp_path / "deployment"
        (project / "model_artifacts").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / APP, project / APP)
        for name in artifacts:
            shutil.copy2(ARTIFACTS_DIR / name, project / "model_artifacts" / name)
        app = AppTest.from_file(str(project / APP), default_timeout=180)
        app.run()
        return app

    return build


def test_a_missing_explainer_says_so_instead_of_rendering_nothing(deployment):
    """No shap_explainer.pkl: the estimate still works, the gap is stated."""
    app = deployment("model_bundle.pkl", "metrics.json")

    submit(fill_every_field(app))

    assert not app.exception, app.exception
    assert app.session_state["assessment_result"]["shap"] == {}
    assert "explanation model is not available" in all_text(app).lower()


def test_a_missing_model_fails_visibly_rather_than_silently(deployment):
    app = deployment()

    assert not app.exception, app.exception
    assert app.error, "a missing model must be reported, not silently skipped"
    assert "not available" in " ".join(str(e.value) for e in app.error).lower()


# ============================================== failure paths stay visible

def test_a_logging_failure_does_not_break_the_assessment(monkeypatch):
    """Monitoring is best-effort; a broken log must not cost the visitor a result.

    The failure is injected through supported configuration rather than a
    patch: an unreachable DATABASE_URL sends log_inference down its PostgreSQL
    branch, where connecting raises. Nothing here can be satisfied vacuously -
    if the except in streamlit_app were removed, this test fails.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/nowhere")

    app = submit(fill_every_field(run_app()))

    assert not app.exception, app.exception
    assert "assessment_result" in app.session_state
    assert "not a diagnosis" in all_text(app).lower()


def test_a_successful_assessment_is_logged_once(isolated_inference_log):
    submit(fill_every_field(run_app()))

    rows = inference_db.fetch_recent_logs(limit=10, db_path=isolated_inference_log)
    assert len(rows) == 1
    assert rows[0]["model_variant"] == "A"
