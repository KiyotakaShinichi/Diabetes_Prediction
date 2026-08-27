"""What the admin dashboard does once someone signs in.

The existing entrypoint tests prove the dashboard fails closed when no
authentication provider is configured. Everything here is what happens after
that: logging in, being refused, and reading each tab with and without data.

Isolation: every test points admin_auth at a temporary credential store and
inference_db at a temporary database, so nothing here can read or write the real
credential file or the real inference log.

Drift semantics are deliberately not asserted. The H0 audit records that the
z-score divides by the population standard deviation where the standard error of
the mean would be conventional; correcting that belongs to a later track, and
these tests only prove the panel renders for both baseline schemas.
"""
import uuid

import pytest
import streamlit as st

import admin_auth
import inference_db
from conftest import REPO_ROOT

pytest.importorskip("streamlit.testing.v1", reason="streamlit testing API unavailable")
from streamlit.testing.v1 import AppTest

APP = "admin_app.py"

USERNAME = "auditor"
PASSWORD = "correct-horse-battery-staple"

SAMPLE_PAYLOAD = {
    "GenHlth": 3, "HighBP": 1, "BMI": 28.0, "HighChol": 0, "Age": 7,
    "DiffWalk": 0, "HeartDiseaseorAttack": 0, "PhysHlth": 2,
    "Education": 5, "PhysActivity": 1,
}


@pytest.fixture(autouse=True)
def isolated_admin_environment(tmp_path, monkeypatch):
    """No real credential store, no real inference database."""
    monkeypatch.setattr(admin_auth, "USERS_PATH", tmp_path / "data" / "admin_users.json")
    monkeypatch.delenv(admin_auth.ENV_USERNAME, raising=False)
    monkeypatch.delenv(admin_auth.ENV_PASSWORD, raising=False)
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setattr(inference_db, "DB_PATH", tmp_path / "inference_logs.db")
    return tmp_path / "inference_logs.db"


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def configured_admin(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, USERNAME)
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, PASSWORD)


def run_app() -> AppTest:
    app = AppTest.from_file(str(REPO_ROOT / APP), default_timeout=180)
    app.run()
    return app


def signed_in_app() -> AppTest:
    """Skip the form and start from an authenticated session."""
    app = AppTest.from_file(str(REPO_ROOT / APP), default_timeout=180)
    app.session_state["admin_authenticated"] = True
    app.session_state["admin_username"] = USERNAME
    app.run()
    return app


def seed_logs(db_path, count: int = 6, variant: str = "A") -> None:
    for index in range(count):
        inference_db.log_inference(
            request_id=str(uuid.uuid4()),
            model_variant=variant,
            model_name="logistic_regression" if variant == "A" else "xgboost_boosted_trees",
            probability=0.2 + index * 0.1,
            prediction=int(index > 2),
            threshold=0.4557,
            payload=SAMPLE_PAYLOAD,
            db_path=db_path,
        )


def all_text(app: AppTest) -> str:
    parts = []
    for group in (app.markdown, app.caption, app.info, app.warning, app.error, app.success):
        parts.extend(str(element.value) for element in group)
    return "\n".join(parts)


# ============================================================== login states

def test_unconfigured_authentication_refuses_and_explains():
    app = run_app()

    assert not app.exception, app.exception
    assert "not configured" in " ".join(str(e.value) for e in app.error)
    assert app.session_state["admin_authenticated"] is False


def test_unconfigured_authentication_renders_no_login_form():
    """Nothing to submit when nothing could ever authenticate it."""
    app = run_app()

    assert not app.text_input


def test_a_configured_provider_renders_a_login_form(configured_admin):
    app = run_app()

    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert len(app.text_input) == 2


def test_valid_credentials_sign_the_user_in(configured_admin):
    app = run_app()

    app.text_input[0].set_value(USERNAME)
    app.text_input[1].set_value(PASSWORD)
    app.button[0].click().run()

    assert not app.exception, app.exception
    assert app.session_state["admin_authenticated"] is True
    assert app.session_state["admin_username"] == USERNAME


@pytest.mark.parametrize(
    ("username", "password"),
    [
        pytest.param(USERNAME, "wrong-password", id="wrong_password"),
        pytest.param("intruder", PASSWORD, id="wrong_username"),
        pytest.param("", "", id="empty"),
    ],
)
def test_invalid_credentials_leave_the_session_signed_out(configured_admin, username, password):
    app = run_app()

    app.text_input[0].set_value(username)
    app.text_input[1].set_value(password)
    app.button[0].click().run()

    assert app.session_state["admin_authenticated"] is False
    assert app.error, "a rejected login must say so"


def test_signing_out_clears_the_session():
    app = signed_in_app()

    logout = [button for button in app.button if "Log out" in button.label]
    assert logout, "expected a sign-out control on the dashboard"
    logout[0].click().run()

    assert app.session_state["admin_authenticated"] is False
    assert app.session_state["admin_username"] == ""


# ================================================================ dashboard

def test_the_dashboard_renders_its_three_tabs():
    app = signed_in_app()

    assert not app.exception, app.exception
    assert len(app.tabs) == 3


def test_empty_analytics_says_so_rather_than_erroring():
    app = signed_in_app()

    assert not app.exception, app.exception
    assert "No inference records match these filters" in all_text(app)


def test_populated_analytics_reports_the_seeded_volume(isolated_admin_environment):
    seed_logs(isolated_admin_environment, count=6)

    app = signed_in_app()

    assert not app.exception, app.exception
    totals = [metric for metric in app.metric if metric.label == "Total inferences"]
    assert totals and totals[0].value == "6"


def test_populated_analytics_lists_the_variant_breakdown(isolated_admin_environment):
    seed_logs(isolated_admin_environment, count=4)

    app = signed_in_app()

    rendered = "\n".join(frame.value.to_string() for frame in app.dataframe)
    assert "Variant" in rendered
    assert "logistic_regression" in rendered


def test_committed_performance_metrics_render_for_both_variants():
    app = signed_in_app()

    text = all_text(app)
    assert "Variant A - Logistic Regression" in text
    assert "Variant B - XGBoost Boosted Trees" in text
    assert [metric for metric in app.metric if metric.label == "ROC-AUC"]


def test_confidence_intervals_use_the_shared_column_labels():
    from ui import formatting

    app = signed_in_app()

    rendered = "\n".join(frame.value.to_string() for frame in app.dataframe)
    for column in formatting.CI_COLUMNS:
        assert column in rendered


# =========================================================== drift monitoring

def test_the_drift_tab_renders_the_variant_a_baseline():
    """Variant A stores {feature: {mean, std, ...}}."""
    app = signed_in_app()

    assert not app.exception, app.exception
    assert "Training Distribution (baseline):" in all_text(app)


def test_the_drift_tab_renders_the_variant_b_baseline():
    """Variant B stores {feature_columns, means, stds, ...} - the other schema."""
    app = signed_in_app()

    app.radio[0].set_value("B (XGBoost)").run()

    assert not app.exception, app.exception
    assert "Training Distribution (baseline):" in all_text(app)


def test_drift_analysis_reports_a_verdict_once_enough_logs_exist(isolated_admin_environment):
    from ui.admin_components import MIN_DRIFT_SAMPLE

    seed_logs(isolated_admin_environment, count=MIN_DRIFT_SAMPLE + 5)

    app = signed_in_app()

    assert not app.exception, app.exception
    text = all_text(app).lower()
    assert "mean shift detected" in text or "no feature mean differs" in text


def test_drift_analysis_refuses_to_judge_a_tiny_sample(isolated_admin_environment):
    """Reporting "no drift" from five rows was the old false comfort."""
    seed_logs(isolated_admin_environment, count=5)

    app = signed_in_app()

    assert not app.exception, app.exception
    assert "not enough recent inferences" in all_text(app).lower()


def test_drift_analysis_states_its_verdict_without_relying_on_an_icon(
    isolated_admin_environment,
):
    """The verdict must be readable as text, not only as a coloured glyph."""
    from ui.admin_components import MIN_DRIFT_SAMPLE

    seed_logs(isolated_admin_environment, count=MIN_DRIFT_SAMPLE + 5)

    app = signed_in_app()

    rendered = "\n".join(frame.value.to_string() for frame in app.dataframe)
    assert "Shifted?" in rendered
    assert "YES" in rendered or "No" in rendered
