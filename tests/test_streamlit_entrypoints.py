"""Streamlit entrypoints: import-safe, but still launchable.

Making these modules importable is only half the contract - the other half is
that `streamlit run` still works. Both are proved here.

Streamlit executes a script with ``__name__ == "__main__"``, which is what makes
the ``if __name__ == "__main__": main()`` guard work for it while leaving a
plain import inert. That is verified directly by
test_streamlit_runs_a_script_as_main rather than assumed.

streamlit.testing.v1.AppTest runs the real script in-process, so these are
deterministic: no browser, no ports, no background processes to reap.
"""
import importlib

import pytest

import admin_auth
from conftest import REPO_ROOT

ENTRYPOINTS = ["streamlit_app.py", "admin_app.py"]

pytest.importorskip("streamlit.testing.v1", reason="streamlit testing API unavailable")
from streamlit.testing.v1 import AppTest  # noqa: E402 - after importorskip


@pytest.fixture(autouse=True)
def isolated_admin_state(tmp_path, monkeypatch):
    """No test here may touch the real credential store or admin env vars."""
    monkeypatch.setattr(admin_auth, "USERS_PATH", tmp_path / "data" / "admin_users.json")
    monkeypatch.delenv(admin_auth.ENV_USERNAME, raising=False)
    monkeypatch.delenv(admin_auth.ENV_PASSWORD, raising=False)
    monkeypatch.setenv("DATABASE_URL", "")
    return tmp_path / "data" / "admin_users.json"


def _run(script: str, **kwargs) -> AppTest:
    app = AppTest.from_file(str(REPO_ROOT / script), default_timeout=180, **kwargs)
    app.run()
    return app


# ------------------------------------------------------- execution semantics

def test_streamlit_runs_a_script_as_main(tmp_path):
    """The premise the __main__ guard depends on, verified rather than assumed."""
    marker = tmp_path / "observed.txt"
    probe = tmp_path / "probe_app.py"
    probe.write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(__name__, encoding='utf-8')\n",
        encoding="utf-8",
    )

    AppTest.from_file(str(probe), default_timeout=60).run()

    assert marker.read_text(encoding="utf-8") == "__main__"


# --------------------------------------------------------------- import safety

@pytest.mark.parametrize("module", ["streamlit_app", "admin_app"])
def test_importing_renders_no_widgets(module):
    """A plain import must not produce any Streamlit element."""
    imported = importlib.import_module(module)
    importlib.reload(imported)

    assert callable(imported.main)


def test_importing_streamlit_app_loads_no_model():
    """Model loading moved into main(); import must not touch the artifacts."""
    import streamlit_app

    importlib.reload(streamlit_app)

    assert not hasattr(streamlit_app, "pipeline")
    assert not hasattr(streamlit_app, "shap_explainer")
    assert not hasattr(streamlit_app, "tab_assess")
    # The path constants and the loader functions stay importable.
    assert streamlit_app.MODEL_BUNDLE_PATH.is_file()
    assert callable(streamlit_app.load_model)


def test_importing_admin_app_configures_no_page():
    import admin_app

    importlib.reload(admin_app)

    assert callable(admin_app.main)
    assert callable(admin_app._configure_page)
    assert "ensure_default" + "_admin" not in dir(admin_app)


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_entrypoint_declares_a_main_guard(script):
    source = (REPO_ROOT / script).read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in source
    assert source.rstrip().endswith("main()")


# ------------------------------------------------------------ launch smokes

def test_public_app_starts_without_exception():
    app = _run("streamlit_app.py")

    assert not app.exception, app.exception
    # The assessment UI renders its header as HTML markdown rather than
    # st.title, and lays the body out in two tabs.
    assert len(app.markdown) > 0, "expected the assessment UI to render"
    assert len(app.tabs) == 2, "expected the assessment and information tabs"
    assert not app.error, [element.value for element in app.error]


def test_admin_app_starts_without_exception():
    app = _run("admin_app.py")

    assert not app.exception, app.exception


def test_admin_app_fails_closed_when_unconfigured(isolated_admin_state):
    """With no provider configured the dashboard must refuse and say so."""
    app = _run("admin_app.py")

    assert not app.exception, app.exception
    messages = " ".join(element.value for element in app.error)
    assert "not configured" in messages
    assert not isolated_admin_state.exists(), "rendering must not create an account"


def test_admin_app_shows_a_login_form_when_configured(monkeypatch, isolated_admin_state):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, "correct-horse-battery-staple")

    app = _run("admin_app.py")

    assert not app.exception, app.exception
    assert not app.error, [element.value for element in app.error]
    assert not isolated_admin_state.exists()


def test_admin_app_does_not_authenticate_by_default(isolated_admin_state):
    app = _run("admin_app.py")

    assert app.session_state["admin_authenticated"] is False
    assert app.session_state["admin_username"] == ""


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_running_an_entrypoint_creates_no_credential_store(script, isolated_admin_state):
    app = _run(script)

    assert not app.exception, app.exception
    assert not isolated_admin_state.exists()
    assert admin_auth._load_users() == []
