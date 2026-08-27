"""The public app can actually reach its API once deployed.

Track I-A made FastAPI the only serving path but left the deployment wiring
open: the Render Blueprint told the Streamlit service nothing about where the
API lives, so the client fell back to loopback - which, on Render, is the
Streamlit container itself. The app would have started, looked healthy, and
failed every submission.

Three layers are asserted here:

* the URL contract, including the scheme-less ``host:port`` that Render's
  ``fromService``/``hostport`` produces for private-network addressing;
* the Blueprint itself, parsed rather than grepped, so the service reference
  cannot silently become a hardcoded hostname;
* an integration smoke over real HTTP that approximates the deployed shape -
  the API in one process listening on an allocated port, the client configured
  only through the environment variable a deployment would set.
"""
import re

import pytest

import inference_db
from conftest import REPO_ROOT, VALID_PAYLOAD
from tests.test_serving_convergence import scoring_calls_in
from ui import api_client

RENDER_BLUEPRINT = REPO_ROOT / "render.yaml"

API_SERVICE = "diabetes-api"
PUBLIC_SERVICE = "diabetes-app"
ADMIN_SERVICE = "diabetes-admin"


def load_blueprint() -> dict:
    """Parse render.yaml with whatever YAML parser the environment already has.

    No dependency is added for this: PyYAML arrives with several packages in
    the runtime lock. If it is genuinely absent the test skips rather than
    silently degrading to a text search that would pass on a broken file.
    """
    yaml = pytest.importorskip("yaml", reason="no YAML parser available")
    return yaml.safe_load(RENDER_BLUEPRINT.read_text(encoding="utf-8"))


def service(blueprint: dict, name: str) -> dict:
    for entry in blueprint["services"]:
        if entry.get("name") == name:
            return entry
    pytest.fail(f"{name} is not defined in render.yaml")


def env_var(svc: dict, key: str) -> dict:
    for item in svc.get("envVars", []):
        if item.get("key") == key:
            return item
    pytest.fail(f"{svc['name']} does not set {key}")


# ============================================================ URL contract

def test_the_default_is_loopback(monkeypatch):
    monkeypatch.delenv(api_client.ENV_BASE_URL, raising=False)

    assert api_client.resolve_base_url() == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "configured",
    [
        "http://diabetes-api:10000",
        "https://diabetes-api.example.test",
        "http://127.0.0.1:8000",
    ],
)
def test_an_explicit_url_is_preserved(monkeypatch, configured):
    monkeypatch.setenv(api_client.ENV_BASE_URL, configured)

    assert api_client.resolve_base_url() == configured


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param("diabetes-api:10000", "http://diabetes-api:10000", id="render_hostport"),
        pytest.param("diabetes-api", "http://diabetes-api", id="host_only"),
        pytest.param("api-internal.local:8080", "http://api-internal.local:8080", id="dotted"),
    ],
)
def test_a_scheme_less_authority_is_normalised(monkeypatch, configured, expected):
    """Render's fromService/hostport yields host:port with no scheme."""
    monkeypatch.setenv(api_client.ENV_BASE_URL, configured)

    assert api_client.resolve_base_url() == expected


@pytest.mark.parametrize(
    "configured",
    ["http://diabetes-api:10000/", "http://diabetes-api:10000///", "diabetes-api:10000/"],
)
def test_a_trailing_slash_cannot_produce_a_double_slash(monkeypatch, configured):
    monkeypatch.setenv(api_client.ENV_BASE_URL, configured)

    base = api_client.resolve_base_url()

    assert not base.endswith("/")
    assert "//" not in base.split("://", 1)[1]


@pytest.mark.parametrize(
    "configured",
    [
        pytest.param("ftp://diabetes-api", id="ftp"),
        pytest.param("file:///etc/passwd", id="file"),
        pytest.param("http://", id="scheme_only"),
        pytest.param("not a host", id="spaces"),
        pytest.param("http://host name:80", id="space_in_authority"),
        pytest.param("///", id="only_slashes"),
    ],
)
def test_a_malformed_address_fails_loudly(monkeypatch, configured):
    """Falling back to loopback would hide an operator's typo in production."""
    monkeypatch.setenv(api_client.ENV_BASE_URL, configured)

    with pytest.raises(api_client.ApiConfigurationError):
        api_client.resolve_base_url()


def test_a_configuration_failure_is_still_a_client_error():
    """So a caller that handles ApiError handles this too."""
    assert issubclass(api_client.ApiConfigurationError, api_client.ApiError)


def test_the_configuration_message_names_no_internals():
    message = api_client.ApiConfigurationError().user_message.lower()

    for leak in ("traceback", "127.0.0.1", ".pkl", "c:\\", "/home/", "psycopg"):
        assert leak not in message


@pytest.mark.parametrize("raw", ["", "   ", "not-a-number", "0", "-5"])
def test_a_malformed_timeout_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv(api_client.ENV_TIMEOUT, raw)

    assert api_client.resolve_timeout() == api_client.DEFAULT_TIMEOUT_SECONDS


def test_no_deployment_hostname_is_embedded_in_source():
    """Only loopback may be hardcoded; a real host belongs in the Blueprint."""
    source = (REPO_ROOT / "ui" / "api_client.py").read_text(encoding="utf-8")

    for host in ("onrender.com", "herokuapp", "amazonaws", "azurewebsites", "ngrok"):
        assert host not in source


# ======================================================= Render Blueprint

def test_the_blueprint_defines_the_three_services():
    blueprint = load_blueprint()

    names = {entry.get("name") for entry in blueprint["services"]}
    assert {API_SERVICE, PUBLIC_SERVICE, ADMIN_SERVICE} <= names


def test_the_public_service_is_told_where_the_api_is():
    """The blocker this track exists to close."""
    blueprint = load_blueprint()

    setting = env_var(service(blueprint, PUBLIC_SERVICE), "DIABETES_API_BASE_URL")

    assert "fromService" in setting, "the API address must come from a service reference"
    assert "value" not in setting, "a literal address would not survive a redeploy"


def test_the_api_address_references_the_api_service():
    blueprint = load_blueprint()

    reference = env_var(service(blueprint, PUBLIC_SERVICE), "DIABETES_API_BASE_URL")["fromService"]

    assert reference["name"] == API_SERVICE
    assert reference["property"] == "hostport"


def test_the_referenced_service_exists_in_this_blueprint():
    """A reference to a service that is not defined here would not resolve."""
    blueprint = load_blueprint()
    reference = env_var(service(blueprint, PUBLIC_SERVICE), "DIABETES_API_BASE_URL")["fromService"]

    target = service(blueprint, reference["name"])

    assert target["type"] == reference.get("type", target["type"])


def test_the_referenced_property_is_one_the_client_understands():
    """hostport yields host:port, which resolve_base_url normalises."""
    blueprint = load_blueprint()
    reference = env_var(service(blueprint, PUBLIC_SERVICE), "DIABETES_API_BASE_URL")["fromService"]

    assert reference["property"] in {"hostport", "host"}
    sample = "diabetes-api:10000" if reference["property"] == "hostport" else "diabetes-api"
    assert api_client.resolve_base_url(sample).startswith("http://")


def test_the_blueprint_hardcodes_no_address():
    text = RENDER_BLUEPRINT.read_text(encoding="utf-8")

    for forbidden in ("onrender.com", "localhost", "127.0.0.1"):
        offending = [
            line for line in text.splitlines()
            if forbidden in line and not line.strip().startswith("#")
        ]
        assert not offending, f"{forbidden} appears in {offending}"


def test_the_runtime_contract_is_unchanged():
    """This track changes wiring, not the Python or install contract."""
    blueprint = load_blueprint()

    for name in (API_SERVICE, PUBLIC_SERVICE, ADMIN_SERVICE):
        svc = service(blueprint, name)
        assert env_var(svc, "PYTHON_VERSION")["value"] == "3.11.16"
        assert svc["buildCommand"] == "pip install -r requirements.lock"


def test_only_the_public_service_needs_the_api_address():
    """The admin dashboard reads artifacts and the log directly; it is not a
    client of the inference API, and wiring it would imply otherwise."""
    blueprint = load_blueprint()

    admin = service(blueprint, ADMIN_SERVICE)
    assert all(item.get("key") != "DIABETES_API_BASE_URL" for item in admin.get("envVars", []))


# ================================================= runtime dependency contract

def test_requests_is_a_declared_runtime_dependency():
    """The public UI imports it directly, so it may not be merely transitive.

    Before this track requirements.lock carried "requests ... # via streamlit":
    if Streamlit ever dropped the dependency, the deployed app would break on
    an import it never declared.
    """
    declared = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert re.search(r"^requests[><=~]", declared, re.MULTILINE), (
        "requests must be declared in the canonical runtime requirements"
    )


def test_the_lock_records_requests_as_a_direct_dependency():
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")

    block = lock.split("\nrequests==", 1)[1].split("\n", 2)
    annotation = lock.split("\nrequests==", 1)[1]
    following = annotation.split("\n")[1:4]

    assert block, "requests is absent from the lock"
    assert any("-r requirements.txt" in line for line in following), (
        "the lock still records requests as transitive only"
    )


def test_every_module_the_public_ui_imports_is_declared():
    """Nothing else slipped in undeclared alongside requests."""
    declared = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    third_party = {"requests", "streamlit", "pandas"}

    source = (REPO_ROOT / "ui" / "api_client.py").read_text(encoding="utf-8")
    for module in third_party:
        if f"import {module}" in source:
            assert module in declared, f"{module} is imported but not declared"


# ==================================================== integration smoke

@pytest.fixture
def deployed_like_client(api_base_url) -> api_client.DiabetesApiClient:
    """A client configured exactly as a deployment configures it.

    Only DIABETES_API_BASE_URL is set - no constructor argument - so this
    exercises the same resolution path Render will use.
    """
    return api_client.DiabetesApiClient()


def test_a_deployed_like_client_can_predict(deployed_like_client):
    result = deployed_like_client.predict(VALID_PAYLOAD, user_id="deploy-smoke")

    assert 0.0 <= result.probability <= 1.0
    assert result.prediction == int(result.probability >= result.threshold)
    assert result.risk_category in {"HIGH", "LOW"}


def test_a_deployed_like_client_can_explain(deployed_like_client):
    prediction = deployed_like_client.predict(VALID_PAYLOAD, user_id="deploy-smoke")

    explanation = deployed_like_client.explain(
        VALID_PAYLOAD, model_variant=prediction.model_variant
    )

    assert explanation.model_variant == prediction.model_variant
    assert explanation.contributions


def test_the_variant_is_chosen_by_the_backend(deployed_like_client):
    first = deployed_like_client.predict(VALID_PAYLOAD, user_id="stable-visitor")
    again = deployed_like_client.predict(VALID_PAYLOAD, user_id="stable-visitor")

    assert first.model_variant == again.model_variant in {"A", "B"}


def test_the_request_id_is_propagated(deployed_like_client):
    result = deployed_like_client.predict(VALID_PAYLOAD, user_id="deploy-smoke")

    assert result.request_id
    assert result.request_id != deployed_like_client.predict(
        VALID_PAYLOAD, user_id="deploy-smoke"
    ).request_id


def test_one_prediction_writes_exactly_one_row(deployed_like_client, isolated_db_path):
    before = len(inference_db.fetch_recent_logs(limit=500, db_path=isolated_db_path))

    result = deployed_like_client.predict(VALID_PAYLOAD, user_id="deploy-smoke")

    rows = inference_db.fetch_recent_logs(limit=500, db_path=isolated_db_path)
    assert len(rows) - before == 1
    assert rows[0]["request_id"] == result.request_id
    assert rows[0]["model_variant"] == result.model_variant


def test_the_base_url_used_is_the_configured_one(deployed_like_client, api_base_url):
    assert deployed_like_client.base_url == api_base_url.rstrip("/")


def test_the_blueprint_and_the_client_agree_on_the_variable_name():
    """A rename on either side would silently unwire the deployment."""
    blueprint = load_blueprint()
    keys = {item.get("key") for item in service(blueprint, PUBLIC_SERVICE).get("envVars", [])}

    assert api_client.ENV_BASE_URL in keys


def test_the_public_entrypoint_names_no_model_artifact():
    """Guard from I-A, restated so this track cannot regress it."""
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    for forbidden in ("model_bundle.pkl", "shap_explainer.pkl", "log_inference"):
        assert forbidden not in source


def test_no_front_end_module_contains_a_second_serving_path():
    """Only the API may score.

    Asserted against the AST, not the source text: both ui/api_client.py and
    ui/public_components.py mention predict_proba in a docstring precisely to
    say they do not call it, and a text search cannot tell the difference.
    Extends the I-A guard to admin_app.py, which it did not cover.
    """
    front_end = [
        REPO_ROOT / "streamlit_app.py",
        REPO_ROOT / "admin_app.py",
        *sorted((REPO_ROOT / "ui").glob("*.py")),
    ]

    scoring = {
        path.name: calls
        for path in front_end
        if (calls := scoring_calls_in(path))
    }

    assert scoring == {}, f"{scoring} perform inference outside the API"
