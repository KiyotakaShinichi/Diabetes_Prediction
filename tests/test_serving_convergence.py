"""One serving path, and proof that the public app no longer has a second one.

Before this track the Streamlit app loaded ``model_bundle.pkl`` and called
``predict_proba`` itself. The API did the same job with validation, canonical
feature ordering, bundle checking, A/B routing, request correlation, sanitised
errors and persistence. Two implementations, one of them unguarded.

Three things are proved here:

* equivalence - the API returns exactly what the old direct scoring returned,
  for both variants, on fixtures captured from the pre-migration code;
* transport - the client maps every backend failure to a message that is safe
  to show a visitor and useless to an attacker;
* ownership - the public UI contains no scoring path at all, asserted against
  the AST rather than by reading source text.

Every HTTP exchange in this module is real and on loopback: either the actual
FastAPI application served by uvicorn, or a scripted stub server. Nothing is
monkeypatched into the request path, so none of it can pass vacuously.
"""
import ast
import json
from pathlib import Path

import pytest

from conftest import REPO_ROOT, VALID_PAYLOAD
from ml_core import feature_contract
from ui import api_client

#: Captured from the direct-scoring implementation at 73b13e5, before the
#: public app stopped loading the bundle. Committed so the equivalence claim
#: can be re-checked rather than taken on trust.
BASELINE_PATH = Path(__file__).parent / "fixtures" / "serving_equivalence_baseline.json"
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

#: Modules that make up the public front end. None of them may score.
PUBLIC_UI_SOURCES = (
    REPO_ROOT / "streamlit_app.py",
    REPO_ROOT / "ui" / "api_client.py",
    REPO_ROOT / "ui" / "public_components.py",
    REPO_ROOT / "ui" / "theme.py",
    REPO_ROOT / "ui" / "formatting.py",
)

#: Calls that mean "this module is doing inference itself".
#:
#: Bare ``predict`` is deliberately NOT listed: ``client.predict(...)`` is the
#: API call this track exists to introduce. Nothing is lost by omitting it,
#: because a local estimator would have to be deserialised first, and
#: test_no_public_ui_module_deserialises_a_model forbids that separately.
SCORING_CALLS = ("predict_proba", "shap_values", "transform")

#: Names that mean "this module is deserialising a model".
LOADING_CALLS = ("joblib", "pickle")


@pytest.fixture
def client(api_base_url) -> api_client.DiabetesApiClient:
    """A client pointed at the real API running on loopback."""
    return api_client.DiabetesApiClient()


# =================================================== Phase 10: equivalence

@pytest.mark.parametrize("variant", ["A", "B"])
def test_the_api_reproduces_the_old_direct_scoring(client, variant):
    """Same probability, prediction and category as the code this replaced."""
    expected = BASELINE["variants"][variant]

    for features, want in zip(BASELINE["fixtures"], expected["results"], strict=True):
        got = client.predict(features, model_variant=variant)

        assert got.probability == pytest.approx(want["probability"], abs=1e-6)
        assert got.prediction == want["prediction"]
        assert got.risk_category == want["risk_category"]


@pytest.mark.parametrize("variant", ["A", "B"])
def test_the_threshold_still_comes_from_the_bundle(client, variant):
    expected = BASELINE["variants"][variant]

    got = client.predict(BASELINE["fixtures"][0], model_variant=variant)

    assert got.threshold == pytest.approx(expected["threshold"], abs=1e-6)
    assert got.model_name == expected["model_name"]


def test_the_two_variants_are_genuinely_different_models(client):
    """A baseline that accidentally captured one model twice would prove nothing."""
    a = client.predict(BASELINE["fixtures"][0], model_variant="A")
    b = client.predict(BASELINE["fixtures"][0], model_variant="B")

    assert a.model_name != b.model_name
    assert a.threshold != b.threshold


def test_the_classification_is_the_api_s_own_threshold_comparison(client):
    for features in BASELINE["fixtures"]:
        got = client.predict(features)

        assert got.prediction == int(got.probability >= got.threshold)
        assert got.risk_category == ("HIGH" if got.prediction else "LOW")


# ================================================ Phase 8: A/B is the backend's

def test_the_backend_chooses_the_variant(client):
    """Deterministic per visitor, and not decided by the UI."""
    first = client.predict(VALID_PAYLOAD, user_id="visitor-one")
    again = client.predict(VALID_PAYLOAD, user_id="visitor-one")

    assert first.model_variant == again.model_variant
    assert first.model_variant in {"A", "B"}


def test_automatic_routing_reaches_both_variants(client):
    """A hardcoded variant would show up here as a single-valued set."""
    seen = {
        client.predict(VALID_PAYLOAD, user_id=f"visitor-{index}").model_variant
        for index in range(40)
    }

    assert seen == {"A", "B"}


def test_the_client_never_derives_a_variant_itself():
    source = (REPO_ROOT / "ui" / "api_client.py").read_text(encoding="utf-8")

    assert "choose_variant" not in source
    assert "md5" not in source.lower()


# ================================================== Phase 5: explanations

def test_the_explanation_covers_every_contract_feature(client):
    explanation = client.explain(VALID_PAYLOAD, model_variant="A")

    assert set(explanation.by_feature()) == set(feature_contract.FEATURE_NAMES)
    assert explanation.model_variant == "A"


@pytest.mark.parametrize("variant", ["A", "B"])
def test_an_explanation_describes_the_variant_that_was_asked_for(client, variant):
    """An explanation from the wrong model would be quietly misleading."""
    explanation = client.explain(VALID_PAYLOAD, model_variant=variant)

    assert explanation.model_variant == variant


def test_explanation_values_are_plain_numbers(client):
    """No pickled objects, no arrays - just what a UI can render."""
    explanation = client.explain(VALID_PAYLOAD, model_variant="A")

    for contribution in explanation.contributions:
        assert isinstance(contribution.feature, str)
        assert isinstance(contribution.value, float)
        assert isinstance(contribution.shap_value, float)


# ================================================ Phase 6: failure semantics

def test_a_refused_connection_is_reported_as_unavailable(unreachable_api_url):
    with pytest.raises(api_client.ApiUnavailableError) as caught:
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)

    assert "not reachable" in caught.value.user_message


def test_a_slow_backend_is_reported_as_a_timeout(stub_api, monkeypatch):
    monkeypatch.setenv("DIABETES_API_TIMEOUT_SECONDS", "0.3")
    stub_api(delay=2.0, body={"request_id": "never-arrives"})

    with pytest.raises(api_client.ApiTimeoutError) as caught:
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)

    assert "took too long" in caught.value.user_message


@pytest.mark.parametrize("status", [400, 422])
def test_a_rejected_payload_is_reported_as_a_validation_failure(stub_api, status):
    stub_api(status=status, body={"detail": "value is not a valid integer"})

    with pytest.raises(api_client.ApiValidationError):
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)


def test_a_real_out_of_range_payload_is_rejected_by_the_api(client):
    """Not a stub: the live API's own validation, reached over HTTP."""
    with pytest.raises(api_client.ApiValidationError):
        client.predict({**VALID_PAYLOAD, "BMI": 999})


def test_an_unavailable_model_is_reported_as_such(stub_api):
    stub_api(status=503, body={"detail": "Model artifact unavailable.", "request_id": "r-1"})

    with pytest.raises(api_client.ModelUnavailableError) as caught:
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)

    assert caught.value.request_id == "r-1"


def test_an_unexpected_status_is_reported_generically(stub_api):
    stub_api(status=500, body={"detail": "boom", "request_id": "r-2"})

    with pytest.raises(api_client.ApiUnexpectedError) as caught:
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)

    assert caught.value.request_id == "r-2"


def test_an_unparseable_body_does_not_crash_the_client(stub_api):
    stub_api(raw=b"<html>gateway error</html>")

    with pytest.raises(api_client.ApiUnexpectedError):
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)


def test_a_response_missing_required_fields_is_rejected(stub_api):
    """A truncated 200 must not become a half-populated result."""
    stub_api(body={"probability": 0.5})

    with pytest.raises(api_client.ApiUnexpectedError):
        api_client.DiabetesApiClient().predict(VALID_PAYLOAD)


@pytest.mark.parametrize(
    "error_type",
    [
        api_client.ApiUnavailableError,
        api_client.ApiTimeoutError,
        api_client.ApiValidationError,
        api_client.ModelUnavailableError,
        api_client.ApiUnexpectedError,
    ],
)
def test_no_error_message_leaks_internals(error_type):
    """Nothing a visitor sees may name a path, a host, a driver or a traceback."""
    message = error_type().user_message.lower()

    for leak in ("traceback", "sqlite", "postgres", "psycopg", ".pkl", "c:\\",
                 "/home/", "exception", "127.0.0.1", "joblib"):
        assert leak not in message


def test_the_error_taxonomy_shares_one_base():
    for error_type in (
        api_client.ApiUnavailableError,
        api_client.ApiTimeoutError,
        api_client.ApiValidationError,
        api_client.ModelUnavailableError,
        api_client.ApiUnexpectedError,
    ):
        assert issubclass(error_type, api_client.ApiError)


# ==================================================== Phase 3: configuration

def test_the_base_url_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv(api_client.ENV_BASE_URL, raising=False)

    assert api_client.resolve_base_url() == api_client.DEFAULT_BASE_URL


def test_the_base_url_is_environment_configurable(monkeypatch):
    monkeypatch.setenv(api_client.ENV_BASE_URL, "https://api.example.test/")

    assert api_client.resolve_base_url() == "https://api.example.test"


def test_an_explicit_base_url_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(api_client.ENV_BASE_URL, "https://ignored.example.test")

    assert api_client.resolve_base_url("http://127.0.0.1:9000") == "http://127.0.0.1:9000"


@pytest.mark.parametrize("raw", ["", "   ", "not-a-number", "0", "-5"])
def test_a_malformed_timeout_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv(api_client.ENV_TIMEOUT, raw)

    assert api_client.resolve_timeout() == api_client.DEFAULT_TIMEOUT_SECONDS


def test_the_timeout_is_environment_configurable(monkeypatch):
    monkeypatch.setenv(api_client.ENV_TIMEOUT, "2.5")

    assert api_client.resolve_timeout() == 2.5


def test_no_deployment_url_is_hardcoded():
    """Only loopback may appear; a real host belongs in configuration."""
    source = (REPO_ROOT / "ui" / "api_client.py").read_text(encoding="utf-8")

    assert "onrender.com" not in source
    for host in ("herokuapp", "amazonaws", "azurewebsites", "ngrok"):
        assert host not in source


# ============================== Phase 12: the UI owns no inference any more

def scoring_calls_in(path: Path) -> set[str]:
    """Attribute and function calls that would mean local inference."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in SCORING_CALLS:
            found.add(name)
    return found


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("path", PUBLIC_UI_SOURCES, ids=lambda p: p.name)
def test_no_public_ui_module_scores_anything(path):
    assert scoring_calls_in(path) == set(), f"{path.name} performs inference"


@pytest.mark.parametrize("path", PUBLIC_UI_SOURCES, ids=lambda p: p.name)
def test_no_public_ui_module_deserialises_a_model(path):
    modules = imported_modules(path)

    for loader in LOADING_CALLS:
        assert loader not in modules, f"{path.name} imports {loader}"


@pytest.mark.parametrize("path", PUBLIC_UI_SOURCES, ids=lambda p: p.name)
def test_no_public_ui_module_imports_the_backend(path):
    """Importing app.py would put the model back in the UI process."""
    assert "app" not in imported_modules(path) - {"streamlit_app"}


def test_the_public_entrypoint_has_no_model_paths():
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "model_bundle.pkl" not in source
    assert "shap_explainer.pkl" not in source


def test_the_authoritative_scoring_path_is_the_api():
    """The one place predict_proba may be called for serving."""
    assert "predict_proba" in scoring_calls_in(REPO_ROOT / "app.py")


def test_the_entrypoint_reaches_the_model_only_through_the_client():
    modules = imported_modules(REPO_ROOT / "streamlit_app.py")

    assert "ui" in modules
    assert "joblib" not in modules
    assert "sklearn" not in modules
    assert "pandas" not in modules


def test_the_public_app_no_longer_writes_inference_telemetry():
    """Persistence belongs to the API; a second writer would double-count."""
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "log_inference" not in source
    assert "inference_db" not in imported_modules(REPO_ROOT / "streamlit_app.py")
