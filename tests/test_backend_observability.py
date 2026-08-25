"""Backend robustness and observability contract.

Covers the failure paths that previously had no deliberate behaviour: what the
client sees, what the operator can see in the logs, and what is guaranteed never
to leak. Nothing here changes a prediction, a threshold or a success schema.

Sanitization is the recurring assertion: an error response may carry a stable
message and a request id, never repr(exc), a traceback, an artifact path, a
database credential or raw SQL text.
"""
import logging
import sqlite3
import uuid
import warnings

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import ArtifactUnavailableError, InferenceError, ServiceError, app

# Substrings that must never appear in a client-facing error body.
LEAK_MARKERS = (
    "Traceback", "model_artifacts", ".pkl", "C:\\", "/home/", "sqlite3",
    "psycopg", "SELECT", "INSERT", "postgresql://", "password",
)


@pytest.fixture(autouse=True)
def _clear_bundle_cache():
    """The loader is cached by (path, mtime, size); keep tests independent."""
    app_module._load_bundle_cached.cache_clear()
    yield
    app_module._load_bundle_cached.cache_clear()


@pytest.fixture
def tolerant(isolated_db_path):
    """Client that surfaces unhandled errors as 500s instead of raising."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _assert_sanitized(body: dict) -> None:
    rendered = str(body)
    for marker in LEAK_MARKERS:
        assert marker not in rendered, f"error body leaked {marker!r}: {rendered}"
    assert "request_id" in body, "an error body must carry a correlation id"
    uuid.UUID(body["request_id"])


# =========================================================== lifespan (Phase 2)

def test_startup_is_a_lifespan_handler_not_the_deprecated_hook():
    source = (app_module.PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert 'on_event("startup")' not in source
    assert "lifespan=lifespan" in source
    assert "asynccontextmanager" in source


def test_no_on_event_deprecation_warning_is_emitted(isolated_db_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TestClient(app) as client:
            client.get("/health")

    offenders = [str(w.message) for w in caught if "on_event is deprecated" in str(w.message)]
    assert offenders == []


def test_lifespan_initialises_the_isolated_schema(isolated_db_path):
    assert not isolated_db_path.exists()

    with TestClient(app) as client:
        client.get("/health")

    assert isolated_db_path.is_file()
    with sqlite3.connect(isolated_db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "inference_logs" in tables


def test_importing_app_does_not_initialise_the_database(tmp_path):
    """Schema work belongs to startup, never to import.

    Run in a subprocess rather than via importlib.reload(): reloading app would
    rebind ServiceError to a new class object while this module still holds the
    old one, silently detaching the registered exception handler.
    """
    import subprocess
    import sys

    probe = tmp_path / "never" / "logs.db"
    runner = tmp_path / "import_probe.py"
    runner.write_text(
        "\n".join([
            "import pathlib",
            "import inference_db",
            f"inference_db.DB_PATH = pathlib.Path(r'{probe}')",
            "inference_db._get_database_url = lambda: ''",
            "import app",
            "print('imported')",
        ]),
        encoding="utf-8",
    )
    import os

    env = {**os.environ, "PYTHONPATH": str(app_module.PROJECT_ROOT), "DATABASE_URL": ""}
    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=app_module.PROJECT_ROOT, capture_output=True, text=True, timeout=300, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
    assert not probe.exists()
    assert not probe.parent.exists()


def test_lifespan_runs_exactly_once_per_client(isolated_db_path, caplog):
    caplog.set_level(logging.INFO, logger="diabetes_api")

    with TestClient(app) as client:
        client.get("/health")
        client.get("/health")

    starts = [r for r in caplog.records if getattr(r, "event", None) == "startup.complete"]
    assert len(starts) == 1


def test_lifespan_startup_failure_propagates(monkeypatch, isolated_db_path):
    """A broken dependency at startup must fail loudly, not start half-alive."""
    def boom():
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(app_module, "init_db", boom)

    with pytest.raises(sqlite3.OperationalError), TestClient(app) as client:
        client.get("/health")


# ================================================ artifact failures (Phase 7)

def test_missing_primary_bundle_returns_503_not_500(tolerant, valid_payload, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", tmp_path / "gone.pkl")

    response = tolerant.post("/predict", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 503
    _assert_sanitized(response.json())


def test_missing_boosted_bundle_falls_back_to_variant_a(client, valid_payload, monkeypatch, tmp_path):
    """Variant B is optional; the documented fallback must still work."""
    monkeypatch.setattr(app_module, "BOOSTED_BUNDLE_PATH", tmp_path / "gone.pkl")

    body = client.post("/predict", json=valid_payload, params={"model_variant": "B"}).json()

    assert body["fallback_to_A"] is True
    assert body["model_name"] == "logistic_regression"
    assert 0.0 <= body["probability"] <= 1.0


def test_corrupt_bundle_returns_503_and_leaks_nothing(tolerant, valid_payload, monkeypatch, tmp_path):
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"this is not a pickle at all")
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", corrupt)

    response = tolerant.post("/predict", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 503
    _assert_sanitized(response.json())


def test_bundle_missing_required_keys_returns_503(tolerant, valid_payload, monkeypatch, tmp_path):
    import joblib

    incomplete = tmp_path / "incomplete.pkl"
    joblib.dump({"model_name": "nonsense"}, incomplete)
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", incomplete)

    response = tolerant.post("/predict", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 503
    _assert_sanitized(response.json())


def test_scoring_failure_returns_500_with_a_safe_body(tolerant, valid_payload, monkeypatch):
    class ExplodingPipeline:
        def predict_proba(self, _frame):
            raise RuntimeError("BLAS failure in /opt/secret/path")

    real_loader = app_module.load_model_bundle

    def sabotaged(path):
        bundle = dict(real_loader(path))
        bundle["pipeline"] = ExplodingPipeline()
        return bundle

    monkeypatch.setattr(app_module, "load_model_bundle", sabotaged)

    response = tolerant.post("/predict", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 500
    body = response.json()
    _assert_sanitized(body)
    assert "BLAS" not in str(body)
    assert "/opt/secret/path" not in str(body)


def test_unreadable_shap_explainer_returns_503(tolerant, valid_payload, monkeypatch, tmp_path):
    broken = tmp_path / "broken_shap.pkl"
    broken.write_bytes(b"not a pickle")
    monkeypatch.setattr(app_module, "SHAP_PATH_A", broken)

    response = tolerant.post("/explain", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 503
    _assert_sanitized(response.json())


def test_unreadable_drift_baseline_returns_503(tolerant, valid_payload, monkeypatch, tmp_path):
    broken = tmp_path / "broken_drift.pkl"
    broken.write_bytes(b"not a pickle")
    monkeypatch.setattr(app_module, "DRIFT_BASELINE_A", broken)

    response = tolerant.post("/drift-check", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 503
    _assert_sanitized(response.json())


def test_absent_shap_explainer_keeps_its_documented_404(tolerant, valid_payload, monkeypatch, tmp_path):
    """A selector naming an artifact that does not exist stays a 404."""
    monkeypatch.setattr(app_module, "SHAP_PATH_B", tmp_path / "absent.pkl")

    response = tolerant.post("/explain", json=valid_payload, params={"model_variant": "B"})

    assert response.status_code == 404


# ================================================ readiness (Phases 6 and 7)

def test_ready_returns_200_when_the_service_can_serve(client):
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    names = {c["name"] for c in body["checks"]}
    assert "primary_model" in names
    assert "inference_log" in names


def test_ready_returns_503_when_the_primary_bundle_is_unusable(tolerant, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", tmp_path / "gone.pkl")

    response = tolerant.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    failed = [c for c in body["checks"] if not c["ok"]]
    assert failed and all("reason" in c for c in failed)
    _assert_no_leak_in(body)


def test_ready_stays_200_when_only_the_optional_variant_is_missing(client, monkeypatch, tmp_path):
    """Variant B is optional, so its absence must not gate traffic."""
    monkeypatch.setattr(app_module, "BOOSTED_BUNDLE_PATH", tmp_path / "gone.pkl")

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_returns_503_when_the_log_store_is_unreachable(tolerant, monkeypatch):
    def boom(*_a, **_k):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(app_module, "init_db", boom)

    response = tolerant.get("/ready")

    assert response.status_code == 503
    _assert_no_leak_in(response.json())


def test_ready_does_not_redeserialize_the_bundle_on_every_probe(client):
    """Probes must reuse the cached bundle rather than unpickling each time."""
    app_module._load_bundle_cached.cache_clear()

    client.get("/ready")
    first = app_module._load_bundle_cached.cache_info()
    client.get("/ready")
    client.get("/ready")
    later = app_module._load_bundle_cached.cache_info()

    assert later.misses == first.misses, "artifact was re-deserialized on a probe"
    assert later.hits > first.hits


def test_health_stays_cheap_and_always_alive(client, monkeypatch, tmp_path):
    """Liveness must not fail just because a dependency is down."""
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", tmp_path / "gone.pkl")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()) == {
        "status", "service", "model_bundle_exists", "boosted_bundle_exists",
    }


def _assert_no_leak_in(body: dict) -> None:
    rendered = str(body)
    for marker in LEAK_MARKERS:
        assert marker not in rendered, f"leaked {marker!r}"


# ================================================== database failures (Phase 8)

def test_telemetry_write_failure_does_not_discard_the_prediction(
    client, valid_payload, monkeypatch, caplog
):
    """The documented policy: analytics failure degrades, it never 500s.

    streamlit_app.py already states that logging must never break the result;
    the API now matches it.
    """
    caplog.set_level(logging.WARNING, logger="diabetes_api")

    def boom(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app_module, "log_inference", boom)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)

    degraded = [r for r in caplog.records if getattr(r, "event", None) == "inference.persist_failed"]
    assert len(degraded) == 1
    assert degraded[0].levelno == logging.WARNING
    assert degraded[0].request_id == body["request_id"]


def test_analytics_read_failure_returns_503_not_raw_sql(tolerant, monkeypatch):
    def boom(**_kwargs):
        raise sqlite3.OperationalError('no such table: inference_logs; SELECT * FROM secrets')

    monkeypatch.setattr(app_module, "fetch_recent_logs", boom)

    response = tolerant.get("/analytics-summary")

    assert response.status_code == 503
    _assert_sanitized(response.json())


def test_inference_log_read_failure_returns_503(tolerant, monkeypatch):
    def boom(**_kwargs):
        raise sqlite3.OperationalError("connection to postgresql://user:pw@host failed")

    monkeypatch.setattr(app_module, "fetch_recent_logs", boom)

    response = tolerant.get("/inference-logs")

    assert response.status_code == 503
    body = response.json()
    _assert_sanitized(body)
    assert "user:pw" not in str(body)


def test_malformed_historical_row_does_not_break_analytics(client, valid_payload, isolated_db_path):
    """The established resilient contract: skip the bad row, serve the rest."""
    client.post("/predict", json=valid_payload)
    with sqlite3.connect(isolated_db_path) as conn:
        conn.execute(
            "INSERT INTO inference_logs (request_id, model_variant, model_name, probability,"
            " prediction, threshold, payload_json) VALUES (?,?,?,?,?,?,?)",
            ("broken", "A", "logistic_regression", 0.4, 0, 0.5, "{not json"),
        )

    logs = client.get("/inference-logs")
    summary = client.get("/analytics-summary")

    assert logs.status_code == 200
    assert summary.status_code == 200
    assert logs.json()["count"] == 2
    assert summary.json()["count"] == 2
    broken = next(r for r in logs.json()["logs"] if r["request_id"] == "broken")
    assert broken["payload"] == {}


# ================================================ request correlation (Phase 9)

def test_request_id_is_consistent_across_response_row_and_log(
    client, valid_payload, isolated_db_path, caplog
):
    caplog.set_level(logging.INFO, logger="diabetes_api")

    body = client.post("/predict", json=valid_payload).json()
    request_id = body["request_id"]

    with sqlite3.connect(isolated_db_path) as conn:
        stored = [r[0] for r in conn.execute("SELECT request_id FROM inference_logs")]
    served = [r for r in caplog.records if getattr(r, "event", None) == "inference.served"]

    uuid.UUID(request_id)
    assert stored == [request_id]
    assert len(served) == 1
    assert served[0].request_id == request_id
    assert served[0].model_variant == body["model_variant"]

    logs = client.get("/inference-logs").json()["logs"]
    assert logs[0]["request_id"] == request_id


def test_each_request_gets_a_distinct_correlation_id(client, valid_payload):
    first = client.post("/predict", json=valid_payload).json()["request_id"]
    second = client.post("/predict", json=valid_payload).json()["request_id"]

    assert first != second


def test_failure_response_carries_a_correlatable_id(tolerant, valid_payload, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="diabetes_api")
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", tmp_path / "gone.pkl")

    body = tolerant.post("/predict", json=valid_payload, params={"model_variant": "A"}).json()

    failed = [r for r in caplog.records if getattr(r, "event", None) == "request.failed"]
    assert len(failed) == 1
    assert failed[0].request_id == body["request_id"]
    assert failed[0].path == "/predict"


# ======================================================== logging (Phase 10)

def test_successful_inference_logs_at_info_not_error(client, valid_payload, caplog):
    caplog.set_level(logging.DEBUG, logger="diabetes_api")

    client.post("/predict", json=valid_payload)

    served = [r for r in caplog.records if getattr(r, "event", None) == "inference.served"]
    assert len(served) == 1
    assert served[0].levelno == logging.INFO
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_health_probes_do_not_emit_errors(client, caplog):
    caplog.set_level(logging.DEBUG, logger="diabetes_api")

    for _ in range(5):
        client.get("/health")
        client.get("/ready")

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_internal_failure_is_logged_with_a_traceback_server_side(
    tolerant, valid_payload, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.ERROR, logger="diabetes_api")
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", tmp_path / "gone.pkl")

    response = tolerant.post("/predict", json=valid_payload, params={"model_variant": "A"})

    failed = [r for r in caplog.records if getattr(r, "event", None) == "request.failed"]
    assert len(failed) == 1
    assert failed[0].exc_info is not None, "the operator needs the traceback"
    assert failed[0].error_type == "ArtifactUnavailableError"
    assert failed[0].status_code == 503
    # ...but the client got none of it.
    _assert_sanitized(response.json())


def test_logs_never_contain_environment_secrets(client, valid_payload, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    secret = "sup3r-secret-admin-password"
    monkeypatch.setenv("ADMIN_PASSWORD", secret)
    monkeypatch.setenv("ADMIN_USERNAME", "alice")

    client.post("/predict", json=valid_payload)
    client.get("/ready")
    client.get("/analytics-summary")

    assert secret not in caplog.text
    assert "postgresql://" not in caplog.text


def test_inference_logs_do_not_dump_the_whole_feature_payload(client, valid_payload, caplog):
    """Operational logs carry identifiers and outcomes, not clinical records."""
    caplog.set_level(logging.DEBUG, logger="diabetes_api")

    client.post("/predict", json=valid_payload)

    served = next(r for r in caplog.records if getattr(r, "event", None) == "inference.served")
    assert not hasattr(served, "payload")
    assert "BMI" not in caplog.text
    assert str(valid_payload["BMI"]) not in caplog.text


def test_skipped_malformed_row_logs_at_warning_not_error(client, valid_payload, isolated_db_path, caplog):
    client.post("/predict", json=valid_payload)
    with sqlite3.connect(isolated_db_path) as conn:
        conn.execute(
            "INSERT INTO inference_logs (request_id, model_variant, model_name, probability,"
            " prediction, threshold, payload_json) VALUES (?,?,?,?,?,?,?)",
            ("broken", "A", "logistic_regression", 0.4, 0, 0.5, "{not json"),
        )
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="inference_db")

    client.get("/inference-logs")

    skipped = [r for r in caplog.records if "unreadable payload" in r.getMessage()]
    assert len(skipped) == 1
    assert skipped[0].levelno == logging.WARNING


# ================================================== taxonomy shape (Phase 4)

def test_taxonomy_is_small_and_maps_to_deliberate_statuses():
    assert issubclass(ArtifactUnavailableError, ServiceError)
    assert issubclass(InferenceError, ServiceError)
    assert ArtifactUnavailableError.status_code == 503
    assert InferenceError.status_code == 500
    assert app_module.LogStoreUnavailableError.status_code == 503

    subclasses = ServiceError.__subclasses__()
    assert len(subclasses) <= 4, "keep the taxonomy minimal"


@pytest.mark.parametrize(
    "error", [ArtifactUnavailableError, InferenceError, app_module.LogStoreUnavailableError]
)
def test_client_details_are_static_and_reveal_nothing(error):
    detail = error.client_detail

    assert detail
    for marker in LEAK_MARKERS:
        assert marker not in detail
    # The message must not be built from an exception argument.
    assert "%" not in detail and "{" not in detail
