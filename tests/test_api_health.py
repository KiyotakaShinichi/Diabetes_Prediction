"""Health-endpoint contract: availability, stable schema, artifact reporting."""
import sqlite3

from conftest import ARTIFACTS_DIR

EXPECTED_HEALTH_KEYS = {
    "status",
    "service",
    "model_bundle_exists",
    "boosted_bundle_exists",
}


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "Diabetes Risk Assessment API"


def test_health_schema_is_exact_and_typed(client):
    body = client.get("/health").json()

    assert set(body) == EXPECTED_HEALTH_KEYS
    assert isinstance(body["status"], str)
    assert isinstance(body["service"], str)
    assert isinstance(body["model_bundle_exists"], bool)
    assert isinstance(body["boosted_bundle_exists"], bool)


def test_health_is_deterministic_across_calls(client):
    first = client.get("/health").json()
    second = client.get("/health").json()

    assert first == second


def test_health_reports_committed_model_artifacts_as_present(client):
    """A fresh clone ships both bundles, so /health must not report them missing."""
    body = client.get("/health").json()

    assert (ARTIFACTS_DIR / "model_bundle.pkl").is_file()
    assert (ARTIFACTS_DIR / "boosted_model_bundle.pkl").is_file()
    assert body["model_bundle_exists"] is True
    assert body["boosted_bundle_exists"] is True


def test_startup_creates_schema_in_the_isolated_database(client, isolated_db_path):
    """The startup hook must build its schema in the temp DB, not the tracked one."""
    client.get("/health")

    assert isolated_db_path.is_file()
    with sqlite3.connect(isolated_db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "inference_logs" in tables
