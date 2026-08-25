"""
Shared pytest fixtures for the Diabetes Prediction test suite.

This file lives at the repository root on purpose: pytest prepends the
directory containing the root ``conftest.py`` to ``sys.path``, which is what
makes the flat top-level modules (``app``, ``inference_db``, the training
scripts) importable from ``tests/``.

Isolation guarantees provided here:

* every test talks to a throw-away SQLite file under ``tmp_path`` instead of
  the tracked ``data/inference_logs.db``;
* PostgreSQL/Neon resolution is short-circuited, so a developer or CI runner
  with ``DATABASE_URL`` exported can never point the suite at a real database;
* the working directory is pinned to the repository root, because production
  code resolves ``model_artifacts/`` and ``data/`` relative to the CWD.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import inference_db
from app import app

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "model_artifacts"
TRACKED_DB_PATH = REPO_ROOT / "data" / "inference_logs.db"

# A payload that satisfies every DiabetesFeatures constraint. Tests that want an
# invalid request start from this and corrupt exactly one field.
VALID_PAYLOAD: dict[str, float] = {
    "GenHlth": 3,
    "HighBP": 1,
    "BMI": 28.5,
    "HighChol": 1,
    "Age": 9,
    "DiffWalk": 0,
    "HeartDiseaseorAttack": 0,
    "PhysHlth": 5,
    "Education": 4,
    "PhysActivity": 1,
}


@pytest.fixture(scope="session", autouse=True)
def _pin_working_directory():
    """Production code uses CWD-relative paths, so pin it to the repo root."""
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    yield REPO_ROOT
    os.chdir(previous)


@pytest.fixture
def isolated_db_path(tmp_path, monkeypatch) -> Path:
    """Redirect all inference logging to a temporary SQLite file."""
    db_path = tmp_path / "inference_logs.db"
    monkeypatch.setattr(inference_db, "DB_PATH", db_path)
    # Belt and braces: unset the env var *and* stub the resolver so neither a
    # stray environment nor a local .streamlit/secrets.toml can select Postgres.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(inference_db, "_get_database_url", lambda: "")
    return db_path


@pytest.fixture(autouse=True)
def _enforce_db_isolation(isolated_db_path: Path) -> Path:
    """Apply :func:`isolated_db_path` to every test, requested or not."""
    return isolated_db_path


@pytest.fixture
def client(isolated_db_path: Path):
    """TestClient that re-raises unhandled server errors as test failures."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tolerant_client(isolated_db_path: Path):
    """TestClient that surfaces unhandled errors as real 500 responses.

    Used by tests asserting "this must be a 4xx, never a 5xx" - with the
    default client an unhandled exception propagates instead of producing a
    status code, which would make such assertions untestable.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return dict(VALID_PAYLOAD)


def payload_with(**overrides) -> dict:
    """Return the valid payload with specific fields replaced."""
    return {**VALID_PAYLOAD, **overrides}


def payload_without(field: str) -> dict:
    """Return the valid payload with one required field removed."""
    payload = dict(VALID_PAYLOAD)
    del payload[field]
    return payload
