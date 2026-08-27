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
* nothing depends on the working directory: production code resolves
  ``model_artifacts/`` and ``data/`` from the project directory, and the
  ``foreign_cwd`` fixture below exists to prove it.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import inference_db
from app import app

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "model_artifacts"
# The runtime SQLite log lives here. It is deliberately untracked and absent
# from a fresh clone, so tests assert on the directory rather than one file.
REPO_DATA_DIR = REPO_ROOT / "data"

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


@pytest.fixture
def foreign_cwd(tmp_path, monkeypatch) -> Path:
    """Run a test from a temporary directory outside the repository.

    Application resources now resolve from the project directory rather than
    the process working directory, so this must change nothing.
    """
    outside = tmp_path / "somewhere-else"
    outside.mkdir()
    monkeypatch.chdir(outside)
    return outside


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


# ------------------------------------------------------- live serving fixtures
#
# The public Streamlit app reaches the model only over HTTP, so proving its
# behaviour needs a real server rather than a patched function. These fixtures
# provide one on loopback: no external network, no fixed port, and the same
# FastAPI application object the production entrypoint serves.


@pytest.fixture(scope="session")
def live_api_server(tmp_path_factory):
    """A real uvicorn server for the API, on an ephemeral loopback port.

    Session-scoped because starting it costs more than any single test. The
    inference log is redirected before startup so the application's lifespan
    initialisation cannot create the repository's runtime database; per-test
    writes land wherever :func:`isolated_db_path` points at the time.
    """
    import socket
    import threading
    import time

    import uvicorn

    original_db_path = inference_db.DB_PATH
    inference_db.DB_PATH = tmp_path_factory.mktemp("live-api") / "startup.db"

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        inference_db.DB_PATH = original_db_path
        pytest.fail("the local API server did not start")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        inference_db.DB_PATH = original_db_path


@pytest.fixture
def api_base_url(live_api_server, monkeypatch) -> str:
    """Point the public app at the live API for the duration of one test."""
    monkeypatch.setenv("DIABETES_API_BASE_URL", live_api_server)
    return live_api_server


@pytest.fixture
def unreachable_api_url(monkeypatch) -> str:
    """An address with nothing listening, for the connection-failure path.

    The port is bound to discover a free number and then released, so a
    connection attempt is refused rather than left hanging.
    """
    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("DIABETES_API_BASE_URL", url)
    return url


@pytest.fixture
def stub_api(monkeypatch):
    """A loopback HTTP server returning a scripted status and body.

    Used for backend states that cannot be provoked against the real service
    without damaging an artifact - a 503, a malformed body, or a response that
    never arrives. It is a genuine HTTP exchange, so the client's transport and
    status handling are exercised rather than stubbed out.
    """
    import json as json_module
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    #: Path -> scripted response. "*" is the fallback for unscripted paths, which
    #: lets a test fail one endpoint (say /explain) while another succeeds.
    script: dict[str, dict] = {"*": {"status": 200, "body": {}, "delay": 0.0, "raw": None}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            path = self.path.split("?", 1)[0]
            entry = script.get(path, script["*"])
            if entry["delay"]:
                import time as time_module

                time_module.sleep(entry["delay"])
            payload = (
                entry["raw"]
                if entry["raw"] is not None
                else json_module.dumps(entry["body"]).encode("utf-8")
            )
            self.send_response(entry["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            """Keep the test output clean."""

    class StubServer(ThreadingHTTPServer):
        """Teardown must not wait on a handler the test deliberately abandoned.

        ThreadingHTTPServer sets daemon_threads = True but inherits
        block_on_close = True, and with that combination server_close() JOINS
        every handler thread. The timeout test scripts a two-second delay and
        gives up on it after 0.3s, so the abandoned handler would hold teardown
        open and push its remaining sleep into whichever test ran next - real
        cross-test timing coupling rather than a race.

        Setting block_on_close = False lets the daemon handler finish on its
        own. handle_error is silenced because that handler will write to a
        socket the client has already closed, which is expected here and would
        otherwise print a traceback for a working test.
        """

        block_on_close = False
        daemon_threads = True

        def handle_error(self, request, client_address):
            """Expected when a test abandons a slow response."""

    server = StubServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("DIABETES_API_BASE_URL", url)

    def configure(path="*", *, status=200, body=None, delay=0.0, raw=None):
        script[path] = {"status": status, "body": body or {}, "delay": delay, "raw": raw}
        return url

    configure.url = url
    try:
        yield configure
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
