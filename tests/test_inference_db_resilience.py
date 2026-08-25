"""Error handling in inference_db after the broad excepts were narrowed.

Two properties are in tension and both must hold:

* the log listing stays resilient - one malformed historical payload must not
  fail an entire admin request;
* but the catch is now narrow, so a genuine programming error surfaces instead
  of being silently swallowed.
"""
import logging
import sqlite3

import pytest

import inference_db

#: The real resolver, captured before the autouse fixture below stubs it out.
REAL_GET_DATABASE_URL = inference_db._get_database_url


@pytest.fixture(autouse=True)
def local_sqlite(tmp_path, monkeypatch):
    """Every test here uses a temporary SQLite file and never PostgreSQL."""
    db = tmp_path / "logs.db"
    monkeypatch.setattr(inference_db, "DB_PATH", db)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(inference_db, "_get_database_url", lambda: "")
    return db


@pytest.fixture
def real_url_resolver(monkeypatch):
    """Restore the genuine _get_database_url for tests that exercise it."""
    monkeypatch.setattr(inference_db, "_get_database_url", REAL_GET_DATABASE_URL)


def _insert_raw_payload(db, payload_json, request_id="r1"):
    """Write a row directly, bypassing log_inference's json.dumps."""
    inference_db.init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO inference_logs (request_id, model_variant, model_name, "
            "probability, prediction, threshold, payload_json) VALUES (?,?,?,?,?,?,?)",
            (request_id, "A", "logistic_regression", 0.5, 1, 0.5, payload_json),
        )


# ------------------------------------------------------------ _decode_payload

def test_valid_payload_round_trips():
    assert inference_db._decode_payload('{"BMI": 28.5}') == {"BMI": 28.5}


@pytest.mark.parametrize(
    "raw",
    ["not json", "", "{unclosed", None, 42, b"\xff\xfe", "[1, 2, 3]", '"a string"', "null"],
    ids=["garbage", "empty", "truncated", "none", "int", "bytes", "list", "string", "null"],
)
def test_unusable_payload_degrades_to_an_empty_dict(raw):
    """Never raise, never return a non-dict."""
    assert inference_db._decode_payload(raw) == {}


def test_decode_failure_logs_the_row_id_but_not_the_payload(caplog):
    caplog.set_level(logging.DEBUG)
    secret_ish = "patient-identifier-that-should-not-be-logged"

    inference_db._decode_payload(f"not json {secret_ish}", row_id=7)

    assert "7" in caplog.text
    assert secret_ish not in caplog.text


# --------------------------------------------- resilient listing, narrow catch

def test_one_malformed_row_does_not_fail_the_whole_listing(local_sqlite):
    """The admin analytics view must survive historical bad data."""
    inference_db.log_inference(
        request_id="good-1", model_variant="A", model_name="logistic_regression",
        probability=0.7, prediction=1, threshold=0.5, payload={"BMI": 30.0},
        db_path=local_sqlite,
    )
    _insert_raw_payload(local_sqlite, "{ this is not json", request_id="bad-1")
    inference_db.log_inference(
        request_id="good-2", model_variant="B", model_name="xgboost_boosted_trees",
        probability=0.2, prediction=0, threshold=0.5, payload={"BMI": 21.0},
        db_path=local_sqlite,
    )

    rows = inference_db.fetch_recent_logs(limit=10, db_path=local_sqlite)

    assert len(rows) == 3
    by_id = {row["request_id"]: row for row in rows}
    assert by_id["good-1"]["payload"] == {"BMI": 30.0}
    assert by_id["good-2"]["payload"] == {"BMI": 21.0}
    assert by_id["bad-1"]["payload"] == {}
    # payload_json is always consumed, never leaked alongside the decoded value.
    assert all("payload_json" not in row for row in rows)


def test_empty_payload_column_is_tolerated(local_sqlite):
    """payload_json is NOT NULL in the schema, so empty string is the edge case."""
    _insert_raw_payload(local_sqlite, "", request_id="empty-1")

    rows = inference_db.fetch_recent_logs(limit=10, db_path=local_sqlite)

    assert len(rows) == 1
    assert rows[0]["payload"] == {}


def test_a_missing_database_is_created_rather_than_failing(local_sqlite):
    assert not local_sqlite.exists()

    rows = inference_db.fetch_recent_logs(limit=5, db_path=local_sqlite)

    assert rows == []
    assert local_sqlite.is_file()


def test_a_genuine_database_error_is_not_swallowed(local_sqlite, monkeypatch):
    """Narrowing means real failures propagate instead of vanishing."""
    def broken_connect(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite3, "connect", broken_connect)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        inference_db.fetch_recent_logs(limit=5, db_path=local_sqlite)


# ------------------------------------------------- optional Streamlit secrets

def test_database_url_prefers_the_environment(monkeypatch, real_url_resolver):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@host/db")

    assert inference_db._get_database_url() == "postgresql://user@host/db"
    assert inference_db._use_postgres() is True


def test_absent_streamlit_secrets_falls_back_to_local_sqlite(monkeypatch, caplog, real_url_resolver):
    """No secrets.toml must mean local SQLite, not a crash."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    class _Boom:
        @property
        def secrets(self):
            raise RuntimeError("no secrets file")

    monkeypatch.setitem(__import__("sys").modules, "streamlit", _Boom())

    assert inference_db._get_database_url() == ""
    assert inference_db._use_postgres() is False


def test_secrets_failure_logs_only_the_exception_type(monkeypatch, caplog, real_url_resolver):
    caplog.set_level(logging.DEBUG)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    secret_url = "postgresql://real-user:real-password@real-host/db"

    class _Leaky:
        @property
        def secrets(self):
            raise RuntimeError(secret_url)

    monkeypatch.setitem(__import__("sys").modules, "streamlit", _Leaky())

    inference_db._get_database_url()

    assert secret_url not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u@h/d", True),
        ("postgres://u@h/d", True),
        ("", False),
        ("sqlite:///local.db", False),
        ("mysql://u@h/d", False),
    ],
)
def test_use_postgres_only_for_postgres_urls(monkeypatch, url, expected):
    monkeypatch.setattr(inference_db, "_get_database_url", lambda: url)

    assert inference_db._use_postgres() is expected
