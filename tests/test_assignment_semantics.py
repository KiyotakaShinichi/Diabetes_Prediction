"""Experiment assignment is optional, and an unassigned request stays unassigned.

The API used to default ``user_id`` to the literal "anonymous". Every caller
that omitted the parameter therefore collapsed into one shared pseudo-subject:
they all received the same deterministic variant, and every row persisted
SHA-256("anonymous") - a fixed, trivially precomputable digest standing in for
an experiment participant that did not exist. The public Streamlit app was
never affected, because it sends a random per-session UUID.

These tests pin the corrected contract from both directions: a supplied key
still routes and persists exactly as it did, and an omitted key produces NULL
rather than a placeholder digest.
"""
import hashlib
import pathlib
import sqlite3
import uuid

import pandas as pd
import pytest

import inference_db
from app import UNASSIGNED_VARIANT, choose_variant, normalize_assignment_key
from conftest import REPO_ROOT, VALID_PAYLOAD
from ui import api_client

PLACEHOLDER = "anonymous"


def predict(client, **params) -> dict:
    response = client.post("/predict", json=VALID_PAYLOAD, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def stored_rows(db_path) -> list[dict]:
    return inference_db.fetch_logs(100, db_path)


def dump_table(db_path) -> str:
    """Every stored value as one string, for "must not appear" assertions."""
    with sqlite3.connect(db_path) as conn:
        return " ".join(
            str(value)
            for row in conn.execute("SELECT * FROM inference_logs")
            for value in row
        )


# ------------------------------------------------------- a supplied key

def test_a_supplied_key_still_routes_deterministically(client):
    key = str(uuid.uuid4())

    first = predict(client, user_id=key, model_variant="auto")
    second = predict(client, user_id=key, model_variant="auto")

    assert first["model_variant"] == second["model_variant"] == choose_variant(key)


def test_a_supplied_key_persists_its_digest(client, isolated_db_path):
    key = str(uuid.uuid4())

    predict(client, user_id=key)

    row = stored_rows(isolated_db_path)[0]
    assert row[inference_db.ASSIGNMENT_COLUMN] == hashlib.sha256(key.encode()).hexdigest()


def test_the_raw_supplied_key_is_never_persisted(client, isolated_db_path):
    key = f"subject-{uuid.uuid4()}"

    predict(client, user_id=key)

    assert key not in dump_table(isolated_db_path)


def test_distinct_supplied_keys_remain_distinct_subjects(client, isolated_db_path):
    for _ in range(6):
        predict(client, user_id=str(uuid.uuid4()))

    digests = {row[inference_db.ASSIGNMENT_COLUMN] for row in stored_rows(isolated_db_path)}
    assert len(digests) == 6


# ------------------------------------------------------ an omitted key

def test_an_omitted_key_persists_null(client, isolated_db_path):
    predict(client)

    assert stored_rows(isolated_db_path)[0][inference_db.ASSIGNMENT_COLUMN] is None


def test_an_omitted_key_never_stores_the_placeholder_digest(client, isolated_db_path):
    """The specific defect: SHA-256 of the placeholder must not appear."""
    placeholder_digest = hashlib.sha256(PLACEHOLDER.encode()).hexdigest()

    predict(client)

    dumped = dump_table(isolated_db_path)
    assert placeholder_digest not in dumped
    assert PLACEHOLDER not in dumped


def test_many_unassigned_requests_do_not_become_one_subject(client, isolated_db_path):
    """They previously shared a digest, which the admin KPI counted as a subject."""
    for _ in range(5):
        predict(client)

    rows = stored_rows(isolated_db_path)
    assert len(rows) == 5
    assert all(row[inference_db.ASSIGNMENT_COLUMN] is None for row in rows)


def test_an_unassigned_request_is_served_by_the_documented_variant(client):
    body = predict(client, model_variant="auto")

    assert body["model_variant"] == UNASSIGNED_VARIANT


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_key_counts_as_unassigned(client, isolated_db_path, blank):
    """Sending an empty value says "no key"; it is not itself a key."""
    body = predict(client, user_id=blank, model_variant="auto")

    assert body["model_variant"] == UNASSIGNED_VARIANT
    assert stored_rows(isolated_db_path)[0][inference_db.ASSIGNMENT_COLUMN] is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("subject-1", "subject-1"),
        ("  subject-1  ", "subject-1"),
    ],
)
def test_key_normalisation(raw, expected):
    assert normalize_assignment_key(raw) == expected


# ---------------------------------------------- explicit placeholder is a key

def test_an_explicitly_supplied_placeholder_is_still_treated_as_a_key(client, isolated_db_path):
    """Backward compatibility: a caller that really sends it keeps working.

    It is a poor key - low entropy and shared - but honouring what a caller
    supplies is the API's job. What changed is that the server no longer
    invents this value on the caller's behalf.
    """
    body = predict(client, user_id=PLACEHOLDER, model_variant="auto")

    assert body["model_variant"] == choose_variant(PLACEHOLDER)
    assert stored_rows(isolated_db_path)[0][inference_db.ASSIGNMENT_COLUMN] == (
        hashlib.sha256(PLACEHOLDER.encode()).hexdigest()
    )


# ------------------------------------------------------- admin KPI effect

def test_unassigned_requests_do_not_inflate_the_subject_count(client, isolated_db_path):
    for _ in range(4):
        predict(client)
    predict(client, user_id=str(uuid.uuid4()))

    frame = pd.DataFrame(stored_rows(isolated_db_path))
    attributed = frame[inference_db.ASSIGNMENT_COLUMN].dropna()

    assert len(frame) == 5, "five requests were served"
    assert int(attributed.nunique()) == 1, "only one of them is an experiment subject"


# --------------------------------------------------- the client's own default

def test_the_client_manufactures_no_placeholder_key():
    """A client-side default would recreate the defect behind the API's back."""
    source = pathlib.Path(api_client.__file__).read_text(encoding="utf-8")

    assert '"anonymous"' not in source


def test_the_client_omits_the_parameter_when_no_key_is_given(stub_api):
    stub_api(
        "/predict",
        body={
            "request_id": "r1", "model_variant": "A", "model_name": "logistic_regression",
            "prediction": 0, "risk_category": "LOW", "probability": 0.2, "threshold": 0.45,
        },
    )

    result = api_client.DiabetesApiClient().predict(VALID_PAYLOAD)

    assert result.model_variant == "A"


def test_the_public_app_still_supplies_a_stable_uuid():
    """The Streamlit path keeps its per-session key rather than going unassigned."""
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "uuid.uuid4()" in source
    assert "user_id=visitor_id()" in source


# --------------------------------------------------------- legacy rows

def test_legacy_rows_without_the_column_remain_readable(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL, model_variant TEXT NOT NULL,
                model_name TEXT NOT NULL, probability REAL NOT NULL,
                prediction INTEGER NOT NULL, threshold REAL NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "INSERT INTO inference_logs (request_id, model_variant, model_name, probability, "
            "prediction, threshold, payload_json) VALUES (?,?,?,?,?,?,?)",
            ("legacy", "A", "logistic_regression", 0.4, 0, 0.4557, "{}"),
        )

    rows = inference_db.fetch_logs(10, db_path)

    assert len(rows) == 1
    assert rows[0][inference_db.ASSIGNMENT_COLUMN] is None
