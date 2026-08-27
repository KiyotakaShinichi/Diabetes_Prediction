"""Whether the admin dashboard tells the operational truth.

The H0 audit found several figures that could not mean what their labels said: a
"unique users" counter arithmetically identical to the request count, an A/B
panel that could only ever show one arm, and a drift check whose statistic made
it incapable of firing. Those are corrected here, and these tests exist so the
corrections cannot quietly regress.

The drift and KPI logic is deliberately written as plain functions over plain
data, so most of what follows needs no Streamlit at all. That is the direct
lesson of H1: an AppTest assertion that reaches nothing real passes forever
while proving nothing.
"""
import ast
import json
import sqlite3
import uuid

import pandas as pd
import pytest

import inference_db
from conftest import REPO_ROOT
from ui import admin_components

FEATURES = {
    "GenHlth": 3, "HighBP": 1, "BMI": 28.0, "HighChol": 0, "Age": 7,
    "DiffWalk": 0, "HeartDiseaseorAttack": 0, "PhysHlth": 2,
    "Education": 5, "PhysActivity": 1,
}

#: A two-feature baseline with realistic spread, used for the drift tests.
BASELINE = {
    "GenHlth": {"means": 2.84, "stds": 1.10},
    "BMI": {"means": 28.4, "stds": 6.60},
}


def baseline_stat(feature: str, statistic: str) -> float:
    return BASELINE[feature][statistic]


def rows_at(count: int, **values) -> list[dict]:
    """`count` identical production rows. Zero variance is fine: the statistic
    compares the production MEAN against the baseline, not its spread."""
    return [dict(values) for _ in range(count)]


def write_row(db_path, *, variant="A", prediction=0, assignment_key=None, probability=0.4):
    inference_db.log_inference(
        request_id=str(uuid.uuid4()),
        model_variant=variant,
        model_name="logistic_regression" if variant == "A" else "xgboost_boosted_trees",
        probability=probability,
        prediction=prediction,
        threshold=0.4557,
        payload=FEATURES,
        db_path=db_path,
        assignment_key=assignment_key,
    )


# ==================================================== schema migration

def legacy_database(path) -> None:
    """A database created before the assignment column existed."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                model_variant TEXT NOT NULL,
                model_name TEXT NOT NULL,
                probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                threshold REAL NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "INSERT INTO inference_logs (request_id, model_variant, model_name, "
            "probability, prediction, threshold, payload_json) VALUES (?,?,?,?,?,?,?)",
            ("legacy-1", "A", "logistic_regression", 0.4, 0, 0.4557, json.dumps(FEATURES)),
        )


def test_an_existing_database_gains_the_column_without_losing_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_database(db_path)

    inference_db.init_db(db_path)

    rows = inference_db.fetch_logs(10, db_path)
    assert len(rows) == 1
    assert rows[0]["request_id"] == "legacy-1"
    assert rows[0][inference_db.ASSIGNMENT_COLUMN] is None


def test_the_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_database(db_path)

    for _ in range(3):
        inference_db.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(inference_logs)")]

    assert columns.count(inference_db.ASSIGNMENT_COLUMN) == 1


def test_legacy_and_new_rows_coexist(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_database(db_path)

    write_row(db_path, assignment_key="visitor-1")

    rows = inference_db.fetch_logs(10, db_path)
    hashes = {row[inference_db.ASSIGNMENT_COLUMN] for row in rows}
    assert None in hashes, "the historical row must stay readable"
    assert len(hashes) == 2


def test_the_migration_never_drops_or_rewrites_data(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_database(db_path)
    with sqlite3.connect(db_path) as conn:
        before = conn.execute("SELECT payload_json, created_at FROM inference_logs").fetchall()

    inference_db.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        after = conn.execute("SELECT payload_json, created_at FROM inference_logs").fetchall()
    assert before == after


# ================================================ assignment identity

def test_the_same_assignment_key_always_hashes_the_same():
    assert inference_db.hash_assignment_key("visitor-1") == inference_db.hash_assignment_key("visitor-1")


def test_different_keys_hash_differently():
    assert inference_db.hash_assignment_key("visitor-1") != inference_db.hash_assignment_key("visitor-2")


def test_the_raw_assignment_key_is_never_stored(tmp_path):
    """Only a one-way digest reaches the database."""
    db_path = tmp_path / "logs.db"
    write_row(db_path, assignment_key="visitor-secret-value")

    with sqlite3.connect(db_path) as conn:
        dumped = " ".join(str(value) for row in conn.execute("SELECT * FROM inference_logs") for value in row)

    assert "visitor-secret-value" not in dumped
    assert inference_db.hash_assignment_key("visitor-secret-value") in dumped


def test_an_omitted_assignment_key_leaves_the_column_null(tmp_path):
    db_path = tmp_path / "logs.db"
    write_row(db_path, assignment_key=None)

    assert inference_db.fetch_logs(10, db_path)[0][inference_db.ASSIGNMENT_COLUMN] is None


def test_the_stored_digest_matches_the_key_the_router_used(client, isolated_db_path):
    """End to end: the API buckets on user_id and stores that key's digest."""
    response = client.post("/predict", params={"user_id": "subject-42"}, json=FEATURES)
    assert response.status_code == 200

    row = inference_db.fetch_logs(10, isolated_db_path)[0]
    assert row[inference_db.ASSIGNMENT_COLUMN] == inference_db.hash_assignment_key("subject-42")


def test_the_persisted_variant_is_the_served_variant(client, isolated_db_path):
    response = client.post("/predict", params={"user_id": "subject-42"}, json=FEATURES)

    row = inference_db.fetch_logs(10, isolated_db_path)[0]
    assert row["model_variant"] == response.json()["model_variant"]
    assert row["request_id"] == response.json()["request_id"]


def test_one_subject_keeps_one_variant_across_requests(client, isolated_db_path):
    for _ in range(4):
        client.post("/predict", params={"user_id": "stable-subject"}, json=FEATURES)

    variants = {row["model_variant"] for row in inference_db.fetch_logs(10, isolated_db_path)}
    assert len(variants) == 1, "a subject's assignment must not move between requests"


def test_distinct_subjects_reach_both_arms(client, isolated_db_path):
    """Not an exact split - just that assignment is not degenerate."""
    for index in range(30):
        client.post("/predict", params={"user_id": f"subject-{index}"}, json=FEATURES)

    variants = {row["model_variant"] for row in inference_db.fetch_logs(100, isolated_db_path)}
    assert variants == {"A", "B"}


# ============================================================ filters

@pytest.fixture
def populated_db(tmp_path):
    db_path = tmp_path / "logs.db"
    for index in range(10):
        write_row(
            db_path,
            variant="A" if index % 2 == 0 else "B",
            prediction=1 if index < 4 else 0,
            assignment_key=f"subject-{index % 5}",
        )
    return db_path


def test_variant_filtering_happens_in_the_query(populated_db):
    rows = inference_db.fetch_logs(100, populated_db, model_variant="A")

    assert rows, "expected variant A rows"
    assert {row["model_variant"] for row in rows} == {"A"}


def test_prediction_filtering_happens_in_the_query(populated_db):
    rows = inference_db.fetch_logs(100, populated_db, prediction=1)

    assert rows
    assert {row["prediction"] for row in rows} == {1}


def test_filters_compose(populated_db):
    rows = inference_db.fetch_logs(100, populated_db, model_variant="A", prediction=1)

    assert all(row["model_variant"] == "A" and row["prediction"] == 1 for row in rows)


def test_a_time_window_excludes_older_rows(tmp_path):
    db_path = tmp_path / "logs.db"
    write_row(db_path, assignment_key="recent")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO inference_logs (request_id, model_variant, model_name, probability, "
            "prediction, threshold, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("ancient", "A", "logistic_regression", 0.3, 0, 0.4557,
             json.dumps(FEATURES), "2020-01-01 00:00:00"),
        )

    recent = inference_db.fetch_logs(100, db_path, within_hours=24)

    assert {row["request_id"] for row in recent} == {
        row["request_id"] for row in recent if row["request_id"] != "ancient"
    }
    assert "ancient" not in {row["request_id"] for row in recent}


def test_an_unfiltered_query_matches_the_legacy_helper(populated_db):
    assert len(inference_db.fetch_logs(100, populated_db)) == len(
        inference_db.fetch_recent_logs(100, populated_db)
    )


def test_the_limit_is_applied(populated_db):
    assert len(inference_db.fetch_logs(3, populated_db)) == 3


# ======================================================== KPI truth

def code_only(path) -> str:
    """Source with docstrings and comments removed.

    These modules describe the defects they fixed, so a raw text search finds
    "unique_users" in the very docstring explaining that it is gone. ast.unparse
    drops comments, and the loop below drops docstrings, leaving executable code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_no_kpi_claims_to_count_people():
    """"Unique Requests" counted uuid4s; the variable was named unique_users."""
    source = code_only(REPO_ROOT / "ui" / "admin_components.py")

    assert "unique_users" not in source
    assert "Unique Requests" not in source
    assert "unique user" not in source.lower()


def test_the_subject_counter_counts_distinct_assignments(populated_db):
    rows = inference_db.fetch_logs(100, populated_db)

    distinct = {row[inference_db.ASSIGNMENT_COLUMN] for row in rows}
    assert len(distinct) == 5, "five assignment keys were written"
    assert len(rows) == 10, "and ten rows - the two must not be the same number"


def test_the_request_id_is_still_unique_per_row(populated_db):
    """Which is exactly why it could never have been a subject count."""
    rows = inference_db.fetch_logs(100, populated_db)

    assert len({row["request_id"] for row in rows}) == len(rows)


# ============================================================== drift

def test_an_unchanged_distribution_raises_no_alert():
    report = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(200, GenHlth=2.84, BMI=28.4)
    )

    assert report.status == "stable"
    assert report.drifted_features == ()


def test_a_clearly_shifted_distribution_raises_an_alert():
    report = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(200, GenHlth=4.2, BMI=28.4)
    )

    assert report.status == "drifted"
    assert [item.feature for item in report.drifted_features] == ["GenHlth"]


def test_a_shift_too_small_for_the_old_rule_is_now_detected():
    """The defect: |d| > 3 needed a three-sigma move, so it never fired.

    A quarter-sigma shift over 200 rows is overwhelming evidence, and the old
    comparison would have called it stable.
    """
    report = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(200, GenHlth=3.10, BMI=28.4)
    )
    shifted = next(item for item in report.features if item.feature == "GenHlth")

    assert abs(shifted.standardized_shift) < 3, "the old rule would not have fired"
    assert shifted.drifted, "the corrected test must fire"


def test_the_test_gets_more_sensitive_with_more_evidence():
    """The property the old statistic lacked entirely."""
    small = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(40, GenHlth=3.05, BMI=28.4)
    )
    large = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(4000, GenHlth=3.05, BMI=28.4)
    )

    small_p = next(i for i in small.features if i.feature == "GenHlth").p_value
    large_p = next(i for i in large.features if i.feature == "GenHlth").p_value
    assert large_p < small_p


def test_an_insufficient_sample_is_reported_as_such():
    report = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(5, GenHlth=4.9, BMI=55.0)
    )

    assert report.status == "insufficient_data"
    assert report.features == ()


def test_no_rows_at_all_is_insufficient_rather_than_stable():
    report = admin_components.assess_drift(list(BASELINE), baseline_stat, [])

    assert report.status == "insufficient_data"


def test_a_zero_variance_baseline_does_not_divide_by_zero():
    report = admin_components.assess_drift(
        ["X"], lambda f, s: 0.0 if s == "stds" else 1.0, [{"X": 5} for _ in range(50)]
    )

    assert report.features[0].z_statistic is None
    assert not report.features[0].drifted


def test_significance_is_corrected_for_multiple_features():
    report = admin_components.assess_drift(
        list(BASELINE), baseline_stat, rows_at(100, GenHlth=2.84, BMI=28.4)
    )

    assert report.corrected_alpha == pytest.approx(report.alpha / len(BASELINE))


@pytest.mark.parametrize("variant", ["A", "B"])
def test_both_committed_baseline_schemas_still_drive_the_test(variant):
    """Variant A is per-feature dicts; variant B is parallel arrays."""
    import joblib

    path = "drift_baseline.pkl" if variant == "A" else "boosted_drift_baseline.pkl"
    baseline = joblib.load(REPO_ROOT / "model_artifacts" / path)
    feature_cols, _n_train, get_stat = admin_components.drift_baseline_accessor(baseline)

    report = admin_components.assess_drift(
        feature_cols, get_stat, [dict(FEATURES) for _ in range(60)]
    )

    assert report.status in {"stable", "drifted"}
    assert len(report.features) == len(feature_cols)


def test_the_dashboard_no_longer_claims_statistical_significance_it_cannot_support():
    source = (REPO_ROOT / "ui" / "admin_components.py").read_text(encoding="utf-8")

    assert "DRIFT_Z_THRESHOLD" not in source
    assert "No significant drift detected" not in source


# ================================================= operational surface

def test_the_backend_label_names_the_engine_and_nothing_else(monkeypatch):
    monkeypatch.setattr(inference_db, "_get_database_url", lambda: "")
    assert inference_db.backend_name() == "SQLite"

    monkeypatch.setattr(
        inference_db, "_get_database_url",
        lambda: "postgresql://someone:secret@db.internal:5432/prod",
    )
    label = inference_db.backend_name()

    assert label == "PostgreSQL"
    for leak in ("secret", "someone", "db.internal", "5432", "prod", "://"):
        assert leak not in label


def test_the_hourly_series_keeps_quiet_hours(populated_db):
    """Dropping empty hours made sparse traffic look continuous."""
    frame = pd.DataFrame(inference_db.fetch_logs(100, populated_db))
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    frame.loc[0, "created_at"] = frame["created_at"].min() - pd.Timedelta(hours=5)

    hourly = admin_components.hourly_volume(frame)

    assert len(hourly) >= 6, "the gap hours must be present, not dropped"
    assert int(hourly["Requests"].sum()) == len(frame)


def test_the_hourly_series_is_empty_when_nothing_is_timestamped():
    frame = pd.DataFrame({"created_at": pd.to_datetime([None, None], errors="coerce")})

    assert admin_components.hourly_volume(frame).empty


def test_the_performance_section_shows_one_brier_pair_only():
    """It used to render brier_after twice, once on its own and once in a pair."""
    source = code_only(REPO_ROOT / "ui" / "admin_components.py")
    body = source.split("def render_performance_section", 1)[1].split("\ndef ", 1)[0]

    assert body.count("brier_after") == 1
    assert "Brier Score" not in body
