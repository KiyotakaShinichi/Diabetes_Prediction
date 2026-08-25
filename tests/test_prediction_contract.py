"""Response contract for /predict, the A/B router, and the logging round-trip.

These tests exercise the real committed model bundles - they assert on the
*shape and invariants* of the response, never on a specific probability, so
they stay valid if the models are retrained.
"""
import hashlib
import sqlite3

import pytest

from app import choose_variant
from conftest import REPO_DATA_DIR, VALID_PAYLOAD, payload_with

EXPECTED_PREDICT_KEYS = {
    "request_id",
    "model_variant",
    "model_name",
    "fallback_to_A",
    "prediction",
    "risk_category",
    "probability",
    "threshold",
    "confidence_intervals",
    "calibration",
}


def test_predict_returns_200_with_the_full_response_contract(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert set(response.json()) == EXPECTED_PREDICT_KEYS


def test_probability_and_threshold_are_valid_probabilities(client, valid_payload):
    body = client.post("/predict", json=valid_payload).json()

    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["threshold"] <= 1.0


@pytest.mark.parametrize(
    "payload",
    [
        payload_with(GenHlth=1, HighBP=0, BMI=19.0, HighChol=0, Age=1, DiffWalk=0,
                     HeartDiseaseorAttack=0, PhysHlth=0, Education=6, PhysActivity=1),
        payload_with(GenHlth=5, HighBP=1, BMI=45.0, HighChol=1, Age=13, DiffWalk=1,
                     HeartDiseaseorAttack=1, PhysHlth=30, Education=1, PhysActivity=0),
        dict(VALID_PAYLOAD),
    ],
    ids=["low-risk-profile", "high-risk-profile", "baseline-profile"],
)
def test_prediction_and_risk_category_agree_with_the_threshold(client, payload):
    body = client.post("/predict", json=payload).json()

    assert body["prediction"] in (0, 1)
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    assert body["risk_category"] == ("HIGH" if body["prediction"] == 1 else "LOW")


def test_identical_payloads_produce_identical_scores_but_unique_request_ids(client, valid_payload):
    first = client.post("/predict", json=valid_payload).json()
    second = client.post("/predict", json=valid_payload).json()

    assert first["probability"] == second["probability"]
    assert first["prediction"] == second["prediction"]
    assert first["threshold"] == second["threshold"]
    assert first["request_id"] != second["request_id"]


def test_higher_risk_profile_does_not_score_below_lower_risk_profile(client):
    """Monotonicity sanity check - a clinically worse profile must not score lower."""
    low = client.post("/predict", json=payload_with(
        GenHlth=1, HighBP=0, BMI=19.0, HighChol=0, Age=1, DiffWalk=0,
        HeartDiseaseorAttack=0, PhysHlth=0, Education=6, PhysActivity=1,
    )).json()
    high = client.post("/predict", json=payload_with(
        GenHlth=5, HighBP=1, BMI=45.0, HighChol=1, Age=13, DiffWalk=1,
        HeartDiseaseorAttack=1, PhysHlth=30, Education=1, PhysActivity=0,
    )).json()

    assert high["probability"] > low["probability"]


@pytest.mark.parametrize(("requested", "expected"), [("A", "A"), ("B", "B"), ("a", "A"), ("b", "B")])
def test_explicit_variant_selection_is_honoured_and_case_insensitive(client, valid_payload, requested, expected):
    body = client.post("/predict", json=valid_payload, params={"model_variant": requested}).json()

    assert body["model_variant"] == expected
    assert body["fallback_to_A"] is False


def test_variant_a_serves_the_logistic_regression_bundle(client, valid_payload):
    body = client.post("/predict", json=valid_payload, params={"model_variant": "A"}).json()

    assert body["model_name"] == "logistic_regression"


@pytest.mark.parametrize("user_id", ["anonymous", "user-1", "user-2", "", "a-very-long-user-identifier"])
def test_auto_variant_matches_the_pure_assignment_function(client, valid_payload, user_id):
    body = client.post(
        "/predict", json=valid_payload, params={"model_variant": "auto", "user_id": user_id}
    ).json()

    assert body["model_variant"] == choose_variant(user_id)


@pytest.mark.parametrize("user_id", ["anonymous", "u1", "u2", "u3", "u4", "u5"])
def test_choose_variant_is_deterministic_and_binary(user_id):
    assert choose_variant(user_id) in {"A", "B"}
    assert choose_variant(user_id) == choose_variant(user_id)


def test_choose_variant_follows_the_documented_md5_rule():
    """Pin the hashing rule so an A/B reassignment can never happen by accident."""
    for user_id in ("anonymous", "user-1", "user-2", "seed"):
        digest = hashlib.md5(user_id.encode("utf-8"), usedforsecurity=False).hexdigest()
        expected = "A" if int(digest[-2:], 16) % 2 == 0 else "B"
        assert choose_variant(user_id) == expected


def test_choose_variant_splits_a_large_population_across_both_arms():
    assignments = {choose_variant(f"user-{i}") for i in range(200)}

    assert assignments == {"A", "B"}


def test_predict_writes_one_row_to_the_isolated_log(client, valid_payload, isolated_db_path):
    body = client.post("/predict", json=valid_payload).json()

    with sqlite3.connect(isolated_db_path) as conn:
        rows = conn.execute(
            "SELECT request_id, model_variant, probability, prediction FROM inference_logs"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == body["request_id"]
    assert rows[0][1] == body["model_variant"]
    assert rows[0][2] == pytest.approx(body["probability"], abs=1e-6)
    assert rows[0][3] == body["prediction"]


def test_logged_inference_round_trips_through_the_logs_endpoint(client, valid_payload):
    predicted = client.post("/predict", json=valid_payload).json()

    logs = client.get("/inference-logs", params={"limit": 10}).json()

    assert logs["count"] == 1
    entry = logs["logs"][0]
    assert entry["request_id"] == predicted["request_id"]
    assert entry["payload"] == valid_payload


def _repo_data_snapshot() -> dict[str, str] | None:
    """Hash every file in the repo's data/ dir, or None when it does not exist."""
    if not REPO_DATA_DIR.is_dir():
        return None
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(REPO_DATA_DIR.iterdir())
        if path.is_file()
    }


def test_predict_does_not_touch_the_repository_data_directory(client, valid_payload):
    """Inference logging is redirected to tmp_path, so data/ must be untouched.

    data/inference_logs.db is runtime state: untracked, and absent from a fresh
    clone. Asserting on the directory keeps this meaningful either way.
    """
    before = _repo_data_snapshot()

    client.post("/predict", json=valid_payload)

    assert _repo_data_snapshot() == before


def test_analytics_summary_is_well_formed_when_no_logs_exist(client):
    body = client.get("/analytics-summary").json()

    assert body == {
        "count": 0,
        "positive_rate": 0.0,
        "average_probability": 0.0,
        "by_variant": {},
        "by_model": {},
    }


def test_analytics_summary_aggregates_logged_inferences(client, valid_payload):
    for user_id in ("u1", "u2", "u3"):
        client.post("/predict", json=valid_payload, params={"user_id": user_id})

    body = client.get("/analytics-summary").json()

    assert body["count"] == 3
    assert 0.0 <= body["positive_rate"] <= 1.0
    assert 0.0 <= body["average_probability"] <= 1.0
    assert sum(body["by_variant"].values()) == 3
    assert sum(body["by_model"].values()) == 3
