"""Request-validation contract for POST /predict.

Every rejection must be a deliberate 4xx produced by Pydantic or an explicit
HTTPException - never an unhandled 500. The Pydantic constraints in
``app.DiabetesFeatures`` are the specification these tests pin down; none of
them are relaxed to make a test pass.
"""
import pytest

from conftest import VALID_PAYLOAD, payload_with, payload_without

BINARY_FIELDS = [
    "HighBP",
    "HighChol",
    "DiffWalk",
    "HeartDiseaseorAttack",
    "PhysActivity",
]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("GenHlth", 0),
        ("GenHlth", 6),
        ("GenHlth", -1),
        ("BMI", 9.9),
        ("BMI", 80.1),
        ("BMI", 0),
        ("Age", 0),
        ("Age", 14),
        ("PhysHlth", -1),
        ("PhysHlth", 31),
        ("Education", 0),
        ("Education", 7),
    ],
)
def test_out_of_range_values_are_rejected_with_422(tolerant_client, field, value):
    response = tolerant_client.post("/predict", json=payload_with(**{field: value}))

    assert response.status_code == 422, f"{field}={value} produced {response.status_code}"
    assert field in str(response.json()["detail"])


@pytest.mark.parametrize("field", BINARY_FIELDS)
@pytest.mark.parametrize("value", [2, -1, 99])
def test_invalid_binary_values_are_rejected_with_422(tolerant_client, field, value):
    response = tolerant_client.post("/predict", json=payload_with(**{field: value}))

    assert response.status_code == 422


@pytest.mark.parametrize("field", sorted(VALID_PAYLOAD))
def test_missing_required_field_is_rejected_with_422(tolerant_client, field):
    response = tolerant_client.post("/predict", json=payload_without(field))

    assert response.status_code == 422
    assert field in str(response.json()["detail"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("BMI", "heavy"),
        ("GenHlth", "excellent"),
        ("Age", None),
        ("HighBP", [1]),
        ("PhysHlth", {"days": 5}),
    ],
)
def test_wrong_types_are_rejected_with_422(tolerant_client, field, value):
    response = tolerant_client.post("/predict", json=payload_with(**{field: value}))

    assert response.status_code == 422


@pytest.mark.parametrize("body", [[], ["GenHlth"], "a string", 42, None])
def test_malformed_json_bodies_fail_safely(tolerant_client, body):
    """Non-object bodies must be refused by validation, not crash the handler."""
    response = tolerant_client.post("/predict", json=body)

    assert response.status_code == 422


def test_non_json_body_fails_safely(tolerant_client):
    response = tolerant_client.post(
        "/predict",
        content=b"GenHlth=3&BMI=28",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_empty_body_is_rejected_with_422(tolerant_client):
    response = tolerant_client.post("/predict")

    assert response.status_code == 422


def test_empty_object_body_reports_every_missing_field(tolerant_client):
    response = tolerant_client.post("/predict", json={})

    assert response.status_code == 422
    missing = {tuple(err["loc"])[-1] for err in response.json()["detail"]}
    assert missing == set(VALID_PAYLOAD)


@pytest.mark.parametrize("variant", ["C", "logistic", "", "AB"])
def test_unknown_model_variant_is_a_400_not_a_500(tolerant_client, variant):
    response = tolerant_client.post(
        "/predict", json=dict(VALID_PAYLOAD), params={"model_variant": variant}
    )

    assert response.status_code == 400
    assert "model_variant" in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("GenHlth", 1),
        ("GenHlth", 5),
        ("BMI", 10),
        ("BMI", 80),
        ("Age", 1),
        ("Age", 13),
        ("PhysHlth", 0),
        ("PhysHlth", 30),
        ("Education", 1),
        ("Education", 6),
    ],
)
def test_inclusive_boundary_values_are_accepted(client, field, value):
    response = client.post("/predict", json=payload_with(**{field: value}))

    assert response.status_code == 200


def test_unknown_extra_fields_are_ignored(client):
    """Pydantic's default is to ignore extras; pin that so it cannot regress silently."""
    response = client.post("/predict", json=payload_with(Smoker=1, unexpected="x"))

    assert response.status_code == 200
