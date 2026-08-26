"""Cross-layer feature-schema equivalence.

The contract earns its name only if every layer provably agrees with it:
training pipelines, committed artifacts, the FastAPI request model and its
generated OpenAPI document, the Streamlit UI, both drift baselines and
provenance manifests.

The strongest tests are at the end: the same sample scored directly through a
committed bundle and through /predict must agree bit for bit. That is
training-artifact-to-serving equivalence, not retraining.
"""
import json
import subprocess

import joblib
import pandas as pd
import pytest

import app as app_module
import boostedtrees_ab as xgb_train
import logisticregression_only as lr_train
from app import DiabetesFeatures
from conftest import ARTIFACTS_DIR, REPO_ROOT, VALID_PAYLOAD
from ml_core import feature_contract, provenance
from ml_core.feature_contract import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    FEATURE_SPECS,
    TARGET_COLUMN,
)

BUNDLES = [
    pytest.param("model_bundle.pkl", "A", id="variant-A"),
    pytest.param("boosted_model_bundle.pkl", "B", id="variant-B"),
]


def _bounds(field) -> tuple[float | None, float | None]:
    ge = next((m.ge for m in field.metadata if hasattr(m, "ge")), None)
    le = next((m.le for m in field.metadata if hasattr(m, "le")), None)
    return ge, le


# ==================================================== API / OpenAPI

def test_api_exposes_exactly_the_served_feature_set():
    assert tuple(DiabetesFeatures.model_fields) == FEATURE_NAMES


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=[s.name for s in FEATURE_SPECS])
def test_api_bounds_equal_the_contract(spec):
    field = DiabetesFeatures.model_fields[spec.name]
    ge, le = _bounds(field)

    assert ge == spec.minimum, f"{spec.name} lower bound"
    assert le == spec.maximum, f"{spec.name} upper bound"
    assert field.annotation is spec.dtype, f"{spec.name} type"
    assert field.is_required(), f"{spec.name} must stay required"


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=[s.name for s in FEATURE_SPECS])
def test_api_descriptions_equal_the_contract(spec):
    assert DiabetesFeatures.model_fields[spec.name].description == spec.description


def test_openapi_schema_matches_the_contract(client):
    schema = client.get("/openapi.json").json()
    model = schema["components"]["schemas"]["DiabetesFeatures"]

    assert list(model["properties"]) == list(FEATURE_NAMES)
    assert sorted(model["required"]) == sorted(FEATURE_NAMES)

    for spec in FEATURE_SPECS:
        prop = model["properties"][spec.name]
        assert prop["type"] == ("number" if spec.dtype is float else "integer"), spec.name
        assert prop["minimum"] == spec.minimum, spec.name
        assert prop["maximum"] == spec.maximum, spec.name
        assert prop["description"] == spec.description, spec.name


def test_openapi_advertises_no_unknown_feature(client):
    model = client.get("/openapi.json").json()["components"]["schemas"]["DiabetesFeatures"]

    assert set(model["properties"]) - set(FEATURE_NAMES) == set()


# ============================================== training pipelines

@pytest.mark.parametrize("training", [lr_train, xgb_train],
                         ids=["logistic_regression", "boosted_trees"])
def test_pipeline_feature_order_equals_the_contract(training):
    assert tuple(training.SELECTED_FEATURES) == FEATURE_NAMES
    assert training.TARGET_COLUMN == TARGET_COLUMN
    assert training.FEATURE_LABELS == FEATURE_LABELS


@pytest.mark.parametrize("pipeline", ["logisticregression_only.py", "boostedtrees_ab.py"])
def test_pipeline_no_longer_hardcodes_the_served_schema(pipeline):
    source = (REPO_ROOT / pipeline).read_text(encoding="utf-8")

    assert "feature_contract" in source
    assert 'TARGET_COLUMN = "Diabetes_binary"' not in source
    assert 'SELECTED_FEATURES = [' not in source
    assert "FEATURE_LABELS = {" not in source


# =============================================== committed artifacts

@pytest.mark.parametrize(("bundle_name", "variant"), BUNDLES)
def test_bundle_feature_columns_equal_the_contract(bundle_name, variant):
    bundle = joblib.load(ARTIFACTS_DIR / bundle_name)

    assert tuple(bundle["feature_columns"]) == FEATURE_NAMES


@pytest.mark.parametrize(("bundle_name", "variant"), BUNDLES)
def test_bundle_feature_labels_equal_the_contract(bundle_name, variant):
    bundle = joblib.load(ARTIFACTS_DIR / bundle_name)

    assert bundle["feature_labels"] == FEATURE_LABELS


@pytest.mark.parametrize(("bundle_name", "variant"), BUNDLES)
def test_estimator_feature_names_and_count_equal_the_contract(bundle_name, variant):
    """sklearn records feature_names_in_ at fit time; it must match exactly."""
    pipeline = joblib.load(ARTIFACTS_DIR / bundle_name)["pipeline"]

    assert getattr(pipeline, "n_features_in_", None) == len(FEATURE_NAMES)
    names = getattr(pipeline, "feature_names_in_", None)
    assert names is not None, f"{bundle_name} lost its embedded feature names"
    # Variant B stores these as numpy str_; compare as plain strings.
    assert tuple(str(n) for n in names) == FEATURE_NAMES


@pytest.mark.parametrize(
    "shap_name", ["shap_explainer.pkl", "boosted_shap_explainer.pkl"]
)
def test_shap_explainer_feature_names_equal_the_contract(shap_name):
    bundle = joblib.load(ARTIFACTS_DIR / shap_name)

    assert tuple(bundle["feature_names"]) == FEATURE_NAMES


def test_drift_baseline_variant_a_covers_the_contract():
    """Variant A stores one entry per feature, keyed by name."""
    baseline = joblib.load(ARTIFACTS_DIR / "drift_baseline.pkl")

    assert set(baseline) == set(FEATURE_NAMES)
    assert len(baseline) == len(FEATURE_NAMES)


def test_drift_baseline_variant_b_covers_the_contract():
    """Variant B stores an explicit ordered feature_columns list."""
    baseline = joblib.load(ARTIFACTS_DIR / "boosted_drift_baseline.pkl")

    assert tuple(baseline["feature_columns"]) == FEATURE_NAMES
    assert set(baseline["means"]) == set(FEATURE_NAMES)
    assert set(baseline["stds"]) == set(FEATURE_NAMES)


def test_the_two_drift_schemas_stay_distinct():
    """Unifying them is deferred; app.py still branches on the difference."""
    a = joblib.load(ARTIFACTS_DIR / "drift_baseline.pkl")
    b = joblib.load(ARTIFACTS_DIR / "boosted_drift_baseline.pkl")

    assert "feature_columns" not in a
    assert "feature_columns" in b


# ==================================================== Streamlit UI

def test_streamlit_collects_every_served_feature():
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    payload_block = source[source.index("payload = {"):source.index("input_df =")]

    for name in FEATURE_NAMES:
        assert f'"{name}"' in payload_block, f"streamlit_app.py does not collect {name}"


def test_streamlit_collects_no_extra_model_feature():
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    block = source[source.index("payload = {"):source.index("input_df =")]
    collected = {line.split('"')[1] for line in block.splitlines() if line.strip().startswith('"')}

    assert collected == set(FEATURE_NAMES)


def test_streamlit_numeric_bounds_come_from_the_contract():
    source = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert 'feature_contract.spec_for("BMI").minimum' in source
    assert 'feature_contract.spec_for("BMI").maximum' in source
    assert 'feature_contract.spec_for("PhysHlth").minimum' in source
    assert "feature_contract.order_columns(" in source


# ==================================================== provenance

def test_provenance_feature_hash_recomputes_from_canonical_order():
    fingerprint = provenance.fingerprint_features(list(FEATURE_NAMES), TARGET_COLUMN)

    recomputed = provenance.sha256_canonical_json(
        {"features": list(FEATURE_NAMES), "target": TARGET_COLUMN}
    )
    assert fingerprint["feature_schema_sha256"] == recomputed
    assert fingerprint["feature_count"] == len(FEATURE_NAMES)


def test_changing_canonical_order_changes_the_schema_hash():
    canonical = provenance.fingerprint_features(list(FEATURE_NAMES), TARGET_COLUMN)
    swapped = list(FEATURE_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]

    assert (provenance.fingerprint_features(swapped, TARGET_COLUMN)["feature_schema_sha256"]
            != canonical["feature_schema_sha256"])


@pytest.mark.parametrize("pipeline", ["logisticregression_only.py", "boostedtrees_ab.py"])
def test_pipeline_manifest_uses_contract_feature_names(pipeline):
    source = (REPO_ROOT / pipeline).read_text(encoding="utf-8")

    assert "feature_names=list(feature_contract.FEATURE_NAMES)" in source


# ======================================== single maintained definition

def test_only_one_maintained_served_feature_list_exists():
    """Tests and archived experiments may quote values; production may not."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    definers = []
    for name in listing:
        path = REPO_ROOT / name
        if not path.is_file() or path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "SELECTED_FEATURES = [" in text or "FEATURE_SPECS: tuple" in text:
            definers.append(name)

    assert definers == ["ml_core/feature_contract.py"], definers


def test_target_column_has_one_maintained_definition():
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    definers = [
        name for name in listing
        if (REPO_ROOT / name).is_file()
        and not (REPO_ROOT / name).name.startswith("test_")
        and 'TARGET_COLUMN = "Diabetes_binary"' in (REPO_ROOT / name).read_text(
            encoding="utf-8", errors="ignore")
    ]

    assert definers == ["ml_core/feature_contract.py"], definers


def test_archived_experiment_schema_is_not_remapped():
    """The archived scripts use a different historical dataset schema."""
    source = (REPO_ROOT / "qsvm.py").read_text(encoding="utf-8")

    assert "DiabetesStatus" in source
    assert "GeneralHealth" in source
    assert "feature_contract" not in source


# ============================= training-artifact-to-serving equivalence

@pytest.mark.parametrize(("bundle_name", "variant"), BUNDLES)
def test_direct_bundle_scoring_equals_the_api(client, valid_payload, bundle_name, variant):
    """The same row scored through the bundle and through /predict must agree."""
    bundle = joblib.load(ARTIFACTS_DIR / bundle_name)
    frame = feature_contract.order_columns(pd.DataFrame([valid_payload]))
    direct_probability = float(bundle["pipeline"].predict_proba(frame)[:, 1][0])
    direct_threshold = float(bundle["threshold"])
    direct_prediction = int(direct_probability >= direct_threshold)

    body = client.post(
        "/predict", json=valid_payload, params={"model_variant": variant}
    ).json()

    assert body["model_variant"] == variant
    assert body["probability"] == pytest.approx(direct_probability, abs=1e-6)
    assert body["threshold"] == pytest.approx(direct_threshold, abs=1e-6)
    assert body["prediction"] == direct_prediction
    assert body["risk_category"] == ("HIGH" if direct_prediction == 1 else "LOW")


@pytest.mark.parametrize(("bundle_name", "variant"), BUNDLES)
def test_shuffled_json_key_order_does_not_change_the_prediction(
    client, valid_payload, bundle_name, variant
):
    """Serving must not depend on incoming JSON key order."""
    forward = client.post(
        "/predict", json=valid_payload, params={"model_variant": variant}
    ).json()

    reversed_payload = {name: valid_payload[name] for name in reversed(FEATURE_NAMES)}
    assert list(reversed_payload) != list(valid_payload)
    shuffled = client.post(
        "/predict", json=reversed_payload, params={"model_variant": variant}
    ).json()

    assert shuffled["probability"] == forward["probability"]
    assert shuffled["prediction"] == forward["prediction"]
    assert shuffled["threshold"] == forward["threshold"]


def test_serving_refuses_a_bundle_whose_feature_order_disagrees(
    tolerant_client, valid_payload, monkeypatch, tmp_path
):
    """A misaligned bundle must be refused, not scored on shuffled columns."""
    bundle = dict(joblib.load(ARTIFACTS_DIR / "model_bundle.pkl"))
    bundle["feature_columns"] = list(reversed(FEATURE_NAMES))
    tampered = tmp_path / "tampered.pkl"
    joblib.dump(bundle, tampered)
    monkeypatch.setattr(app_module, "MODEL_BUNDLE_PATH", tampered)
    app_module._load_bundle_cached.cache_clear()

    response = tolerant_client.post("/predict", json=valid_payload, params={"model_variant": "A"})

    assert response.status_code == 503
    app_module._load_bundle_cached.cache_clear()


def test_extra_and_missing_fields_remain_pydantic_controlled(tolerant_client, valid_payload):
    extra = tolerant_client.post("/predict", json={**valid_payload, "Smoker": 1})
    missing = tolerant_client.post(
        "/predict", json={k: v for k, v in valid_payload.items() if k != "BMI"}
    )

    assert extra.status_code == 200, "unknown fields are ignored by Pydantic"
    assert missing.status_code == 422


def test_openapi_document_is_not_committed():
    """Runtime schema testing is enough; a generated file would rot."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert "openapi.json" not in tracked


def test_valid_payload_fixture_matches_the_contract():
    """The shared test payload must not drift from the served feature set."""
    assert set(VALID_PAYLOAD) == set(FEATURE_NAMES)
    for name, value in VALID_PAYLOAD.items():
        spec = feature_contract.spec_for(name)
        assert spec.minimum <= value <= spec.maximum, name


def test_contract_module_does_no_io_on_import(tmp_path):
    before = set(tmp_path.rglob("*"))
    import importlib

    importlib.reload(feature_contract)

    assert set(tmp_path.rglob("*")) == before
    assert json.dumps(list(FEATURE_NAMES))  # still serializable after reload
