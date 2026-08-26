"""Provenance primitives, manifest verification and the legacy attestation.

Two things are being defended here.

Integrity: the verifier must actually fail when something no longer matches, not
print a warning and pass. Every tamper case below flips exactly one thing.

Honesty: the committed artifacts predate this system, so the attestation may
state only what is observable now. Any field describing historical lineage must
be an explicit null. Several tests exist purely to stop a future change from
quietly inventing that history.
"""
import json
import subprocess
import sys

import pytest

from conftest import ARTIFACTS_DIR, REPO_ROOT
from ml_core import provenance

ATTESTATION_PATH = REPO_ROOT / "provenance" / "legacy_artifact_attestation.json"

#: Artifacts the serving code loads. Cross-checked against the source below.
EXPECTED_SERVING_ROLES = {
    "model_bundle_variant_a", "model_bundle_variant_b",
    "shap_explainer_variant_a", "shap_explainer_variant_b",
    "drift_baseline_variant_a", "drift_baseline_variant_b",
    "metrics_variant_a", "metrics_variant_b",
}
EXPECTED_NON_SERVING_ROLES = {
    "unused_legacy_estimator", "unused_legacy_scaler",
    "unused_legacy_threshold", "training_output_not_served",
}

HISTORICAL_FIELDS = [
    "producer_git_sha", "training_run_id", "training_dataset_sha256",
    "training_started_at", "training_configuration", "training_environment",
]


@pytest.fixture
def sample_run(tmp_path):
    """A miniature completed training run: artifacts on disk plus a manifest."""
    root = tmp_path / "repo"
    (root / "model_artifacts").mkdir(parents=True)
    (root / "provenance").mkdir()

    (root / "model_artifacts" / "model_bundle.pkl").write_bytes(b"pretend-bundle")
    (root / "model_artifacts" / "metrics.json").write_text('{"roc_auc": 0.8}', encoding="utf-8")
    (root / "requirements.lock").write_text("scikit-learn==1.8.0\n", encoding="utf-8")
    (root / "data.csv").write_text("a,b,target\n1,2,0\n3,4,1\n1,2,0\n", encoding="utf-8")

    manifest_path = root / "provenance" / "manifest.json"
    provenance.emit_training_manifest(
        project_root=root,
        output_path=manifest_path,
        variant="A",
        model_name="test_model",
        dataset_path=root / "data.csv",
        target_column="target",
        feature_names=["a", "b"],
        training={"random_state": 42, "threshold_method": "youden_j"},
        evaluation={"test_metrics": {"roc_auc": 0.8}},
        artifact_specs=[
            ("model_bundle", root / "model_artifacts" / "model_bundle.pkl", True),
            ("metrics", root / "model_artifacts" / "metrics.json", True),
        ],
        source_files=[root / "requirements.lock"],
        lockfile=root / "requirements.lock",
    )
    return root, manifest_path


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite(path, manifest):
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ============================================================== primitives

def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    target = tmp_path / "f.bin"
    target.write_bytes(b"x" * (3 << 20))  # spans several read chunks

    assert provenance.sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_canonical_json_is_order_independent():
    a = provenance.sha256_canonical_json({"x": 1, "y": [1, 2]})
    b = provenance.sha256_canonical_json({"y": [1, 2], "x": 1})

    assert a == b


def test_canonical_json_is_sensitive_to_sequence_order():
    assert provenance.sha256_canonical_json([1, 2]) != provenance.sha256_canonical_json([2, 1])


def test_feature_fingerprint_is_order_sensitive():
    forward = provenance.fingerprint_features(["a", "b", "c"], "t")
    reordered = provenance.fingerprint_features(["a", "c", "b"], "t")

    assert forward["feature_schema_sha256"] != reordered["feature_schema_sha256"]
    assert forward["feature_count"] == 3


def test_feature_fingerprint_is_deterministic():
    first = provenance.fingerprint_features(["a", "b"], "t")
    second = provenance.fingerprint_features(["a", "b"], "t")

    assert first == second


def test_dataset_fingerprint_records_shape_not_contents(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b,target\n1,2,0\n3,4,1\n1,2,0\n", encoding="utf-8")

    fingerprint = provenance.fingerprint_dataset(csv, tmp_path, "target")

    assert fingerprint["rows"] == 3
    assert fingerprint["columns"] == 3
    assert fingerprint["column_names"] == ["a", "b", "target"]
    assert fingerprint["duplicate_rows"] == 1
    assert fingerprint["sha256"] == provenance.sha256_file(csv)
    assert fingerprint["path"] == "d.csv"
    # No row values anywhere in the fingerprint.
    assert "3,4,1" not in json.dumps(fingerprint)


def test_environment_fingerprint_uses_installed_metadata():
    import sklearn

    env = provenance.fingerprint_environment()

    assert env["packages"]["scikit-learn"] == sklearn.__version__
    assert env["python_version"] == __import__("platform").python_version()


def test_environment_fingerprint_captures_no_environment_variables(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "sup3r-secret-value")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host/db")

    rendered = json.dumps(provenance.fingerprint_environment())

    assert "sup3r-secret-value" not in rendered
    assert "postgresql://" not in rendered
    assert "ADMIN_PASSWORD" not in rendered


def test_relative_path_rejects_paths_outside_the_project(tmp_path):
    with pytest.raises(ValueError):
        provenance.relative_path(tmp_path / "elsewhere.txt", REPO_ROOT)


def test_embedded_versions_read_committed_artifacts_without_unpickling():
    """Byte scan only - reading bytes cannot execute code from a pickle."""
    lr = provenance.embedded_library_versions(ARTIFACTS_DIR / "model_bundle.pkl")
    xgb = provenance.embedded_library_versions(ARTIFACTS_DIR / "boosted_model_bundle.pkl")

    assert lr["sklearn"] == "1.8.0"
    assert xgb["sklearn"] == "1.8.0"
    assert xgb["xgboost"] == "3.0.4"


def test_embedded_versions_returns_none_rather_than_guessing(tmp_path):
    blank = tmp_path / "blank.pkl"
    blank.write_bytes(b"no version markers here")

    assert provenance.embedded_library_versions(blank) == {"sklearn": None, "xgboost": None}


# ========================================================== manifest basics

def test_manifest_verifies_when_nothing_has_changed(sample_run):
    root, manifest_path = sample_run

    assert provenance.verify_manifest_file(manifest_path, root) == []


def test_manifest_is_written_atomically_and_last(sample_run):
    root, manifest_path = sample_run

    assert manifest_path.is_file()
    # No temp files left behind.
    assert [p.name for p in manifest_path.parent.iterdir()] == ["manifest.json"]
    manifest = _load(manifest_path)
    # Every artifact hash matches the file as it finally sits on disk.
    for entry in manifest["artifacts"]:
        assert provenance.sha256_file(root / entry["path"]) == entry["sha256"]


def test_manifest_is_not_written_when_artifact_hashing_fails(tmp_path, monkeypatch):
    """A half-finished run must leave no valid-looking manifest."""
    root = tmp_path / "repo"
    (root / "model_artifacts").mkdir(parents=True)
    (root / "model_artifacts" / "a.pkl").write_bytes(b"x")
    output = root / "manifest.json"

    monkeypatch.setattr(
        provenance, "fingerprint_dataset",
        lambda *a, **k: (_ for _ in ()).throw(OSError("dataset vanished")),
    )

    with pytest.raises(OSError):
        provenance.emit_training_manifest(
            project_root=root, output_path=output, variant="A", model_name="m",
            dataset_path=root / "missing.csv", target_column="t", feature_names=["a"],
            training={}, evaluation={},
            artifact_specs=[("model_bundle", root / "model_artifacts" / "a.pkl", True)],
            source_files=[],
        )

    assert not output.exists()


def test_manifest_records_training_configuration(sample_run):
    manifest = _load(sample_run[1])

    assert manifest["provenance_type"] == provenance.TRAINING_RUN
    assert manifest["training"]["random_state"] == 42
    assert manifest["training"]["threshold_method"] == "youden_j"
    assert manifest["features"]["feature_names"] == ["a", "b"]
    assert manifest["environment"]["lockfile"]["path"] == "requirements.lock"


def test_manifest_contains_no_absolute_paths(sample_run):
    rendered = json.dumps(_load(sample_run[1]))

    assert str(sample_run[0]) not in rendered
    assert str(REPO_ROOT) not in rendered


def test_manifest_dirty_tree_is_representable_honestly(sample_run, monkeypatch):
    """A dirty run must never claim to have come from clean HEAD."""
    monkeypatch.setattr(
        provenance, "git_provenance",
        lambda _root: {"commit_sha": "abc123", "dirty": True, "branch": "wip"},
    )
    manifest = provenance.build_training_manifest(
        project_root=sample_run[0], variant="A", model_name="m",
        dataset={}, features={}, training={}, evaluation={}, artifacts=[],
        source_files=[],
    )

    assert manifest["run"]["git"]["dirty"] is True
    assert manifest["run"]["git"]["commit_sha"] == "abc123"


def test_git_provenance_reports_nulls_outside_a_repository(tmp_path):
    result = provenance.git_provenance(tmp_path)

    assert set(result) == {"commit_sha", "dirty", "branch"}
    if result["commit_sha"] is None:
        assert result["dirty"] is None


def test_identical_inputs_produce_identical_hashes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same")
    b.write_bytes(b"same")

    assert provenance.sha256_file(a) == provenance.sha256_file(b)
    assert (provenance.fingerprint_features(["x"], "y")["feature_schema_sha256"]
            == provenance.fingerprint_features(["x"], "y")["feature_schema_sha256"])


# ============================================================ tamper cases

def test_tamper_model_artifact_byte_change(sample_run):
    root, manifest_path = sample_run
    (root / "model_artifacts" / "model_bundle.pkl").write_bytes(b"pretend-bundlE")

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("sha256 mismatch" in p for p in problems)


def test_tamper_metrics_json_change(sample_run):
    root, manifest_path = sample_run
    (root / "model_artifacts" / "metrics.json").write_text('{"roc_auc": 0.99}', encoding="utf-8")

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("metrics.json" in p for p in problems)


def test_tamper_artifact_missing(sample_run):
    root, manifest_path = sample_run
    (root / "model_artifacts" / "model_bundle.pkl").unlink()

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("missing artifact" in p for p in problems)


def test_tamper_artifact_renamed_without_manifest_update(sample_run):
    root, manifest_path = sample_run
    src = root / "model_artifacts" / "model_bundle.pkl"
    src.rename(src.with_name("renamed_bundle.pkl"))

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("missing artifact" in p for p in problems)


def test_tamper_dataset_changes(sample_run):
    root, manifest_path = sample_run
    (root / "data.csv").write_text("a,b,target\n9,9,1\n", encoding="utf-8")

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("dataset sha256 mismatch" in p for p in problems)


def test_tamper_feature_ordering_changes(sample_run):
    root, manifest_path = sample_run
    manifest = _load(manifest_path)
    manifest["features"]["feature_names"] = ["b", "a"]
    _rewrite(manifest_path, manifest)

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("feature_schema_sha256" in p for p in problems)


def test_tamper_lockfile_changes(sample_run):
    root, manifest_path = sample_run
    (root / "requirements.lock").write_text("scikit-learn==1.9.0\n", encoding="utf-8")

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("lockfile sha256 mismatch" in p for p in problems)


def test_tamper_unsupported_schema_version(sample_run):
    root, manifest_path = sample_run
    manifest = _load(manifest_path)
    manifest["schema_version"] = 999
    _rewrite(manifest_path, manifest)

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("unsupported schema_version" in p for p in problems)


@pytest.mark.parametrize(
    "absolute",
    ["/etc/passwd", "C" + ":" + "\\Users\\dev\\model.pkl", "../outside.pkl"],
    ids=["posix-absolute", "windows-drive", "parent-escape"],
)
def test_tamper_absolute_developer_path(sample_run, absolute):
    root, manifest_path = sample_run
    manifest = _load(manifest_path)
    manifest["artifacts"][0]["path"] = absolute
    _rewrite(manifest_path, manifest)

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("must be relative" in p for p in problems)


def test_tamper_duplicate_artifact_roles(sample_run):
    root, manifest_path = sample_run
    manifest = _load(manifest_path)
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    _rewrite(manifest_path, manifest)

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("duplicate artifact role" in p for p in problems)
    assert any("duplicate artifact path" in p for p in problems)


def test_tamper_size_mismatch_alone_is_detected(sample_run):
    root, manifest_path = sample_run
    manifest = _load(manifest_path)
    manifest["artifacts"][0]["bytes"] = 999999
    _rewrite(manifest_path, manifest)

    problems = provenance.verify_manifest_file(manifest_path, root)

    assert any("size mismatch" in p for p in problems)


def test_unreadable_manifest_is_a_failure_not_a_pass(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    assert provenance.verify_manifest_file(broken, tmp_path) != []


# ================================================= committed attestation

def test_attestation_exists_and_verifies():
    assert ATTESTATION_PATH.is_file()

    assert provenance.verify_manifest_file(ATTESTATION_PATH, REPO_ROOT) == []


def test_attestation_is_not_labelled_as_a_training_run():
    """The critical integrity rule: no fabricated lineage for legacy artifacts."""
    attestation = _load(ATTESTATION_PATH)

    assert attestation["provenance_type"] == provenance.LEGACY_ATTESTATION
    assert attestation["provenance_type"] != provenance.TRAINING_RUN


@pytest.mark.parametrize("field", HISTORICAL_FIELDS)
def test_attestation_leaves_unprovable_history_null(field):
    unknown = _load(ATTESTATION_PATH)["unknown_history"]

    assert field in unknown
    assert unknown[field] is None, f"{field} must not be reconstructed"


def test_attestation_hashes_match_the_committed_artifacts():
    for entry in _load(ATTESTATION_PATH)["artifacts"]:
        target = REPO_ROOT / entry["path"]
        assert target.is_file(), entry["path"]
        assert provenance.sha256_file(target) == entry["sha256"], entry["path"]
        assert target.stat().st_size == entry["bytes"], entry["path"]


def test_attestation_covers_every_committed_artifact():
    listed = {entry["path"] for entry in _load(ATTESTATION_PATH)["artifacts"]}
    on_disk = {f"model_artifacts/{p.name}" for p in ARTIFACTS_DIR.iterdir() if p.is_file()}

    assert listed == on_disk


def test_attestation_serving_roles_match_the_serving_code():
    """A dead artifact must not silently become a serving dependency."""
    sources = "\n".join(
        (REPO_ROOT / name).read_text(encoding="utf-8")
        for name in ("app.py", "streamlit_app.py", "admin_app.py")
    )
    for entry in _load(ATTESTATION_PATH)["artifacts"]:
        filename = entry["path"].split("/")[-1]
        referenced = filename in sources
        assert entry["required_for_serving"] == referenced, (
            f"{filename}: attestation says served={entry['required_for_serving']} "
            f"but serving source reference={referenced}"
        )


def test_attestation_role_partition_is_as_expected():
    entries = _load(ATTESTATION_PATH)["artifacts"]
    served = {e["role"] for e in entries if e["required_for_serving"]}
    unserved = {e["role"] for e in entries if not e["required_for_serving"]}

    assert served == EXPECTED_SERVING_ROLES
    assert unserved == EXPECTED_NON_SERVING_ROLES


def test_attestation_records_observed_versions_for_served_bundles():
    by_role = {e["role"]: e for e in _load(ATTESTATION_PATH)["artifacts"]}

    assert by_role["model_bundle_variant_a"]["embedded_versions"]["sklearn"] == "1.8.0"
    assert by_role["model_bundle_variant_b"]["embedded_versions"]["sklearn"] == "1.8.0"
    assert by_role["model_bundle_variant_b"]["embedded_versions"]["xgboost"] == "3.0.4"


def test_attestation_flags_the_older_sklearn_dead_artifacts():
    """Track B found these carry scikit-learn 1.7.1 and are served by nothing."""
    by_role = {e["role"]: e for e in _load(ATTESTATION_PATH)["artifacts"]}

    for role in ("unused_legacy_estimator", "unused_legacy_scaler"):
        assert by_role[role]["embedded_versions"]["sklearn"] == "1.7.1"
        assert by_role[role]["required_for_serving"] is False


def test_attestation_labels_the_dataset_hash_as_current_not_historical():
    dataset = _load(ATTESTATION_PATH)["current_dataset"]

    assert dataset["sha256"] == provenance.sha256_file(REPO_ROOT / "cleaned_data.csv")
    assert "NOT established" in dataset["note"]


def test_attestation_states_integrity_is_not_provenance():
    statement = _load(ATTESTATION_PATH)["attestation"]["statement"]

    assert "NOT proof of training provenance" in statement


def test_attestation_contains_no_absolute_paths():
    rendered = ATTESTATION_PATH.read_text(encoding="utf-8")

    assert str(REPO_ROOT) not in rendered
    for entry in _load(ATTESTATION_PATH)["artifacts"]:
        assert not provenance._is_absolute_like(entry["path"])


# ==================================================================== CLIs

def _run_tool(*args, cwd=None):
    return subprocess.run(
        [sys.executable, *args], cwd=cwd or REPO_ROOT,
        capture_output=True, text=True, timeout=600,
    )


def test_attestation_check_mode_passes_against_committed_artifacts():
    result = _run_tool("tools/build_artifact_attestation.py", "--check")

    assert result.returncode == 0, result.stderr


def test_verifier_cli_exits_zero_when_everything_matches():
    result = _run_tool("tools/verify_provenance.py")

    assert result.returncode == 0, result.stderr
    assert "verify" in result.stdout


def test_verifier_cli_exits_nonzero_on_tampering(sample_run):
    root, manifest_path = sample_run
    (root / "model_artifacts" / "model_bundle.pkl").write_bytes(b"tampered")

    result = _run_tool(
        str(REPO_ROOT / "tools" / "verify_provenance.py"),
        str(manifest_path), "--project-root", str(root),
    )

    assert result.returncode != 0
    assert "FAILED" in result.stderr


def test_verifier_cli_reports_the_specific_problem(sample_run):
    root, manifest_path = sample_run
    (root / "model_artifacts" / "metrics.json").unlink()

    result = _run_tool(
        str(REPO_ROOT / "tools" / "verify_provenance.py"),
        str(manifest_path), "--project-root", str(root),
    )

    assert result.returncode != 0
    assert "missing artifact" in result.stderr


# ============================================ pipelines emit manifests

@pytest.mark.parametrize(
    ("pipeline", "manifest_name"),
    [("logisticregression_only.py", "training_manifest.json"),
     ("boostedtrees_ab.py", "boosted_training_manifest.json")],
)
def test_pipeline_emits_a_manifest_as_its_final_step(pipeline, manifest_name):
    source = (REPO_ROOT / pipeline).read_text(encoding="utf-8")

    assert f'PROVENANCE_PATH = ARTIFACTS_DIR / "{manifest_name}"' in source
    assert "provenance.emit_training_manifest(" in source

    # The manifest call must come after every artifact write in main(). Only
    # the write forms this pipeline actually uses are checked - boostedtrees_ab
    # writes no CSV, and rindex on an absent form would raise rather than fail.
    emit_at = source.index("provenance.emit_training_manifest(")
    writes = [w for w in ("joblib.dump(", "json.dump(", ".to_csv(") if w in source]
    assert writes, "expected the pipeline to write artifacts"
    for write in writes:
        assert source.rindex(write) < emit_at, f"{write} happens after the manifest"


@pytest.mark.parametrize(
    "pipeline", ["logisticregression_only.py", "boostedtrees_ab.py"]
)
def test_pipeline_manifest_path_follows_an_overridden_artifacts_dir(pipeline):
    """--artifacts-dir must relocate the manifest along with the artifacts."""
    source = (REPO_ROOT / pipeline).read_text(encoding="utf-8")

    assert source.count("PROVENANCE_PATH = ARTIFACTS_DIR / ") == 2
