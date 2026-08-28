"""The benchmark runner, its artifacts, and the boundary it must not cross.

The smoke run here is the same code path the full research run uses, with tiny
configurations. That is the point: CI proves the plumbing end to end in seconds
without ever presenting the result as a finding.

The boundary tests are the important ones. Track K writes only under its own
research root, never touches model_artifacts/ or the production attestation, and
records that fact in every manifest.
"""
import ast
import dataclasses
import hashlib
import json

import numpy as np
import pytest

from conftest import REPO_ROOT
from research.track_k import artifacts, benchmark, protocol

pytest.importorskip("torch", reason="PyTorch is a Track K research dependency")

RESEARCH_PACKAGE = REPO_ROOT / "research" / "track_k"

#: The recompute tests assert on point estimates and on integrity, never on
#: interval width, so they prove the same thing at a fraction of the cost.
RESAMPLES = 50


def _hash_tree(directory) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


@pytest.fixture(scope="module")
def production_before() -> dict[str, str]:
    """The deployed artifacts as they stood before any benchmark ran."""
    return _hash_tree(REPO_ROOT / "model_artifacts")


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, production_before):
    """One smoke benchmark, reused by every test in this module."""
    root = tmp_path_factory.mktemp("track_k_smoke")
    manifest = benchmark.run(smoke=True, output_root=root)
    return manifest, root / manifest["track_k_run_id"]


# ================================================ the production boundary

def _executable_strings(source: str) -> list[str]:
    """Every string literal a module can actually act on.

    Docstrings are excluded deliberately. Several Track K modules explain in
    prose that they must never touch ``model_artifacts/``, and a naive substring
    search would flag exactly the documentation that states the rule. What
    matters is whether a path the code can open names the production directory,
    so this looks at the literals the interpreter evaluates and nothing else.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


@pytest.mark.parametrize(
    "module",
    [
        "protocol.py", "split.py", "evaluation.py", "comparison.py",
        "calibration.py", "baselines.py", "challengers.py", "benchmark.py",
        "artifacts.py", "deep/models.py", "deep/training.py", "deep/preprocessing.py",
    ],
)
def test_no_research_module_names_the_production_artifact_directory(module):
    """A research run must be incapable of overwriting a deployed model."""
    source = (RESEARCH_PACKAGE / module).read_text(encoding="utf-8")

    for literal in _executable_strings(source):
        assert "model_artifacts" not in literal, f"{module} can address the production directory"
        assert "legacy_artifact_attestation" not in literal
        assert "provenance/" not in literal


def test_the_docstring_exclusion_would_still_catch_a_real_reference():
    """The AST filter must not be so permissive that it catches nothing."""
    offending = '''"""A docstring naming model_artifacts is allowed."""
path = "model_artifacts/bundle.joblib"
'''

    literals = _executable_strings(offending)

    assert any("model_artifacts" in literal for literal in literals)


def test_a_run_writes_nothing_into_the_production_namespace(smoke_run):
    manifest, run_dir = smoke_run

    assert manifest["production_artifacts_touched"] is False
    written = {path.name for path in run_dir.rglob("*") if path.is_file()}
    assert written, "the run produced no artifacts"
    for name in written:
        assert not name.startswith("model_bundle")
        assert not name.startswith("boosted_model_bundle")


def test_the_committed_production_artifacts_are_untouched_by_a_run(
    smoke_run, production_before
):
    """Hashed before the run, compared after. Bit-for-bit, or the test fails."""
    _manifest, _run_dir = smoke_run

    assert production_before, "no production artifacts were found to protect"
    assert _hash_tree(REPO_ROOT / "model_artifacts") == production_before


# ==================================================== the run itself

def test_every_family_is_benchmarked(smoke_run):
    manifest, _run_dir = smoke_run

    assert set(manifest["models"]) == set(protocol.MODEL_FAMILIES)


def test_the_manifest_records_the_frozen_protocol(smoke_run):
    manifest, _run_dir = smoke_run

    assert manifest["protocol_version"] == protocol.PROTOCOL_VERSION
    assert manifest["evaluation"]["primary_metric"] == protocol.PRIMARY_METRIC
    assert manifest["split"]["seed"] == protocol.SPLIT_SEED


def test_a_smoke_run_declares_itself(smoke_run):
    """Smoke numbers must be impossible to mistake for research results."""
    manifest, _run_dir = smoke_run

    assert manifest["smoke"] is True
    assert "NOT research results" in manifest["warning"]


def test_the_summary_labels_a_smoke_run(smoke_run):
    manifest, run_dir = smoke_run

    text = benchmark.summarise(manifest, run_dir)

    assert "SMOKE RUN" in text
    for family in protocol.MODEL_FAMILIES:
        assert family in text


def test_every_model_records_its_seed_and_configuration(smoke_run):
    manifest, _run_dir = smoke_run

    for family, record in manifest["models"].items():
        assert record["seed"] == protocol.model_seed(family)
        assert record["config"], f"{family} recorded no configuration"
        assert 0.0 <= record["threshold"] <= 1.0


def test_deep_models_record_a_parameter_count(smoke_run):
    manifest, _run_dir = smoke_run

    for family in protocol.DEEP_FAMILIES:
        assert manifest["models"][family]["parameter_count"] > 0


def test_every_model_records_its_calibration_decision(smoke_run):
    manifest, _run_dir = smoke_run

    for family, record in manifest["models"].items():
        decision = record["calibration"]
        assert decision["fitted_on"] == "validation"
        assert decision["method"] in {"none", "sigmoid", "isotonic"}
        assert decision["reason"], f"{family} gave no reason for its calibration choice"


def test_the_manifest_carries_the_dataset_limitations(smoke_run):
    """The base-rate caveat travels with the result, not only in the docs."""
    manifest, _run_dir = smoke_run

    joined = " ".join(manifest["limitations"]).lower()
    assert "base rate" in joined
    assert "population disease probability" in joined


def test_environment_capture_includes_the_deep_learning_stack(smoke_run):
    manifest, _run_dir = smoke_run

    assert manifest["environment"]["torch_version"]
    assert manifest["environment"]["torch_cuda_available"] is False


# ================================================== artifacts and integrity

def test_every_declared_artifact_exists_and_matches_its_hash(smoke_run):
    manifest, run_dir = smoke_run

    assert artifacts.verify_run_manifest(manifest, run_dir) == []


def test_a_tampered_artifact_is_detected(smoke_run):
    manifest, run_dir = smoke_run
    target = run_dir / manifest["models"]["logistic_regression"]["artifacts"]["model"]
    original = target.read_bytes()

    try:
        target.write_bytes(original + b"tampered")
        problems = artifacts.verify_run_manifest(manifest, run_dir)
    finally:
        target.write_bytes(original)

    assert any("no longer matches its hash" in problem for problem in problems)


def test_a_missing_artifact_is_detected(smoke_run):
    manifest, run_dir = smoke_run
    target = run_dir / manifest["models"]["mlp"]["artifacts"]["checkpoint"]
    original = target.read_bytes()

    try:
        target.unlink()
        problems = artifacts.verify_run_manifest(manifest, run_dir)
    finally:
        target.write_bytes(original)

    assert any("missing artifact" in problem for problem in problems)


def test_a_protocol_version_change_invalidates_a_recorded_run(smoke_run):
    manifest, run_dir = smoke_run
    altered = dict(manifest)
    altered["protocol_version"] = "0.0.1"

    problems = artifacts.verify_run_manifest(altered, run_dir)

    assert any("protocol_version" in problem for problem in problems)


def test_a_run_claiming_to_touch_production_is_rejected(smoke_run):
    manifest, run_dir = smoke_run
    altered = dict(manifest)
    altered["production_artifacts_touched"] = True

    problems = artifacts.verify_run_manifest(altered, run_dir)

    assert any("production artifacts" in problem for problem in problems)


def test_per_model_metrics_and_predictions_are_written(smoke_run):
    manifest, run_dir = smoke_run

    for family in protocol.MODEL_FAMILIES:
        metrics = json.loads((run_dir / f"{family}_metrics.json").read_text(encoding="utf-8"))
        assert metrics["family"] == family
        assert "test_metrics" in metrics and "validation_metrics" in metrics
        assert metrics["reliability_bins"], "no reliability data persisted"
        proba = np.load(run_dir / f"{family}_test_proba.npy")
        assert proba.shape == (manifest["split_sizes"]["test"],)


def test_deep_models_persist_a_learning_curve(smoke_run):
    manifest, run_dir = smoke_run

    for family in protocol.DEEP_FAMILIES:
        curve = json.loads(
            (run_dir / manifest["models"][family]["artifacts"]["learning_curve"]).read_text(
                encoding="utf-8"
            )
        )
        assert curve["history"], f"{family} recorded no learning curve"
        for point in curve["history"]:
            assert {"epoch", "train_loss", "val_loss", "val_roc_auc"} <= set(point)


# ==================================================== comparison and policy

def test_every_prespecified_comparison_is_reported(smoke_run):
    manifest, _run_dir = smoke_run

    reported = {(item["challenger"], item["baseline"]) for item in manifest["comparisons"]}
    assert reported == set(benchmark.comparison.default_pairs())


def test_every_comparison_carries_an_allowed_outcome(smoke_run):
    manifest, _run_dir = smoke_run

    for item in manifest["comparisons"]:
        assert item["outcome"] in protocol.COMPARISON_OUTCOMES
        assert item["delta"]["ci_lower"] <= item["delta"]["ci_upper"]


def test_the_promotion_baseline_is_a_classical_family(smoke_run):
    """A deep challenger is measured against the best classical model."""
    manifest, _run_dir = smoke_run

    assert manifest["promotion"]["baseline"] in protocol.CLASSICAL_FAMILIES


def test_each_challenger_receives_a_verdict_with_reasons(smoke_run):
    manifest, _run_dir = smoke_run

    for challenger in protocol.DEEP_FAMILIES:
        decision = manifest["promotion"]["decisions"][challenger]
        assert decision["verdict"] in protocol.VERDICTS
        assert len(decision["reasons"]) == 4, "every gate must be reported"


def test_the_bootstrap_is_recorded_as_paired(smoke_run):
    manifest, _run_dir = smoke_run

    assert manifest["bootstrap"]["paired"] is True
    assert manifest["bootstrap"]["seed"] == protocol.BOOTSTRAP_SEED


# ======================================================== error analysis

def test_error_analysis_reports_both_error_directions(smoke_run):
    _manifest, run_dir = smoke_run
    metrics = json.loads(
        (run_dir / "logistic_regression_metrics.json").read_text(encoding="utf-8")
    )

    analysis = metrics["error_analysis"]

    assert "false_positives" in analysis and "false_negatives" in analysis
    assert analysis["confidently_wrong"]["definition"]
    assert analysis["uncertain_band"]["definition"]


def test_error_analysis_invents_no_demographic_subgroups():
    """The contract holds no protected attribute; none is fabricated."""
    rng = np.random.default_rng(0)
    frame = __import__("pandas").DataFrame(
        {name: rng.integers(1, 5, 50) for name in ("GenHlth", "BMI", "Age", "HighBP")}
    )
    analysis = benchmark.error_analysis(
        rng.integers(0, 2, 50), rng.random(50), 0.5, frame
    )

    assert "no demographic fairness analysis" in analysis["subgroups_note"].lower()


# ============================================================ fail closed

def test_a_drifted_split_aborts_the_run(monkeypatch, tmp_path):
    """The benchmark must refuse to score rows that are not the frozen ones."""
    from research.track_k import split as split_module

    monkeypatch.setattr(
        split_module, "verify_split", lambda *args, **kwargs: ["dataset_sha256: changed"]
    )

    with pytest.raises(split_module.SplitIntegrityError):
        benchmark.run(smoke=True, output_root=tmp_path)


# ================================================ artifact bookkeeping units

def test_a_run_directory_inside_the_project_is_recorded_relatively():
    described = artifacts.describe_root(artifacts.RESEARCH_ROOT / "run-1")

    assert described == "research_artifacts/track_k/run-1"
    assert ":" not in described, "a portable manifest records no drive letter"


def test_a_run_directory_outside_the_project_is_recorded_absolutely(tmp_path):
    """A CI temp directory is a legitimate place to run; it must not crash."""
    described = artifacts.describe_root(tmp_path / "elsewhere")

    assert described.endswith("elsewhere")


def test_artifacts_are_named_relative_to_their_run_directory(tmp_path):
    """So a run directory can be moved or archived and still verify."""
    run_dir = tmp_path / "run-2"
    (run_dir / "nested").mkdir(parents=True)
    shallow = run_dir / "mlp_checkpoint.pt"
    shallow.write_bytes(b"weights")
    nested = run_dir / "nested" / "curve.json"
    nested.write_text("{}", encoding="utf-8")

    names, hashes = artifacts.inventory(
        {"checkpoint": shallow, "learning_curve": nested}, run_dir
    )

    assert names == {"checkpoint": "mlp_checkpoint.pt", "learning_curve": "nested/curve.json"}
    assert set(hashes) == {"checkpoint", "learning_curve"}
    assert all(len(digest) == 64 for digest in hashes.values())


def test_an_artifact_outside_the_run_directory_falls_back_to_its_name(tmp_path):
    outside = tmp_path / "stray.json"
    outside.write_text("{}", encoding="utf-8")

    names, _hashes = artifacts.inventory({"metrics": outside}, tmp_path / "run-3")

    assert names == {"metrics": "stray.json"}


def test_the_source_fingerprint_covers_every_module_that_changes_a_result():
    fingerprint = artifacts.source_fingerprint()

    recorded = {entry["path"] for entry in fingerprint["files"]}
    for module in artifacts.SOURCE_MODULES:
        assert any(name.endswith(module) for name in recorded), f"{module} is not fingerprinted"


def test_a_research_manifest_is_typed_apart_from_a_production_one():
    """So a research run can never be mistaken for a training run."""
    from ml_core import provenance as production_provenance

    assert artifacts.RESEARCH_RUN != production_provenance.TRAINING_RUN


def test_a_manifest_of_the_wrong_type_is_rejected(tmp_path):
    problems = artifacts.verify_run_manifest(
        {"provenance_type": "training_run", "protocol_version": protocol.PROTOCOL_VERSION},
        tmp_path,
    )

    assert any("provenance_type" in problem for problem in problems)


# ================================================== the command-line entry

def test_the_cli_runs_a_smoke_benchmark_and_labels_its_output(tmp_path, capsys):
    exit_code = benchmark.main(["--smoke", "--output-root", str(tmp_path)])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "SMOKE RUN" in printed
    assert "baseline for promotion" in printed
    assert list(tmp_path.glob("smoke-*/run_manifest.json")), "no manifest was written"


def test_a_full_run_trains_on_the_searched_configuration(tmp_path, monkeypatch):
    """The non-smoke path must use the search result, not a fixed default.

    The search itself is budgeted in minutes, so it is replaced here by a known
    outcome. What is under test is the wiring: that a full run asks for a
    configuration and then trains the one it was given.
    """
    from research.track_k import baselines, challengers
    from research.track_k import split as split_module

    chosen = {"hidden_dims": [8], "batch_size": 512, "dropout": 0.0}
    recorded = baselines.SearchOutcome(
        family="mlp", trials=7, best_params=chosen, best_validation_score=0.5
    )
    monkeypatch.setattr(challengers, "search_mlp", lambda *a, **k: recorded)
    profile = dataclasses.replace(protocol.CPU_CONSTRAINED_PROFILE, final_max_epochs=1)

    splits = split_module.build_split(split_module.load_dataset())
    result = benchmark._run_deep(
        "mlp", splits, profile=profile, smoke=False, out_dir=tmp_path
    )

    assert result.record.config == chosen
    assert result.record.search["trials"] == 7
    assert result.record.training["smoke"] is False
    assert result.record.training["profile"] == profile.name
    assert result.record.resources["search_and_fit_seconds"] > 0


# ============================== recomputing a finished run without retraining

def test_recomputing_a_run_reproduces_it_exactly(smoke_run):
    """Same predictions, same code, same numbers - or the pipeline is not sound."""
    from research.track_k import recompute

    _manifest, run_dir = smoke_run

    record = recompute.recompute(run_dir, reason="reproducibility check", resamples=RESAMPLES)

    assert record["retrained"] is False
    assert record["differences_from_original"] == {}, "a recomputation drifted"


def test_a_recomputation_never_retrains(smoke_run):
    """The whole point: scoring is cheap, training is not."""
    import hashlib

    from research.track_k import recompute

    _manifest, run_dir = smoke_run
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.iterdir()
        if path.is_file()
    }

    recompute.recompute(run_dir, reason="checking nothing is rewritten", resamples=RESAMPLES)

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.iterdir()
        if path.is_file() and path.name in before
    }
    assert after == before, "a recomputation modified the run it read"


def test_a_recomputation_is_written_beside_the_original_not_over_it(smoke_run):
    from research.track_k import recompute

    manifest, run_dir = smoke_run
    original = (run_dir / "run_manifest.json").read_bytes()

    record = recompute.recompute(run_dir, reason="write placement", resamples=RESAMPLES)
    path = recompute.write(record, run_dir)

    assert path.name == recompute.RECOMPUTED_FILENAME
    assert (run_dir / "run_manifest.json").read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8"))["source_run_id"] == (
        manifest["track_k_run_id"]
    )


def test_a_recomputation_records_the_run_and_the_reason_it_was_made(smoke_run):
    from research.track_k import recompute

    manifest, run_dir = smoke_run

    record = recompute.recompute(run_dir, reason="because the metric was wrong", resamples=RESAMPLES)

    assert record["source_run_id"] == manifest["track_k_run_id"]
    assert record["reason"] == "because the metric was wrong"
    assert record["source"]["combined_sha256"], "the code version must be recorded"


def test_a_recomputation_refuses_a_tampered_prediction_file(smoke_run):
    from research.track_k import recompute

    _manifest, run_dir = smoke_run
    target = run_dir / "mlp_checkpoint.pt"
    original = target.read_bytes()

    try:
        target.write_bytes(original + b"tampered")
        with pytest.raises(recompute.RecomputeError):
            recompute.recompute(run_dir, reason="should not get this far", resamples=RESAMPLES)
    finally:
        target.write_bytes(original)


def test_a_recomputation_refuses_a_run_whose_split_has_moved(smoke_run, monkeypatch):
    """Recomputing over different rows would be a new experiment in disguise."""
    from research.track_k import recompute
    from research.track_k import split as split_module

    _manifest, run_dir = smoke_run
    real = split_module.fingerprint_split

    def drifted(*args, **kwargs):
        fingerprint = real(*args, **kwargs)
        return dataclasses.replace(fingerprint, combined_sha256="0" * 64)

    monkeypatch.setattr(recompute.split, "fingerprint_split", drifted)

    with pytest.raises(recompute.RecomputeError, match="combined_sha256"):
        recompute.recompute(run_dir, reason="should not get this far", resamples=RESAMPLES)


def test_a_recomputation_reports_when_nothing_changed(smoke_run):
    from research.track_k import recompute

    _manifest, run_dir = smoke_run

    text = recompute.summarise(
        recompute.recompute(run_dir, reason="no-op", resamples=RESAMPLES)
    )

    assert "No statistic changed" in text


def test_the_recompute_cli_writes_a_record(smoke_run, capsys):
    from research.track_k import recompute

    _manifest, run_dir = smoke_run

    exit_code = recompute.main([str(run_dir), "--reason", "cli smoke"])

    assert exit_code == 0
    assert "without retraining" in capsys.readouterr().out
    assert (run_dir / recompute.RECOMPUTED_FILENAME).is_file()


def test_a_missing_run_directory_fails_clearly(tmp_path):
    from research.track_k import recompute

    with pytest.raises(recompute.RecomputeError, match="no run manifest"):
        recompute.recompute(tmp_path / "not-a-run", reason="nothing here", resamples=RESAMPLES)


def test_a_recomputation_reports_each_statistic_that_moved():
    """The path that actually mattered: naming what changed, and by how much."""
    from research.track_k import recompute

    record = {
        "source_run_id": "full-abc123",
        "reason": "average precision mishandled tied scores",
        "differences_from_original": {
            "logistic_regression": {
                "bootstrap.pr_auc.point": {"original": 0.79134, "recomputed": 0.78442}
            }
        },
    }

    text = recompute.summarise(record)

    assert "logistic_regression" in text
    assert "bootstrap.pr_auc.point" in text
    assert "-0.00692" in text, "the size of the correction must be stated"


def test_a_recomputation_reports_non_numeric_changes_without_a_delta():
    from research.track_k import recompute

    record = {
        "source_run_id": "full-abc123",
        "reason": "checking the formatter",
        "differences_from_original": {
            "mlp": {"calibration": {"original": "isotonic", "recomputed": "none"}}
        },
    }

    text = recompute.summarise(record)

    assert "n/a" in text, "a non-numeric change must not be given a fabricated delta"


def test_a_run_missing_its_saved_predictions_is_refused(smoke_run, tmp_path):
    """Recomputation depends on the predictions; their absence is not recoverable."""
    import shutil

    from research.track_k import recompute

    _manifest, run_dir = smoke_run
    copied = tmp_path / "copy"
    shutil.copytree(run_dir, copied)
    (copied / "mlp_test_proba.npy").unlink()

    with pytest.raises(recompute.RecomputeError, match="no saved test predictions"):
        recompute.load_predictions(copied, recompute.load_run(copied))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (1.0, 1.0, True),
        (1.0, 1.0 + 1e-15, True),
        (1.0, 1.1, False),
        (None, None, True),
        (None, 1.0, False),
        ("isotonic", "isotonic", True),
        ("isotonic", "sigmoid", False),
    ],
)
def test_comparison_of_recorded_values_handles_every_kind(left, right, expected):
    """Manifest values are numbers, strings and nulls; all three must compare."""
    from research.track_k import recompute

    assert recompute._close(left, right) is expected
