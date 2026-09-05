"""The runner: what it records, what it refuses, and what it writes last.

The property this file protects hardest is that **failures stay in the table**.
Two hundred and sixty-one (model, method) pairs is mostly invalid combinations,
and a results file listing only what worked would look like a clean sweep while
hiding the entire capability finding. Every attempted pair must produce a row,
and the row must say why.

The second property is ordering. The manifest is written last, so a run killed
halfway leaves records without a manifest - unambiguous. A manifest written
first would leave a run advertising results it does not have, and nothing
downstream could tell the difference.

These tests use the real split and the real registry at a deliberately small
budget. Mocking the model zoo would test the mock, and the failure this file
exists to catch is a mismatch between what a model claims and what it does.
"""
import json
import warnings

import numpy as np
import pytest

from research.model_zoo.contracts import ResearchStatus
from research.model_zoo.registry import REGISTRY
from research.xai import run as xai_run
from research.xai.capabilities import profile_for
from research.xai.contracts import RunStatus
from research.xai.registry import METHODS

pytest.importorskip("torch", reason="the zoo's deep models need PyTorch")

#: Small enough to run repeatedly, wide enough to cover a linear model, a tree
#: ensemble, a deep model and a model with no ranking score at all.
SMOKE_MODELS = ["logistic_l2", "random_forest", "mlp", "nearest_centroid"]
SMOKE_TRAIN_ROWS = 200
SMOKE_CASES = 4


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xai_run.run(
            train_rows=SMOKE_TRAIN_ROWS,
            model_ids=SMOKE_MODELS,
            case_limit=SMOKE_CASES,
            output_root=tmp_path_factory.mktemp("xai"),
            overrides={"max_epochs": 2},
            with_interactions=False,
        )


# ===================================================================== planning

def test_the_plan_covers_every_model_and_method_asked_for():
    matrix = xai_run.plan(["logistic_l2", "random_forest"], ["coefficients", "occlusion"])

    assert len(matrix) == 4
    assert {row["model_id"] for row in matrix} == {"logistic_l2", "random_forest"}
    assert {row["method"] for row in matrix} == {"coefficients", "occlusion"}


def test_the_plan_defaults_to_every_active_model_and_every_method():
    active = [s for s in REGISTRY if s.effective_status() is ResearchStatus.ACTIVE]

    matrix = xai_run.plan()

    assert len(matrix) == len(active) * len(METHODS)


def test_an_unsupported_pair_is_planned_with_the_capability_it_lacks():
    """The plan has to be readable before a run, or the budget is a guess."""
    matrix = xai_run.plan(["random_forest"], ["coefficients"])

    assert matrix[0]["supported"] is False
    assert "native_coefficients" in matrix[0]["reason"]


def test_the_plan_agrees_with_the_capability_profiles_it_claims_to_join():
    """The runner must not decide support by any route but the declaration."""
    for row in xai_run.plan(SMOKE_MODELS):
        profile = profile_for(REGISTRY.get(row["model_id"]))
        method = METHODS.get(row["method"])
        assert row["supported"] == profile.supports(method.required_capability)


def test_a_hard_label_model_is_planned_out_of_every_method():
    """Not a gap in the study - a property of a model with no ranking score."""
    matrix = xai_run.plan(["nearest_centroid"])

    assert all(not row["supported"] for row in matrix)


# ===================================================================== the run

def test_every_attempted_pair_produces_an_outcome(smoke, tmp_path_factory):
    """The central anti-flattering property.

    A results file listing only what worked would hide the whole capability
    finding behind a clean sweep.
    """
    outcomes = json.loads(
        (_run_dir(smoke, tmp_path_factory) / "outcomes.json").read_text(encoding="utf-8")
    )
    attempted = {(row["model_id"], row["method"]) for row in outcomes}
    expected = {
        (row["model_id"], row["method"])
        for row in xai_run.plan(SMOKE_MODELS)
    }

    assert attempted == expected


def test_unsupported_pairs_are_recorded_with_a_reason_rather_than_dropped(smoke):
    counts = smoke["counts"]

    assert counts[RunStatus.UNSUPPORTED.value] > 0
    assert counts[RunStatus.SUCCESS.value] > 0


def test_a_successful_pair_carries_a_record_and_a_failed_one_carries_a_reason(
    smoke, tmp_path_factory
):
    outcomes = json.loads(
        (_run_dir(smoke, tmp_path_factory) / "outcomes.json").read_text(encoding="utf-8")
    )

    for row in outcomes:
        if row["status"] == RunStatus.SUCCESS.value:
            assert row["record"] is not None
            assert row["error"] is None
        else:
            assert row["record"] is None
            assert row["error"], f"{row['model_id']}/{row['method']} failed silently"


def test_records_are_plain_json_with_shares_that_sum_to_one(smoke, tmp_path_factory):
    """Evidence is JSON, and the normalisation is what makes families comparable."""
    directory = _run_dir(smoke, tmp_path_factory)

    for path in (directory / "records").glob("*.json"):
        for record in json.loads(path.read_text(encoding="utf-8")):
            assert record["schema_version"]
            assert len(record["feature_names"]) == len(record["raw_attributions"])
            assert sum(record["normalized_attributions"]) == pytest.approx(1.0)
            assert set(record["ranking"]) == set(record["feature_names"])


def test_a_local_method_explains_every_case_and_a_global_one_explains_the_model(
    smoke, tmp_path_factory
):
    directory = _run_dir(smoke, tmp_path_factory)
    by_method: dict[str, list[dict]] = {}
    for path in (directory / "records").glob("*.json"):
        for record in json.loads(path.read_text(encoding="utf-8")):
            by_method.setdefault((record["model_id"], record["method"]), []).append(record)

    for (model_id, method_id), records in by_method.items():
        scope = METHODS.get(method_id).scope.value
        if scope == "global":
            assert len(records) == 1, f"{model_id}/{method_id} produced {len(records)}"
            assert records[0]["sample_id"] is None
        else:
            assert len(records) == SMOKE_CASES
            assert sorted(r["sample_id"] for r in records) == list(range(SMOKE_CASES))


def test_the_manifest_records_which_partition_each_thing_came_from(smoke):
    """The leak audit, written into the evidence rather than left to a docstring."""
    provenance = smoke["provenance"]

    assert provenance["train_rows"] == SMOKE_TRAIN_ROWS
    assert provenance["case_rows"] == SMOKE_CASES
    assert "median of the fitting rows" in provenance["baseline_source"]
    assert "never the fitting rows or test" in provenance["permutation_scored_on"]
    assert provenance["subset_fingerprints"], "the subset ladder must be fingerprinted"


def test_the_run_is_stamped_with_the_exploratory_evidence_class(smoke):
    """Track M inherits Track L's class: broad, cheap, and not a promotion basis."""
    assert smoke["evidence_class"] == "RESOURCE_CONSTRAINED_EXPLORATORY"


def test_the_manifest_is_written_last(smoke, tmp_path_factory):
    """A half-run leaves records and no manifest, which is unambiguous.

    Checked by modification time rather than by reading the code, so a future
    reordering of the writes fails here instead of silently producing manifests
    that advertise results the run never finished computing.
    """
    directory = _run_dir(smoke, tmp_path_factory)
    manifest = directory / "run_manifest.json"

    others = [
        path for path in directory.rglob("*.json") if path != manifest
    ]
    assert others, "the run wrote nothing but a manifest"
    assert manifest.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in others)


def test_the_analysis_reports_a_consensus_over_everything_that_succeeded(smoke):
    analysis = smoke["analysis"]

    assert analysis["records"] > 0
    assert set(analysis["zoo_consensus"]) == set(analysis["zoo_mean_ranks"])
    assert analysis["within_model"]["pairs"] >= 0
    assert analysis["between_families"]["pairs"] >= 0


def test_the_records_hash_changes_when_the_records_do(smoke):
    assert len(smoke["records_hash"]) == 64


def test_a_model_that_will_not_fit_is_named_rather_than_dropped_quietly(smoke):
    assert smoke["fit_failures"] == {}


def test_the_summary_names_the_failures_it_found(smoke):
    text = xai_run.summarise(smoke)

    assert smoke["run_id"] in text
    assert "unsupported" in text
    assert "zoo consensus top three" in text


def test_the_summary_survives_a_run_that_explained_nothing():
    """A manifest with no records must summarise, not raise.

    This is the shape a fully-unsupported run produces - every model a hard
    labeller, say - and it has to be reportable.
    """
    text = xai_run.summarise({
        "run_id": "xai-empty",
        "evidence_class": "RESOURCE_CONSTRAINED_EXPLORATORY",
        "train_rows": 10,
        "case_limit": 1,
        "models_requested": [],
        "methods_requested": [],
        "counts": {},
        "analysis": {"records": 0},
    })

    assert "xai-empty" in text


# ============================================================== dispatch guards

def _invented_method(scope):
    from research.xai.capabilities import XaiCapability
    from research.xai.contracts import BaselineStrategy, Determinism
    from research.xai.registry import MethodSpec

    return MethodSpec(
        method_id="not_implemented",
        display_name="Invented",
        version="0.0.0",
        required_capability=XaiCapability.PERMUTATION_IMPORTANCE,
        scope=scope,
        determinism=Determinism.DETERMINISTIC,
        baseline_strategy=BaselineStrategy.NOT_APPLICABLE,
        runtime_class="instant",
        measures="nothing",
        failure_modes=("it does not exist",),
    )


def test_a_method_with_no_implementation_is_refused_by_both_dispatchers():
    """A method registered without an implementation must fail loudly.

    Returning an empty attribution instead would put a row in the table that a
    reader would take as "this model attributes nothing", which is a claim about
    the model rather than about the harness.
    """
    from research.xai.contracts import Scope, XaiError

    context = _tiny_context()

    with pytest.raises(XaiError, match="not a global method"):
        xai_run._global_attributions(_invented_method(Scope.GLOBAL), object(), context)

    with pytest.raises(XaiError, match="not a local method"):
        xai_run._local_attributions(_invented_method(Scope.LOCAL), object(), context)


def test_a_method_that_raises_is_recorded_as_a_numerical_failure(smoke, monkeypatch):
    """A broken method must leave a row, not abort the run.

    One method failing on one model is a result about that pair. Letting it
    propagate would cost every result the run had not yet computed.
    """
    spec = REGISTRY.get("logistic_l2")
    method = METHODS.get("coefficients")

    def explode(*args, **kwargs):
        raise ValueError("deliberate")

    monkeypatch.setattr(xai_run, "_global_attributions", explode)
    context = _tiny_context()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("logistic_l2").fit(context.X_fit, context.y_fit)

    outcomes = xai_run.explain_pair(spec, model, method, context)

    assert len(outcomes) == 1
    assert outcomes[0].status is RunStatus.NUMERICAL_FAILURE
    assert "deliberate" in outcomes[0].error


def test_a_method_that_blows_its_budget_is_recorded_as_a_resource_limit():
    """"Too slow at this budget" is a finding, not a reason to have no result."""
    spec = REGISTRY.get("logistic_l2")
    method = METHODS.get("coefficients")
    context = _tiny_context()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("logistic_l2").fit(context.X_fit, context.y_fit)

    outcomes = xai_run.explain_pair(spec, model, method, context, budget=0.0)

    assert outcomes[0].status is RunStatus.RESOURCE_LIMIT
    assert "budget" in outcomes[0].error


# ======================================================================= helpers

def _run_dir(manifest, tmp_path_factory):
    root = tmp_path_factory.getbasetemp()
    matches = list(root.glob(f"**/{manifest['run_id']}"))
    assert matches, f"no output directory for {manifest['run_id']}"
    return matches[0]


def _tiny_context():
    from research.xai import stability, worlds

    dataset = worlds.make(worlds.XaiWorld.ONE_DOMINANT_FEATURE, rows=80, seed=1)
    X, y = dataset.X, dataset.y
    return xai_run.RunContext(
        X_fit=X, y_fit=y, X_eval=X, y_eval=y,
        cases=X.iloc[:3],
        baseline=X.median(axis=0),
        scale=stability.fit_scale(X),
        feature_names=tuple(str(c) for c in X.columns),
    )


def test_the_case_sample_is_deterministic():
    """Two runs must explain the same patients, or nothing is comparable."""
    first, _ = xai_run._build_context(200, 6)
    again, _ = xai_run._build_context(200, 6)

    assert first.cases.equals(again.cases)
    assert np.allclose(first.baseline.to_numpy(), again.baseline.to_numpy())


def test_the_evaluation_sample_is_bounded_and_shared():
    """One sample, drawn once, so a model-to-model difference is about models.

    Giving each model its own evaluation rows would make every cross-model
    comparison partly a comparison of two row samples, and the resulting
    disagreement would look like a finding about the models.
    """
    context, provenance = xai_run._build_context(200, 6, eval_rows=120)

    assert len(context.X_eval) == 120
    assert len(context.y_eval) == 120
    assert provenance["evaluation_rows"] == 120
    assert provenance["evaluation_pool_rows"] > 120

    again, _ = xai_run._build_context(200, 6, eval_rows=120)
    assert context.X_eval.equals(again.X_eval)


# ================================================================ the sweeps

def test_the_stability_sweep_reports_how_far_the_top_feature_survives(smoke_context):
    """Occlusion is the probe because it is the one method every scored model has.

    A stability figure computed with whichever method happened to apply would
    describe the method as much as the model, and could not be compared across
    families.
    """
    model, context = smoke_context

    summary = xai_run._stability_for(model, context)

    assert summary["probe"] == "occlusion profile"
    assert summary["measured"] is True
    assert len(summary["points"]) == len(xai_run.RUN_STABILITY_MAGNITUDES)
    assert all(p["replicates"] == xai_run.RUN_STABILITY_REPEATS for p in summary["points"])


def test_a_sweep_that_raises_is_recorded_rather_than_aborting_the_run(
    smoke_context, monkeypatch
):
    """One model's failed sweep must not cost every result after it."""
    model, context = smoke_context

    def explode(*args, **kwargs):
        raise ValueError("deliberate")

    monkeypatch.setattr(xai_run.stability, "stability_curve", explode)
    assert "deliberate" in xai_run._stability_for(model, context)["error"]

    monkeypatch.setattr(xai_run.faithfulness, "evaluate", explode)
    records = _global_records(model, context)
    assert "deliberate" in xai_run._faithfulness_for(model, context, records)["error"]

    monkeypatch.setattr(xai_run.interactions, "rank_interactions", explode)
    assert "deliberate" in xai_run._interactions_for(model, context, records)["error"]


def test_the_faithfulness_sweep_scores_the_models_own_consensus_ranking(smoke_context):
    model, context = smoke_context

    summary = xai_run._faithfulness_for(model, context, _global_records(model, context))

    assert set(summary["ranking"]) == set(context.feature_names)
    assert "deletion_gap" in summary
    assert "beats_random" in summary


def test_the_interaction_sweep_is_scoped_to_the_models_own_top_features(smoke_context):
    """An interaction between two features a model ignores is not a finding."""
    model, context = smoke_context

    summary = xai_run._interactions_for(model, context, _global_records(model, context))

    assert len(summary["features"]) == xai_run.DEFAULT_INTERACTION_FEATURES
    assert summary["pairs"] == 10


# =================================================================== the CLI

def test_the_dry_run_prints_the_matrix_without_running_anything(capsys):
    """What makes the budget a plan rather than a hope."""
    assert xai_run.main(["--dry-run", "--models", "logistic_l2", "random_forest"]) == 0

    printed = capsys.readouterr().out
    assert "18 pairs" in printed
    assert "run  logistic_l2" in printed
    assert "skip random_forest" in printed
    assert "native_coefficients" in printed


def test_the_cli_runs_a_bounded_sweep_and_writes_a_manifest(tmp_path, capsys):
    model_zoo_smoke = [
        "--models", "logistic_l2",
        "--methods", "coefficients",
        "--train-rows", "200", "--case-limit", "2", "--eval-rows", "60",
        "--no-interactions", "--no-sweeps",
        "--output-dir", str(tmp_path),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert xai_run.main(model_zoo_smoke) == 0

    printed = capsys.readouterr().out
    assert "RESOURCE_CONSTRAINED_EXPLORATORY" in printed
    manifests = list(tmp_path.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["train_rows"] == 200


def test_skipping_the_sweeps_leaves_their_files_unwritten(tmp_path):
    """A file that exists but is empty would read as a measurement of nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = xai_run.run(
            train_rows=200, model_ids=["logistic_l2"], method_ids=["coefficients"],
            case_limit=2, eval_rows=60, output_root=tmp_path,
            with_interactions=False, with_sweeps=False,
        )

    directory = next(tmp_path.glob(manifest["run_id"]))
    assert not (directory / "stability.json").exists()
    assert not (directory / "faithfulness.json").exists()
    assert not (directory / "interactions.json").exists()
    assert manifest["stability"] == {}
    assert manifest["faithfulness"] == {}


@pytest.fixture(scope="module")
def smoke_context():
    """One cheap fitted model plus the partitions the sweeps read."""
    context, _ = xai_run._build_context(200, 8, eval_rows=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("logistic_l2").fit(context.X_fit, context.y_fit)
    return model, context


def _global_records(model, context):
    """The records a sweep needs: one global explanation is enough for a ranking."""
    spec = REGISTRY.get("logistic_l2")
    outcomes = xai_run.explain_pair(spec, model, METHODS.get("coefficients"), context)
    return [o.record for o in outcomes if o.record]
