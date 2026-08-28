"""Training profiles and the sample-efficiency study.

A profile fixes how much training data and search a run may use, and nothing
else. The tests below pin that: two arms of this study differ in training budget
alone, so any difference between them is attributable to the budget rather than
to a quietly different split, metric or promotion rule.
"""
import pytest

from research.track_k import protocol, sample_efficiency

pytest.importorskip("torch", reason="PyTorch is a Track K research dependency")


# ========================================================= profile contract

def test_both_arms_of_the_study_are_declared():
    assert set(protocol.TRAINING_PROFILES) == {"full_reference", "cpu_constrained"}


@pytest.mark.parametrize("name", ["full_reference", "cpu_constrained"])
def test_a_profile_knows_its_own_name(name):
    assert protocol.training_profile(name).name == name


@pytest.mark.parametrize("name", ["full_reference", "cpu_constrained"])
def test_every_profile_budgets_every_family(name):
    """A family with no declared budget would fail only at runtime, mid-run."""
    profile = protocol.training_profile(name)

    assert set(profile.trials) == set(protocol.MODEL_FAMILIES)
    for family, trials in profile.trials.items():
        assert trials > 0, f"{name}: {family} has no search budget"


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="unknown training profile"):
        protocol.training_profile("wishful")


def test_the_reference_arm_uses_the_whole_training_partition():
    assert protocol.FULL_REFERENCE_PROFILE.training_rows is None


def test_the_constrained_arm_trains_on_a_size_the_subset_ladder_provides():
    """Otherwise the profile would name a subset that cannot be built."""
    rows = protocol.CPU_CONSTRAINED_PROFILE.training_rows

    assert rows in protocol.SAMPLE_EFFICIENCY_SIZES


def test_the_constrained_arm_is_actually_a_constraint():
    """Every budget must be no larger than the reference arm's."""
    reference = protocol.FULL_REFERENCE_PROFILE
    constrained = protocol.CPU_CONSTRAINED_PROFILE

    for family in protocol.MODEL_FAMILIES:
        assert constrained.trials[family] <= reference.trials[family], family
    assert constrained.search_max_epochs <= reference.search_max_epochs
    assert constrained.final_max_epochs <= reference.final_max_epochs


def test_every_profile_explains_itself():
    for name, profile in protocol.TRAINING_PROFILES.items():
        assert len(profile.rationale) > 40, f"{name} has no stated rationale"


def test_a_profile_changes_no_frozen_evaluation_decision():
    """The whole point of profiles: only the training budget varies.

    If a profile could move the split, the metric or the promotion bar, two arms
    would not be comparable and the protocol version would have to change.
    """
    fields = set(protocol.TrainingProfile.__dataclass_fields__)

    assert fields == {
        "name",
        "training_rows",
        "trials",
        "search_max_epochs",
        "final_max_epochs",
        "rationale",
    }


def test_the_subset_seed_is_distinct_from_the_split_seed():
    """So changing which rows a subset holds cannot look like changing the split."""
    assert protocol.SUBSET_SEED != protocol.SPLIT_SEED


def test_the_sample_efficiency_ladder_is_increasing_and_distinct():
    sizes = protocol.SAMPLE_EFFICIENCY_SIZES

    assert list(sizes) == sorted(set(sizes))
    assert len(sizes) >= 3, "two points do not make a curve"


# ==================================================== sample-efficiency study

def test_every_family_has_a_representative_configuration():
    """A missing config would silently drop a family from the curve."""
    assert set(sample_efficiency.REPRESENTATIVE_CONFIGS) == set(protocol.MODEL_FAMILIES)


def test_the_study_reports_one_point_per_family_and_size(tmp_path):
    families = ("logistic_regression", "mlp")
    sizes = (200, 400)

    manifest = sample_efficiency.run(
        sizes=sizes, families=families, max_epochs=2, output_root=tmp_path
    )

    seen = {(p["family"], p["training_rows"]) for p in manifest["points"]}
    assert seen == {(f, s) for f in families for s in sizes}


def test_the_study_scores_on_the_full_test_partition(tmp_path):
    """Training is cheap to shrink; evaluation is not, and must not be shrunk."""
    manifest = sample_efficiency.run(
        sizes=(200,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    assert manifest["evaluated_on"] == "test"
    assert manifest["evaluation_rows"] == manifest["split_sizes"]["test"]
    assert manifest["evaluation_rows"] > 10_000


def test_the_study_records_cost_alongside_quality(tmp_path):
    manifest = sample_efficiency.run(
        sizes=(200,), families=("mlp",), max_epochs=2, output_root=tmp_path
    )

    point = manifest["points"][0]
    assert point["train_seconds"] > 0
    assert point["parameter_count"] > 0
    assert point["epochs_run"] >= 1
    assert 0.0 <= point["roc_auc"] <= 1.0


def test_a_classical_family_reports_no_parameter_count(tmp_path):
    """Rather than a fabricated one: a fitted pipeline has no comparable count."""
    manifest = sample_efficiency.run(
        sizes=(200,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    assert manifest["points"][0]["parameter_count"] is None


def test_the_study_declines_to_calibrate_and_says_so(tmp_path):
    """A per-size calibrator would confound learning with calibrator data."""
    manifest = sample_efficiency.run(
        sizes=(200,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    assert manifest["calibration"].startswith("none")


def test_the_study_records_the_subset_it_trained_on(tmp_path):
    manifest = sample_efficiency.run(
        sizes=(200, 400), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    subset = manifest["training_subset"]
    assert subset["drawn_from"] == "train"
    assert sorted(int(k) for k in subset["subsets"]) == [200, 400]


def test_the_study_touches_no_production_artifact(tmp_path):
    manifest = sample_efficiency.run(
        sizes=(200,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    assert manifest["production_artifacts_touched"] is False


def test_the_study_is_written_to_disk(tmp_path):
    manifest = sample_efficiency.run(
        sizes=(200,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    written = tmp_path / manifest["track_k_run_id"] / "sample_efficiency.json"
    assert written.is_file()


def test_the_study_is_deterministic(tmp_path):
    """Same subset, same seed, same configuration - same number."""
    first = sample_efficiency.run(
        sizes=(400,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )
    second = sample_efficiency.run(
        sizes=(400,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    assert first["points"][0]["roc_auc"] == second["points"][0]["roc_auc"]


def test_the_summary_names_every_family_and_size(tmp_path):
    manifest = sample_efficiency.run(
        sizes=(200, 400), families=("logistic_regression", "xgboost"),
        max_epochs=2, output_root=tmp_path,
    )

    text = sample_efficiency.summarise(manifest)

    assert "logistic_regression" in text and "xgboost" in text
    assert "roc_auc" in text and "train_seconds" in text


def test_gain_per_doubling_is_reported_per_family(tmp_path):
    manifest = sample_efficiency.run(
        sizes=(200, 400), families=("logistic_regression", "xgboost"),
        max_epochs=2, output_root=tmp_path,
    )

    gains = sample_efficiency.gain_per_doubling(manifest)

    assert set(gains) == {"logistic_regression", "xgboost"}
    assert all(isinstance(value, float) for value in gains.values())


def test_gain_per_doubling_needs_two_points_to_report_anything(tmp_path):
    """One measurement is not a trend, and must not be presented as one."""
    manifest = sample_efficiency.run(
        sizes=(200,), families=("logistic_regression",), max_epochs=2, output_root=tmp_path
    )

    assert sample_efficiency.gain_per_doubling(manifest) == {}


def test_the_cli_runs_the_study(tmp_path, capsys):
    exit_code = sample_efficiency.main(
        ["--sizes", "200", "--max-epochs", "2", "--output-root", str(tmp_path)]
    )

    assert exit_code == 0
    assert "sample efficiency" in capsys.readouterr().out


# ================================= the constrained arm, end to end (smoke)

@pytest.fixture(scope="module")
def constrained_smoke(tmp_path_factory):
    """A smoke run under the constrained profile: tiny models, narrowed train."""
    from research.track_k import benchmark

    root = tmp_path_factory.mktemp("constrained")
    manifest = benchmark.run(smoke=True, profile="cpu_constrained", output_root=root)
    return manifest, root / manifest["track_k_run_id"]


def test_the_constrained_arm_trains_on_the_declared_subset(constrained_smoke):
    manifest, _run_dir = constrained_smoke

    profile = manifest["training_profile"]
    assert profile["name"] == "cpu_constrained"
    assert profile["training_rows"] == protocol.CPU_CONSTRAINED_PROFILE.training_rows


def test_the_constrained_arm_leaves_validation_and_test_at_full_size(constrained_smoke):
    """Training is what is constrained. Evaluation is cheap and stays whole."""
    manifest, _run_dir = constrained_smoke

    assert manifest["split_sizes"]["val"] > 10_000
    assert manifest["split_sizes"]["test"] > 10_000


def test_the_constrained_arm_records_the_subset_it_used(constrained_smoke):
    manifest, run_dir = constrained_smoke

    subset = manifest["training_subset"]
    assert subset["drawn_from"] == "train"
    assert subset["nested"] is True
    assert (run_dir / "subset_manifest.json").is_file()


def test_every_family_in_the_constrained_arm_saw_the_same_rows(constrained_smoke):
    """The fairness rule: one training budget, applied to everyone."""
    manifest, _run_dir = constrained_smoke

    rows = {record["training"]["training_rows"] for record in manifest["models"].values()}
    assert rows == {protocol.CPU_CONSTRAINED_PROFILE.training_rows}


def test_every_family_records_what_its_training_cost(constrained_smoke):
    manifest, _run_dir = constrained_smoke

    for family, record in manifest["models"].items():
        resources = record["resources"]
        assert resources["search_and_fit_seconds"] > 0, family
        assert resources["device"] == "cpu"
        assert resources["training_rows"] == protocol.CPU_CONSTRAINED_PROFILE.training_rows


def test_the_third_challenger_is_benchmarked_alongside_the_others(constrained_smoke):
    manifest, _run_dir = constrained_smoke

    assert "tabular_resnet" in manifest["models"]
    assert manifest["models"]["tabular_resnet"]["parameter_count"] > 0


def test_the_summary_states_the_profile_and_the_training_size(constrained_smoke):
    from research.track_k import benchmark

    manifest, run_dir = constrained_smoke

    text = benchmark.summarise(manifest, run_dir)

    assert "cpu_constrained" in text
    assert "training rows" in text


def test_an_unknown_profile_is_refused_before_anything_is_trained(tmp_path):
    from research.track_k import benchmark

    with pytest.raises(ValueError, match="unknown training profile"):
        benchmark.run(smoke=True, profile="wishful", output_root=tmp_path)

    assert not list(tmp_path.iterdir()), "a rejected run left artifacts behind"
