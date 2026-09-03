"""The benchmark harness: leakage, the negative control, and failure handling.

Thirty models sharing one harness means a single harness bug would corrupt
thirty results at once, and the corruption would look like a finding rather
than a fault. These tests attack the harness itself.

The negative control is the most important test in the Track L suite. On labels
that are independent of the features, no model can beat chance on held-out
rows. If one appears to, the harness is scoring a model on rows it was fitted
on, and every number the zoo produces is worthless. That failure mode is
invisible in the results table - the metrics simply look good - so it has to be
caught here.
"""
import warnings

import numpy as np
import pytest

from research.model_zoo import serialization, synthetic
from research.model_zoo.contracts import Framework, ProbabilityBehavior, RunOutcome
from research.model_zoo.registry import REGISTRY

pytest.importorskip("torch", reason="the zoo's deep models need PyTorch")

#: A representative slice: one model per family, cheap enough to run repeatedly.
REPRESENTATIVE = [
    "logistic_l2",
    "gaussian_nb",
    "knn",
    "linear_svm",
    "decision_tree",
    "random_forest",
    "xgboost",
    "mlp",
]


def _build(model_id: str, *, max_epochs: int = 3, **overrides):
    """Build a model, sending the epoch budget only to the ones that have epochs.

    Passing ``max_epochs`` to a decision tree is a TypeError, so the deep-only
    override is resolved here rather than at every call site.
    """
    spec = REGISTRY.get(model_id)
    if spec.framework is Framework.TORCH:
        overrides["max_epochs"] = max_epochs
    return REGISTRY.build(model_id, **overrides)


def _auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


# ==================================================== the negative control

@pytest.mark.parametrize("model_id", REPRESENTATIVE)
def test_no_model_beats_chance_on_pure_noise(model_id):
    """The leakage canary.

    Labels here are independent of every feature, so held-out ROC-AUC must sit
    near 0.5. A model scoring well above it would mean the harness let it see
    the rows it is being judged on - and that would silently inflate every
    result in the zoo, not just this one.

    The band is deliberately generous. With 240 held-out rows the sampling
    noise on an AUC of 0.5 is large, and a tight bound would make this test
    flaky rather than informative. Leakage does not produce 0.62; it produces
    0.95.
    """
    dataset = synthetic.make(synthetic.SyntheticProblem.NOISE_ONLY, rows=600, seed=11)
    X_train, y_train, X_test, y_test = synthetic.split(dataset, seed=11)

    spec = REGISTRY.get(model_id)
    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        pytest.skip(f"{model_id} has no ranking score")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(model_id).fit(X_train, y_train)
        auc = _auc(y_test, model.decision_scores(X_test))

    assert 0.30 < auc < 0.70, (
        f"{model_id} scored {auc:.4f} on labels independent of the features; "
        "the harness is leaking the held-out rows"
    )


def test_the_noise_dataset_really_is_unlearnable():
    """Guards the guard: a mislabelled control would silently pass everything."""
    dataset = synthetic.make(synthetic.SyntheticProblem.NOISE_ONLY, rows=800, seed=5)

    assert dataset.learnable_floor is None
    for column in dataset.X.columns:
        correlation = abs(np.corrcoef(dataset.X[column], dataset.y)[0, 1])
        assert correlation < 0.15, f"{column} correlates with the noise labels"


@pytest.mark.parametrize("model_id", REPRESENTATIVE)
def test_every_model_learns_a_trivially_learnable_problem(model_id):
    """The opposite failure: a harness that trains nothing would also pass above."""
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=600, seed=7)
    X_train, y_train, X_test, y_test = synthetic.split(dataset, seed=7)

    spec = REGISTRY.get(model_id)
    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        pytest.skip(f"{model_id} has no ranking score")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(model_id, max_epochs=15).fit(X_train, y_train)
        auc = _auc(y_test, model.decision_scores(X_test))

    assert auc > 0.75, f"{model_id} scored {auc:.4f} on a linearly separable problem"


def test_a_hard_label_model_still_learns_the_separable_problem():
    """Nearest centroid has no scores, so it is checked on accuracy instead."""
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=600, seed=7)
    X_train, y_train, X_test, y_test = synthetic.split(dataset, seed=7)

    model = _build("nearest_centroid").fit(X_train, y_train)
    accuracy = float((model.predict(X_test) == y_test.to_numpy()).mean())

    assert accuracy > 0.70


def test_the_imbalanced_problem_is_actually_imbalanced():
    dataset = synthetic.make(synthetic.SyntheticProblem.CLASS_IMBALANCE, rows=1000, seed=2)

    rate = float(dataset.y.mean())
    assert 0.05 < rate < 0.20, f"positive rate {rate:.3f} is not imbalanced"


def test_xor_with_distractors_defeats_a_linear_model_and_a_single_greedy_tree():
    """The fixture has a sharper shape than "linear fails, trees win".

    XOR over two binaries, with eight irrelevant features alongside, is the
    textbook failure case for greedy recursive partitioning. Neither XOR column
    has any *marginal* information gain - each is 50/50 against the label on its
    own - so CART has no reason to split on either first, and with distractors
    available it splits on noise instead. Measured here: an oracle that computes
    the parity directly scores 0.934, a tree given only the two relevant columns
    scores 0.935, and the zoo's tree on all ten scores 0.63.

    A random forest recovers it, because feature subsampling forces some trees
    to consider the XOR pair with the distractors withheld.

    This test originally asserted that a tree would clear 0.85, which is folklore
    that only holds when the tree is handed the right features. The corrected
    version is a stronger statement about the fixture and about the algorithms.
    """
    dataset = synthetic.make(synthetic.SyntheticProblem.NONLINEAR_XOR, rows=800, seed=4)
    X_train, y_train, X_test, y_test = synthetic.split(dataset, seed=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linear = _auc(y_test, _build("logistic_l2").fit(X_train, y_train).decision_scores(X_test))
        tree = _auc(y_test, _build("decision_tree").fit(X_train, y_train).decision_scores(X_test))
        forest = _auc(
            y_test, _build("random_forest").fit(X_train, y_train).decision_scores(X_test)
        )

    assert linear < 0.65, f"a linear model scored {linear:.4f} on XOR"
    assert tree < 0.80, (
        f"a single greedy tree scored {tree:.4f}; if it now solves XOR with "
        "distractors present, the fixture no longer has distractors"
    )
    assert forest > 0.85, (
        f"the forest scored {forest:.4f}; feature subsampling should recover XOR"
    )


def test_the_xor_fixture_is_solvable_by_something():
    """Guards the guard above: an unsolvable fixture would satisfy it trivially."""
    dataset = synthetic.make(synthetic.SyntheticProblem.NONLINEAR_XOR, rows=800, seed=4)
    _Xtr, _ytr, X_test, y_test = synthetic.split(dataset, seed=4)

    oracle = (X_test["HighBP"].astype(int) ^ X_test["HighChol"].astype(int)).to_numpy()

    assert _auc(y_test, oracle) > 0.90, "the XOR rule does not describe the labels"


# ========================================================== leakage tests

def test_the_scaler_is_fitted_on_training_rows_only():
    """A scaler fitted on more than train leaks distribution into the metrics."""
    from sklearn.preprocessing import StandardScaler

    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=600, seed=9)
    X_train, y_train, _X_test, _y = synthetic.split(dataset, seed=9)

    model = _build("logistic_l2").fit(X_train, y_train)
    fitted_scaler = model.estimator.named_steps["prepare"]

    train_only = StandardScaler().fit(X_train)
    everything = StandardScaler().fit(dataset.X)

    assert np.allclose(fitted_scaler.mean_, train_only.mean_)
    assert not np.allclose(fitted_scaler.mean_, everything.mean_), (
        "the scaler's statistics match a fit on the whole dataset"
    )


def test_the_deep_adapter_never_touches_the_zoo_validation_partition():
    """Early stopping must split the training rows it was handed, nothing else."""
    from research.model_zoo.adapters.torch_adapter import INTERNAL_VALIDATION_FRACTION

    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=500, seed=6)
    X_train, y_train, _X, _y = synthetic.split(dataset, seed=6)

    model = _build("mlp").fit(X_train, y_train)

    assert 0 < INTERNAL_VALIDATION_FRACTION < 0.5
    assert model.training_record.training_rows == len(X_train)
    assert "validation partition was not used" in model.training_record.notes
    # The standardiser saw only the inner training split, so it cannot have been
    # fitted on all the rows handed in.
    assert model.standardiser.fitted_rows < len(X_train)


def test_a_model_fitted_on_a_subset_reports_that_subset_size():
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=400, seed=8)
    X_train, y_train, _X, _y = synthetic.split(dataset, seed=8)

    model = _build("random_forest", n_estimators=5).fit(X_train.head(120), y_train.head(120))

    assert model.training_record.training_rows == 120


# ==================================================== the serialization lab

@pytest.mark.parametrize("model_id", [*REPRESENTATIVE, "nearest_centroid"])
def test_every_model_survives_a_round_trip_unchanged(model_id, tmp_path):
    """Saving is not enough: the reloaded model must predict the same values."""
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=400, seed=12)
    X_train, y_train, X_test, _y = synthetic.split(dataset, seed=12)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(model_id).fit(X_train, y_train)

    record = serialization.round_trip(model_id, model, X_test, directory=tmp_path)

    assert record.round_trip_ok, f"{model_id}: {record.error}"
    assert record.bytes_written > 0
    assert record.max_abs_difference is not None
    assert record.max_abs_difference <= serialization.TOLERANCE


def test_a_serialization_failure_is_reported_rather_than_raised(tmp_path, monkeypatch):
    """A model that cannot be saved is a row in the table, not a crashed run."""
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=200, seed=1)
    X_train, y_train, X_test, _y = synthetic.split(dataset, seed=1)
    model = _build("logistic_l2").fit(X_train, y_train)

    def explode(*args, **kwargs):
        raise OSError("disk is on fire")

    monkeypatch.setattr(model, "serialize", explode)
    record = serialization.round_trip("logistic_l2", model, X_test, directory=tmp_path)

    assert record.round_trip_ok is False
    assert "disk is on fire" in (record.error or "")


def test_a_model_whose_predictions_move_on_reload_is_caught(tmp_path, monkeypatch):
    """The check must be able to fail, or it proves nothing."""
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=200, seed=1)
    X_train, y_train, X_test, _y = synthetic.split(dataset, seed=1)
    model = _build("logistic_l2").fit(X_train, y_train)

    real_restore = serialization._restore

    def drifted(model_, path):
        restored = real_restore(model_, path)
        original = restored.decision_scores

        def shifted(X):
            return np.asarray(original(X)) + 0.01

        restored.decision_scores = shifted
        return restored

    monkeypatch.setattr(serialization, "_restore", drifted)
    record = serialization.round_trip("logistic_l2", model, X_test, directory=tmp_path)

    assert record.round_trip_ok is False
    assert "moved by" in (record.error or "")


def test_the_serialization_summary_names_failures():
    from research.model_zoo.contracts import SerializationRecord

    records = {
        "good": SerializationRecord("joblib", 100, True, 0.0),
        "bad": SerializationRecord("joblib", 100, False, 0.5, error="state lost"),
    }

    text = serialization.summarise(records)

    assert "1 model(s) failed the round trip" in text
    assert "bad: state lost" in text


# ================================================== the runner's contracts

def test_the_runner_records_a_failure_instead_of_crashing(tmp_path, monkeypatch):
    """A model that explodes must become a FAILED row with its reason attached."""
    from research.model_zoo import run as runner

    real_build = REGISTRY.build

    def sometimes_explode(model_id, **kwargs):
        if model_id == "qda":
            raise ValueError("singular covariance")
        return real_build(model_id, **kwargs)

    monkeypatch.setattr(REGISTRY, "build", sometimes_explode)
    manifest = runner.run(
        train_rows=250,
        model_ids=["logistic_l2", "qda"],
        output_root=tmp_path,
    )

    outcomes = {r["model_id"]: r for r in manifest["results"]}
    assert outcomes["logistic_l2"]["outcome"] == RunOutcome.COMPLETED.value
    assert outcomes["qda"]["outcome"] == RunOutcome.FAILED.value
    assert "singular covariance" in outcomes["qda"]["error"]


def test_a_model_over_its_time_budget_is_recorded_not_dropped(tmp_path):
    from research.model_zoo import run as runner

    manifest = runner.run(
        train_rows=250,
        model_ids=["random_forest"],
        time_budget=0.0,  # nothing can fit this fast
        output_root=tmp_path,
    )

    result = manifest["results"][0]
    assert result["outcome"] == RunOutcome.RESOURCE_LIMIT.value
    assert "budget" in result["error"]


def test_absent_optional_dependencies_are_skipped_with_a_reason(tmp_path):
    from research.model_zoo import run as runner

    manifest = runner.run(
        train_rows=250, model_ids=["lightgbm", "catboost"], output_root=tmp_path
    )

    for result in manifest["results"]:
        if REGISTRY.get(result["model_id"]).is_available():
            continue
        assert result["outcome"] == RunOutcome.SKIPPED.value
        assert "not installed" in result["error"]


def test_the_runner_trains_only_on_the_declared_subset(tmp_path):
    from research.model_zoo import run as runner

    manifest = runner.run(train_rows=300, model_ids=["logistic_l2"], output_root=tmp_path)

    assert manifest["train_rows"] == 300
    assert manifest["test_rows"] > 10_000, "evaluation must stay at full size"
    subset = manifest["training_subset"]
    assert subset["drawn_from"] == "train"


def test_the_runner_labels_its_evidence_class(tmp_path):
    """Track L must never be mistaken for Track K's stronger evidence."""
    from research.model_zoo import run as runner

    manifest = runner.run(train_rows=250, model_ids=["logistic_l2"], output_root=tmp_path)

    assert manifest["evidence_class"] == "RESOURCE_CONSTRAINED_EXPLORATORY"
    joined = " ".join(manifest["limitations"])
    assert "Track K" in joined
    assert "not a promotion candidate" in joined.lower() or "promotion candidate" in joined


def test_the_runner_touches_no_production_artifact(tmp_path):
    from research.model_zoo import run as runner

    manifest = runner.run(train_rows=250, model_ids=["logistic_l2"], output_root=tmp_path)

    assert manifest["production_artifacts_touched"] is False


def test_a_hard_label_model_gets_undefined_threshold_free_metrics(tmp_path):
    """Rather than a ROC-AUC computed from its 0/1 output."""
    from research.model_zoo import run as runner

    manifest = runner.run(
        train_rows=250, model_ids=["nearest_centroid"], output_root=tmp_path
    )

    metrics = manifest["results"][0]["metrics"]
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["recall"] is not None, "threshold metrics are still defined"
    assert "undefined" in metrics["threshold_free_metrics"]


def test_the_manifest_is_written_last(tmp_path):
    from research.model_zoo import run as runner

    manifest = runner.run(train_rows=250, model_ids=["logistic_l2"], output_root=tmp_path)
    run_dir = tmp_path / manifest["run_id"]

    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "test_predictions.npz").is_file()
