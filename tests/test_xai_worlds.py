"""The ground-truth worlds must be what they claim to be.

Every explainer test downstream reads these worlds as truth: "the driver is
GenHlth, so a correct method ranks GenHlth first." That reasoning is only valid
if the world actually has the property its metadata asserts, and nothing about
a label-generating expression guarantees it. A noise draw large enough to swamp
the signal would make `ONE_DOMINANT_FEATURE` unlearnable; an exclusive-or built
from correlated inputs would leak a marginal association and quietly turn the
interaction trap into an ordinary two-feature problem.

So these tests check the worlds before anything is asked of an explainer. The
XOR marginal-independence test is the load-bearing one: if either driver had a
marginal association with the label, the interaction findings in Track M would
be measuring something else entirely and would still look perfectly reasonable.
"""
import json
import warnings

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ml_core import feature_contract
from research.model_zoo.registry import REGISTRY
from research.xai import worlds
from research.xai.worlds import XaiWorld

ALL_WORLDS = list(XaiWorld)


def _fit(model_id, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REGISTRY.build(model_id).fit(X, y)


def _held_out_auc(model_id, dataset):
    X_train, y_train, X_test, y_test = worlds.split(dataset, seed=3)
    model = _fit(model_id, X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_test, model.decision_scores(X_test)))


# ================================================== the worlds are well formed

@pytest.mark.parametrize("world", ALL_WORLDS)
def test_every_world_produces_contract_valid_features(world):
    """Same columns, same order, same ranges as the real serving contract.

    The worlds run through the same pipelines, scalers and ordinal vocabularies
    as the real data. A column out of range would fail somewhere deep in a
    preprocessing step and look like an explainer bug.
    """
    dataset = worlds.make(world, rows=300, seed=1)

    assert tuple(dataset.X.columns) == feature_contract.FEATURE_NAMES
    for spec in feature_contract.FEATURE_SPECS:
        column = dataset.X[spec.name]
        assert column.min() >= spec.minimum, f"{spec.name} below contract minimum"
        assert column.max() <= spec.maximum, f"{spec.name} above contract maximum"


@pytest.mark.parametrize("world", ALL_WORLDS)
def test_a_world_is_reproducible_from_its_seed(world):
    first = worlds.make(world, rows=200, seed=17)
    second = worlds.make(world, rows=200, seed=17)
    other = worlds.make(world, rows=200, seed=18)

    assert first.X.equals(second.X)
    assert first.y.equals(second.y)
    assert not first.X.equals(other.X), "different seeds produced identical rows"


@pytest.mark.parametrize("world", ALL_WORLDS)
def test_every_world_sits_near_balanced_prevalence(world):
    """Median thresholding, so no world is an accidental imbalance test.

    A world whose base rate drifted with the noise draw would move every
    attribution magnitude for reasons unrelated to the explainer under test.
    """
    dataset = worlds.make(world, rows=800, seed=5)
    assert 0.40 <= float(dataset.y.mean()) <= 0.60


@pytest.mark.parametrize("world", ALL_WORLDS)
def test_driving_and_inert_features_partition_the_contract(world):
    dataset = worlds.make(world, rows=100, seed=1)

    assert set(dataset.driving_features).isdisjoint(dataset.inert_features)
    assert set(dataset.driving_features) | set(dataset.inert_features) == set(
        feature_contract.FEATURE_NAMES
    )
    assert all(dataset.is_driving(f) for f in dataset.driving_features)
    assert not any(dataset.is_driving(f) for f in dataset.inert_features)


@pytest.mark.parametrize("world", ALL_WORLDS)
def test_a_world_describes_itself_as_plain_json(world):
    """Evidence is JSON. A world's metadata travels with the results."""
    payload = worlds.make(world, rows=100, seed=1).as_dict()
    restored = json.loads(json.dumps(payload))

    assert restored["world"] == world.value
    assert restored["rows"] == 100
    assert restored["expectation"]
    assert restored["description"]


def test_an_unknown_world_is_refused():
    with pytest.raises(ValueError, match="unknown XAI world"):
        worlds.make("not_a_world", rows=10)  # type: ignore[arg-type]


# ============================================ the worlds have the properties
# ============================================ the explainer tests depend on

def test_the_dominant_world_has_exactly_one_marginally_associated_feature():
    """GenHlth carries the signal and no other column carries any.

    Asserted on the raw columns rather than through a model, so this is a
    property of the data and cannot be rescued by a model that happens to fit.
    """
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=800, seed=5)

    driver = abs(roc_auc_score(dataset.y, dataset.X["GenHlth"]) - 0.5)
    others = [
        abs(roc_auc_score(dataset.y, dataset.X[f]) - 0.5)
        for f in dataset.inert_features
    ]

    assert driver > 0.30, f"the declared driver is only {driver:.3f} from chance"
    assert max(others) < 0.10, "an inert column carries a marginal association"


def test_the_additive_world_gives_its_two_drivers_comparable_weight():
    """Neither driver dominates, so a method must find both to be right."""
    dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=800, seed=5)

    strengths = {
        f: abs(roc_auc_score(dataset.y, dataset.X[f]) - 0.5)
        for f in dataset.driving_features
    }
    weakest, strongest = min(strengths.values()), max(strengths.values())

    assert weakest > 0.15, f"a declared driver is nearly inert: {strengths}"
    assert strongest < 2.0 * weakest, f"one driver dominates the other: {strengths}"


def test_neither_xor_driver_has_any_marginal_association_with_the_label():
    """The defining property of the interaction trap, and the fragile one.

    If either driver leaked a marginal signal, partial dependence would find it,
    the "PD is blind to interaction" result would evaporate, and nothing in the
    downstream tests would look wrong - they would simply be measuring an
    ordinary two-feature world while claiming otherwise.
    """
    dataset = worlds.make(XaiWorld.XOR_INTERACTION, rows=2000, seed=5)

    for driver in dataset.driving_features:
        marginal = roc_auc_score(dataset.y, dataset.X[driver])
        assert abs(marginal - 0.5) < 0.05, (
            f"{driver} has a marginal AUC of {marginal:.4f}; the XOR world is "
            "supposed to hide both drivers from any single-feature view"
        )


def test_the_xor_rule_is_recoverable_jointly_even_though_it_hides_marginally():
    """Invisible one at a time, obvious together - which is what makes it a trap.

    Without this, marginal independence alone would be satisfied by a world
    containing no signal at all, and the interaction findings would be
    indistinguishable from noise.
    """
    dataset = worlds.make(XaiWorld.XOR_INTERACTION, rows=2000, seed=5)
    parity = (
        dataset.X["HighBP"].astype(int) ^ dataset.X["HighChol"].astype(int)
    ).to_numpy()

    assert roc_auc_score(dataset.y, parity) > 0.90


def test_the_noise_world_contains_no_association_at_all():
    dataset = worlds.make(XaiWorld.PURE_NOISE, rows=2000, seed=5)

    strengths = [
        abs(roc_auc_score(dataset.y, dataset.X[f]) - 0.5)
        for f in feature_contract.FEATURE_NAMES
    ]
    assert max(strengths) < 0.08, "the negative control carries a signal"


# ================================================ the worlds are usable at all

@pytest.mark.parametrize(
    ("world", "model_id", "floor"),
    [
        (XaiWorld.ONE_DOMINANT_FEATURE, "logistic_l2", 0.90),
        (XaiWorld.ONE_DOMINANT_FEATURE, "random_forest", 0.90),
        (XaiWorld.ADDITIVE_TWO_FEATURE, "logistic_l2", 0.90),
        (XaiWorld.ADDITIVE_TWO_FEATURE, "random_forest", 0.90),
    ],
)
def test_the_additive_worlds_are_learnable(world, model_id, floor):
    """An explanation of a model that never fit would be an explanation of noise."""
    auc = _held_out_auc(model_id, worlds.make(world, rows=600, seed=3))
    assert auc > floor, f"{model_id} only reached {auc:.4f} on {world.value}"


def test_a_forest_learns_the_xor_world_and_a_linear_model_cannot():
    """Both halves matter.

    The forest reaching 0.96 is what licenses "partial dependence missed a
    feature the model demonstrably uses". Logistic regression failing is what
    keeps the world honest: the rule really has no linear form, so a linear
    model's attribution there describes a model that learned nothing.
    """
    dataset = worlds.make(XaiWorld.XOR_INTERACTION, rows=600, seed=3)

    forest = _held_out_auc("random_forest", dataset)
    linear = _held_out_auc("logistic_l2", dataset)

    assert forest > 0.85, f"the forest failed to learn XOR ({forest:.4f})"
    assert linear < 0.70, f"a linear model should not fit XOR ({linear:.4f})"


@pytest.mark.parametrize("model_id", ["logistic_l2", "random_forest", "decision_tree"])
def test_no_model_learns_anything_from_the_noise_world(model_id):
    """The leakage canary, restated for the XAI worlds' own split helper."""
    auc = _held_out_auc(model_id, worlds.make(XaiWorld.PURE_NOISE, rows=600, seed=3))
    assert 0.30 < auc < 0.70, (
        f"{model_id} scored {auc:.4f} on independent labels; the split leaks"
    )


# =========================================================== split and baseline

def test_the_split_is_disjoint_reproducible_and_sized_as_asked():
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=500, seed=2)

    X_train, y_train, X_test, y_test = worlds.split(dataset, seed=2)
    again = worlds.split(dataset, seed=2)

    assert len(X_train) == 350 and len(X_test) == 150
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert X_train.equals(again[0]) and X_test.equals(again[2])

    combined = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    assert len(np.unique(combined, axis=0)) == len(np.unique(dataset.X.to_numpy(), axis=0))


def test_the_split_honours_a_different_train_fraction():
    dataset = worlds.make(XaiWorld.PURE_NOISE, rows=400, seed=2)
    X_train, _, X_test, _ = worlds.split(dataset, train_fraction=0.5, seed=2)

    assert len(X_train) == 200
    assert len(X_test) == 200


def test_the_baseline_is_the_median_of_exactly_the_rows_it_was_given():
    """The baseline must not see rows outside the partition handed in.

    Occlusion and integrated gradients both measure against this row. A
    baseline computed over more rows than the caller passed would import
    distribution information the caller believed it had withheld.
    """
    dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=400, seed=4)
    X_train, _, X_test, _ = worlds.split(dataset, seed=4)

    baseline = worlds.baseline_row(X_train)

    assert list(baseline.index) == list(feature_contract.FEATURE_NAMES)
    for name in feature_contract.FEATURE_NAMES:
        assert baseline[name] == pytest.approx(float(X_train[name].median()))

    whole = worlds.baseline_row(dataset.X)
    assert not np.allclose(baseline.to_numpy(), whole.to_numpy()), (
        "the training baseline is indistinguishable from one built on all rows; "
        "this test can no longer detect a partition mistake"
    )
    assert len(X_test) > 0
