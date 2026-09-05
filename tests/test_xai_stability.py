"""Perturbation and seed stability, and the two ways they can lie.

The perturbation machinery has one job that is easy to get subtly wrong: noise
must leave the row inside the served feature contract. Unconstrained Gaussian
noise produces a BMI of 91 and an Education level of 2.7, and a model asked to
score those is extrapolating. The instability that follows is real - the numbers
move - but it measures the model leaving its training distribution rather than
the explanation being fragile, and nothing about the output distinguishes the
two. `test_a_perturbed_row_still_satisfies_the_served_contract` is the guard.

The other trap is the seed sweep. Running one over a deterministic method yields
a variance of exactly zero, which reads in a table as "measured, and found
perfectly stable". `seed_stability` refuses instead, and that refusal is tested
directly.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from ml_core import feature_contract
from research.model_zoo.registry import REGISTRY
from research.xai import stability, worlds
from research.xai.contracts import Determinism
from research.xai.explainers import classical
from research.xai.stability import DeterminismError
from research.xai.worlds import XaiWorld

FEATURES = feature_contract.FEATURE_NAMES


@pytest.fixture(scope="module")
def world():
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("logistic_l2").fit(X_train, y_train)
    return model, X_train, y_train, X_eval, y_eval


@pytest.fixture(scope="module")
def scale(world):
    _, X_train, _, _, _ = world
    return stability.fit_scale(X_train)


# ====================================================================== scale

def test_a_scale_records_the_partition_it_was_fitted_on(world, scale):
    """So a perturbation can be audited back to training rows.

    A scale sized on evaluation rows would leak that distribution into the
    explanation of a model being judged on it, and would look like nothing worse
    than unusually well-behaved noise.
    """
    _, X_train, _, _, _ = world

    assert scale.feature_names == FEATURES
    assert scale.fitted_rows == len(X_train)
    assert scale.source == "training partition"
    for position, name in enumerate(FEATURES):
        assert scale.deviations[position] == pytest.approx(float(X_train[name].std(ddof=0)))


def test_a_constant_column_gets_no_noise_because_it_showed_no_variation(world):
    _, X_train, _, _, _ = world
    frozen = X_train.copy()
    frozen["BMI"] = 30.0

    frozen_scale = stability.fit_scale(frozen)
    perturbed = stability.perturb(frozen, frozen_scale, 1.0, seed=1)

    assert frozen_scale.deviations[FEATURES.index("BMI")] == pytest.approx(0.0)
    assert (perturbed["BMI"] == 30.0).all()


def test_a_scale_is_serialisable_as_plain_data(scale):
    payload = scale.as_dict()

    assert payload["fitted_rows"] == scale.fitted_rows
    assert len(payload["deviations"]) == len(FEATURES)


# ================================================================ perturbation

def test_a_perturbed_row_still_satisfies_the_served_contract(world, scale):
    """The guard that keeps this measuring explanations rather than extrapolation.

    Every value must land back inside the contract's declared range, and every
    integer feature must land on an integer. A model scoring a BMI of 91 is
    outside everything it was fitted on, and the instability that produces would
    be indistinguishable in the output from a fragile explanation.
    """
    _, X_train, _, _, _ = world

    for magnitude in (0.1, 0.5, 1.0, 3.0):
        perturbed = stability.perturb(X_train, scale, magnitude, seed=7)

        assert list(perturbed.columns) == list(X_train.columns)
        for spec in feature_contract.FEATURE_SPECS:
            column = perturbed[spec.name]
            assert column.min() >= spec.minimum, f"{spec.name} escaped below its range"
            assert column.max() <= spec.maximum, f"{spec.name} escaped above its range"
            if spec.dtype is int:
                assert np.allclose(column.to_numpy(), np.rint(column.to_numpy())), (
                    f"{spec.name} is an integer feature but took a fractional value"
                )


def test_a_binary_flag_is_perturbed_by_flipping_and_nothing_else(world, scale):
    """One rule for every feature, and for a binary that rule means a flip.

    Adding noise and rounding gives binary indicators a principled treatment
    without a special case: the flag flips exactly when the noise crosses a
    half. It must never take an intermediate value.
    """
    _, X_train, _, _, _ = world
    perturbed = stability.perturb(X_train, scale, 1.0, seed=5)

    binary = [s.name for s in feature_contract.FEATURE_SPECS if s.kind == "binary"]
    for name in binary:
        assert set(np.unique(perturbed[name].to_numpy())) <= {0.0, 1.0}

    flipped = [
        float((perturbed[name].to_numpy() != X_train[name].to_numpy()).mean())
        for name in binary
    ]
    assert min(flipped) > 0.0, "a full-deviation perturbation flipped nothing"


def test_stronger_perturbations_move_more_rows(world, scale):
    _, X_train, _, _, _ = world

    def moved(magnitude):
        perturbed = stability.perturb(X_train, scale, magnitude, seed=11)
        return float((perturbed.to_numpy() != X_train.to_numpy()).mean())

    assert moved(0.05) < moved(0.5) < moved(2.0)


def test_a_zero_magnitude_perturbation_changes_nothing(world, scale):
    _, X_train, _, _, _ = world
    assert stability.perturb(X_train, scale, 0.0, seed=1).equals(X_train)


def test_perturbation_is_reproducible_from_its_seed(world, scale):
    _, X_train, _, _, _ = world

    first = stability.perturb(X_train, scale, 0.5, seed=3)
    again = stability.perturb(X_train, scale, 0.5, seed=3)
    other = stability.perturb(X_train, scale, 0.5, seed=4)

    assert first.equals(again)
    assert not first.equals(other)


def test_a_negative_perturbation_is_refused(world, scale):
    _, X_train, _, _, _ = world
    with pytest.raises(ValueError, match="cannot be negative"):
        stability.perturb(X_train, scale, -0.1, seed=1)


def test_a_scale_refuses_a_frame_it_was_not_fitted_for(world, scale):
    _, X_train, _, _, _ = world
    renamed = X_train.rename(columns={"BMI": "Weight"})

    with pytest.raises(ValueError, match="scale fitted on"):
        stability.perturb(renamed, scale, 0.5, seed=1)


# ============================================================== the curve

def test_an_explanation_of_a_dominant_feature_survives_real_perturbation(world, scale):
    """The stability that licenses reporting a top feature at all.

    One feature generates the label, so its position should hold under noise
    that genuinely moves the data. If it did not, no statement about which
    feature this model relies on would be worth making.
    """
    model, X_train, _, _, _ = world

    points = stability.stability_curve(
        lambda frame: classical.coefficient_attributions(model),
        X_train, scale, magnitudes=(0.05, 0.25, 1.0), repeats=3, seed=1,
    )
    summary = stability.summarise(points)

    assert [p.magnitude for p in points] == [0.05, 0.25, 1.0]
    assert all(p.replicates == 3 for p in points)
    assert summary["top_1_stable_through"] == 1.0


def test_the_curve_reports_the_spread_and_not_only_the_mean(world, scale):
    model, X_train, _, _, _ = world

    points = stability.stability_curve(
        lambda frame: classical.permutation_importance(
            model, frame, world[2], repeats=1, seed=2
        ),
        X_train, scale, magnitudes=(0.5,), repeats=4, seed=1,
    )

    assert points[0].min_spearman <= points[0].mean_spearman
    assert 0.0 <= points[0].top_1_retention <= 1.0
    assert 0.0 <= points[0].mean_top_3_overlap <= 1.0


def test_a_summary_of_no_measurements_says_so(world):
    summary = stability.summarise([])

    assert summary["measured"] is False
    assert summary["top_1_stable_through"] is None


def test_the_reported_stable_range_stops_at_the_first_magnitude_that_broke(world, scale):
    """A later recovery must not extend the claim.

    Retention is not monotone in noise - a harder perturbation can happen to
    restore the original leader - and reporting the largest magnitude that
    survived would overstate the range a reader can rely on.
    """
    points = [
        stability.StabilityPoint(0.1, 0.9, 0.8, 1.0, 1.0, 3),
        stability.StabilityPoint(0.5, 0.6, 0.4, 0.5, 0.7, 3),
        stability.StabilityPoint(1.0, 0.5, 0.3, 1.0, 0.6, 3),
    ]

    assert stability.summarise(points)["top_1_stable_through"] == 0.1


# ============================================================ seed stability

def test_a_seed_sweep_over_a_deterministic_method_is_refused():
    """Zero variance reported as a measurement is worse than no measurement."""
    with pytest.raises(DeterminismError, match="only to stochastic methods"):
        stability.seed_stability(
            lambda seed: np.arange(len(FEATURES), dtype=float),
            FEATURES,
            determinism=Determinism.DETERMINISTIC,
        )


def test_a_seed_sweep_needs_more_than_one_seed():
    with pytest.raises(ValueError, match="at least two seeds"):
        stability.seed_stability(
            lambda seed: np.arange(len(FEATURES), dtype=float),
            FEATURES,
            determinism=Determinism.STOCHASTIC,
            seeds=(1,),
        )


def test_permutation_importance_is_stable_across_its_own_seeds(world):
    """Stochastic, but not arbitrary - which is what makes it usable as a baseline.

    The method shuffles, so its numbers move between seeds. If its *ranking*
    moved too, the cross-family comparison built on it would be measuring the
    random number generator.
    """
    model, _, _, X_eval, y_eval = world

    result = stability.seed_stability(
        lambda seed: classical.permutation_importance(model, X_eval, y_eval, seed=seed),
        FEATURES,
        determinism=Determinism.STOCHASTIC,
        seeds=(1, 2, 3, 4),
    )

    assert result["top_1_stable"]
    assert result["distinct_top_features"] == ["GenHlth"]
    assert result["mean_spearman"] > 0.5
    assert result["per_feature_dispersion"]["GenHlth"] >= 0.0
    assert set(result["per_feature_dispersion"]) == set(FEATURES)


def test_a_seed_sweep_reports_every_leader_it_saw():
    """Naming the alternatives is the difference between a spread and a summary."""
    rankings = {
        1: np.array([9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        2: np.array([8.0, 9, 7, 6, 5, 4, 3, 2, 1, 0]),
    }

    result = stability.seed_stability(
        lambda seed: rankings[seed],
        FEATURES,
        determinism=Determinism.STOCHASTIC,
        seeds=(1, 2),
    )

    assert not result["top_1_stable"]
    assert result["distinct_top_features"] == sorted([FEATURES[0], FEATURES[1]])


def test_the_scale_is_never_taken_from_the_evaluation_rows(world):
    """A property test for the leak the module is built to prevent.

    Training and evaluation partitions have measurably different spreads, so a
    scale fitted on one is distinguishable from a scale fitted on the other. If
    they ever became identical this test would stop being able to detect the
    mistake, and it says so rather than passing quietly.
    """
    _, X_train, _, X_eval, _ = world

    train_scale = stability.fit_scale(X_train)
    eval_scale = stability.fit_scale(X_eval)

    assert train_scale.fitted_rows == len(X_train)
    assert not np.allclose(train_scale.deviations, eval_scale.deviations), (
        "the two partitions now have identical spreads; this test can no longer "
        "tell a training-fitted scale from an evaluation-fitted one"
    )


def test_a_perturbed_frame_keeps_the_index_and_dtypes_a_model_expects(world, scale):
    _, X_train, _, _, _ = world
    perturbed = stability.perturb(X_train, scale, 0.5, seed=1)

    assert isinstance(perturbed, pd.DataFrame)
    assert perturbed.index.equals(X_train.index)
    assert len(perturbed) == len(X_train)
