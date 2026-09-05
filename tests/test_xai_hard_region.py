"""The rows every model gets wrong, and whether they are explained differently.

The partition here is defined by model errors, which means it is selected on the
very thing being explained. Every statement drawn from it inherits that
selection, and `test_the_reading_is_never_phrased_as_a_cause` holds the module's
own wording to it - the summary must offer an association and must not offer a
mechanism.

The arithmetic is checked against a hand-built partition where the answer is
known by construction, because the real one comes out of a completed run and
cannot be checked against anything.
"""
import warnings

import numpy as np
import pytest

from ml_core import feature_contract
from research.model_zoo.registry import REGISTRY
from research.xai import hard_region, worlds
from research.xai.explainers import classical
from research.xai.worlds import XaiWorld

FEATURES = feature_contract.FEATURE_NAMES


@pytest.fixture(scope="module")
def world():
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("random_forest").fit(X_train, y_train)
    return model, X_train, X_eval, y_eval, worlds.baseline_row(X_train)


# ================================================================= partition

def test_a_partition_counts_rows_by_how_many_models_missed_them():
    truth = np.array([0, 1, 0, 1, 0, 1])
    predictions = {
        "a": np.array([0, 1, 1, 0, 0, 0]),
        "b": np.array([0, 1, 1, 0, 0, 1]),
        "c": np.array([0, 1, 1, 1, 0, 1]),
    }

    result = hard_region.partition(truth, predictions)

    assert result.model_count == 3
    assert result.rows == 6
    assert result.universally_wrong == (2,)
    assert result.universally_right == (0, 1, 4)
    assert result.majority_wrong == (2, 3)


def test_the_majority_band_contains_the_universally_wrong_rows():
    """A band of increasing difficulty, not disjoint buckets.

    Treating them as disjoint would make the two shares incomparable - the
    universal share would not be a subset of the majority share, and a reader
    comparing them would draw the wrong conclusion about how the difficulty is
    distributed.
    """
    truth = np.zeros(10, dtype=int)
    predictions = {f"m{i}": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]) for i in range(4)}

    result = hard_region.partition(truth, predictions)

    assert set(result.universally_wrong) <= set(result.majority_wrong)
    assert result.universally_wrong_share <= result.majority_wrong_share


def test_a_partition_reports_its_shares_as_fractions_of_the_evaluated_rows():
    truth = np.zeros(8, dtype=int)
    predictions = {"a": np.array([1, 1, 0, 0, 0, 0, 0, 0])}

    result = hard_region.partition(truth, predictions)

    assert result.universally_wrong_share == pytest.approx(0.25)
    assert result.as_dict()["rows"] == 8
    assert result.as_dict()["universally_wrong"] == 2


def test_a_partition_needs_models_and_matching_row_counts():
    with pytest.raises(ValueError, match="at least one model"):
        hard_region.partition(np.zeros(3, dtype=int), {})

    with pytest.raises(ValueError, match="but 3 labels were given"):
        hard_region.partition(np.zeros(3, dtype=int), {"a": np.zeros(4, dtype=int)})


def test_an_empty_partition_reports_zero_shares_rather_than_dividing_by_zero():
    result = hard_region.DifficultyPartition((), (), (), model_count=2, rows=0)

    assert result.universally_wrong_share == 0.0
    assert result.majority_wrong_share == 0.0


# ================================================================== profiles

def test_a_profile_is_absolute_before_it_is_averaged():
    """Otherwise a feature pushing two patients opposite ways looks unused."""
    attributions = np.array([[2.0, 1.0], [-2.0, 1.0]])

    profile = hard_region.attribution_profile(attributions)

    assert profile[0] == pytest.approx(2 / 3)
    assert profile[1] == pytest.approx(1 / 3)


def test_a_single_row_of_attributions_is_still_a_profile():
    profile = hard_region.attribution_profile(np.array([3.0, 1.0]))
    assert profile == pytest.approx([0.75, 0.25])


def test_an_all_zero_attribution_becomes_a_flat_profile_not_a_division_by_zero():
    profile = hard_region.attribution_profile(np.zeros((3, 4)))
    assert profile == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_concentration_runs_from_evenly_spread_to_all_on_one_feature():
    """The statistic that separates "confidently wrong" from "undecided".

    A ranking exists either way, so the ordering alone cannot express the
    difference and this is the only place it is visible.
    """
    even = np.full(10, 0.1)
    concentrated = np.array([1.0, *([0.0] * 9)])
    lopsided = np.array([0.7, 0.1, 0.1, 0.1])

    assert hard_region.concentration(even) == pytest.approx(0.0)
    assert hard_region.concentration(concentrated) == pytest.approx(1.0)
    assert 0.0 < hard_region.concentration(lopsided) < 1.0


# ================================================================== contrast

def test_two_regions_explained_identically_report_high_agreement(world):
    """The null case: same rows on both sides must produce no difference.

    If this reported a difference, every real contrast would be measuring the
    sampling of the two regions rather than anything about them.
    """
    model, X_train, _, _, baseline = world

    def explain(frame):
        return np.vstack([
            classical.occlusion_attributions(model, frame.iloc[[i]], baseline)
            for i in range(len(frame))
        ])

    index = list(range(20))
    contrast = hard_region.contrast_regions(explain, X_train, index, index)

    assert contrast.profile_agreement == pytest.approx(1.0)
    assert contrast.hard_concentration == pytest.approx(contrast.easy_concentration)
    assert contrast.hard_top_feature == contrast.easy_top_feature
    assert contrast.largest_shift[1] == pytest.approx(0.0, abs=1e-12)


def test_a_contrast_needs_rows_on_both_sides(world):
    _, X_train, _, _, _ = world

    with pytest.raises(ValueError, match="both regions need at least one row"):
        hard_region.contrast_regions(lambda frame: np.zeros((1, 10)), X_train, [], [1])


def test_a_contrast_reports_the_feature_whose_share_moved_most(world):
    _, X_train, _, _, _ = world
    names = tuple(str(c) for c in X_train.columns)

    contrast = hard_region.RegionContrast(
        hard_rows=5, easy_rows=5,
        hard_profile=(0.6, *([0.4 / 9] * 9)),
        easy_profile=(0.1, 0.5, *([0.4 / 8] * 8)),
        feature_names=names,
        profile_agreement=0.2,
        hard_concentration=0.5,
        easy_concentration=0.4,
    )
    feature, delta = contrast.largest_shift

    assert feature == names[0]
    assert delta == pytest.approx(0.5)
    assert contrast.hard_top_feature == names[0]
    assert contrast.easy_top_feature == names[1]


# ================================================================== summaries

def test_a_matching_pair_of_profiles_is_read_as_the_same_structure():
    profile = (0.5, 0.2, 0.15, 0.1, 0.05)
    contrast = hard_region.RegionContrast(
        hard_rows=10, easy_rows=10,
        hard_profile=profile, easy_profile=profile,
        feature_names=("a", "b", "c", "d", "e"),
        profile_agreement=1.0,
        hard_concentration=0.4, easy_concentration=0.4,
    )

    assert "same structure" in hard_region.summarise(contrast)["summary"]


def test_attribution_spreading_out_on_failures_is_read_as_indecision():
    contrast = hard_region.RegionContrast(
        hard_rows=10, easy_rows=10,
        hard_profile=(0.25, 0.25, 0.25, 0.25),
        easy_profile=(0.85, 0.05, 0.05, 0.05),
        feature_names=("a", "b", "c", "d"),
        profile_agreement=0.1,
        hard_concentration=0.0, easy_concentration=0.6,
    )

    assert "diffuse on failure" in hard_region.summarise(contrast)["summary"]


def test_attribution_narrowing_on_failures_is_read_the_other_way():
    contrast = hard_region.RegionContrast(
        hard_rows=10, easy_rows=10,
        hard_profile=(0.85, 0.05, 0.05, 0.05),
        easy_profile=(0.25, 0.25, 0.25, 0.25),
        feature_names=("a", "b", "c", "d"),
        profile_agreement=0.1,
        hard_concentration=0.6, easy_concentration=0.0,
    )

    assert "concentrated on failure" in hard_region.summarise(contrast)["summary"]


def test_an_ambiguous_contrast_refuses_to_pick_a_story():
    """The summary must be able to say it found nothing.

    A reading that always lands on one of two narratives would support whichever
    story the reader arrived with.
    """
    contrast = hard_region.RegionContrast(
        hard_rows=10, easy_rows=10,
        hard_profile=(0.4, 0.3, 0.2, 0.1),
        easy_profile=(0.1, 0.2, 0.3, 0.4),
        feature_names=("a", "b", "c", "d"),
        profile_agreement=-0.9,
        hard_concentration=0.2, easy_concentration=0.2,
    )

    assert "no clear reading" in hard_region.summarise(contrast)["summary"]


def test_the_reading_is_never_phrased_as_a_cause():
    """The vocabulary boundary, enforced on generated text rather than reviewed.

    This module's output is the most tempting place in Track M to slip into
    causal language, because its subject is failure and its audience wants a
    reason. The wording is checked instead of trusted.
    """
    profile = (0.5, 0.2, 0.15, 0.1, 0.05)
    contrast = hard_region.RegionContrast(
        hard_rows=10, easy_rows=10,
        hard_profile=profile, easy_profile=profile,
        feature_names=("a", "b", "c", "d", "e"),
        profile_agreement=1.0,
        hard_concentration=0.4, easy_concentration=0.4,
    )
    payload = hard_region.summarise(contrast)
    text = " ".join(str(v) for v in payload.values()).lower()

    assert "association" in payload["reading"].lower()
    for forbidden in ("causes", "caused by", "will reduce", "treatment", "cure"):
        assert forbidden not in text, f"the summary used causal wording: {forbidden!r}"
