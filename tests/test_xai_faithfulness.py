"""Faithfulness, and the random baseline that decides whether it means anything.

The temptation this file exists to resist is a deletion curve that plunges
dramatically and gets written up as proof the explanation is faithful. On a
model dominated by one feature, *any* ranking produces that plunge - delete
three of ten features and you have a good chance of catching the important one.
`test_a_shuffled_ranking_also_produces_a_dramatic_looking_curve` measures that
directly, so no result here can be read without its control.

The ground-truth worlds make the rest checkable. On a world with one feature
generating the label, the true ranking must beat shuffling and a deliberately
inverted ranking must lose to it. A module that scored the inverted ranking well
would be measuring the act of deleting features rather than the order.
"""
import warnings

import numpy as np
import pytest

from ml_core import feature_contract
from research.model_zoo.registry import REGISTRY
from research.xai import faithfulness, worlds
from research.xai.explainers import classical
from research.xai.worlds import XaiWorld

FEATURES = feature_contract.FEATURE_NAMES


@pytest.fixture(scope="module")
def dominant():
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("random_forest").fit(X_train, y_train)
    return model, X_train, X_eval, worlds.baseline_row(X_train), y_eval


@pytest.fixture(scope="module")
def true_ranking(dominant):
    """GenHlth first, then the rest in a fixed order. The known answer."""
    return ("GenHlth", *[f for f in FEATURES if f != "GenHlth"])


# ==================================================================== curves

def test_a_deletion_curve_starts_at_zero_and_ends_fully_ablated(dominant, true_ranking):
    """Both endpoints are ranking-independent, which is what makes curves comparable.

    Nothing has moved before the first deletion, and after the last one every
    feature sits at its baseline whatever order they were removed in. Two curves
    therefore differ only in the path between, and the area between them is
    entirely about order.
    """
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]

    curve = faithfulness.deletion_curve(model, sample, true_ranking, baseline)
    shuffled = faithfulness.deletion_curve(model, sample, true_ranking[::-1], baseline)

    assert len(curve) == len(FEATURES) + 1
    assert curve[0] == 0.0
    assert curve[-1] == pytest.approx(shuffled[-1])
    assert curve[-1] > 0.0


def test_an_insertion_curve_is_the_deletion_curve_run_backwards_in_endpoints(
    dominant, true_ranking
):
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]

    deletion = faithfulness.deletion_curve(model, sample, true_ranking, baseline)
    insertion = faithfulness.insertion_curve(model, sample, true_ranking, baseline)

    assert insertion[0] == pytest.approx(deletion[-1])
    assert insertion[-1] == pytest.approx(0.0, abs=1e-12)


def test_deleting_the_generating_feature_first_moves_the_score_furthest(
    dominant, true_ranking
):
    """The property the whole module rests on, checked against a known driver."""
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]

    true_first = faithfulness.deletion_curve(model, sample, true_ranking, baseline)
    driver_last = faithfulness.deletion_curve(model, sample, true_ranking[::-1], baseline)

    assert true_first[1] > 10 * driver_last[1], (
        f"removing the driver first shifted the score by {true_first[1]:.4f}; "
        f"removing an inert feature first shifted it by {driver_last[1]:.4f}"
    )
    assert true_first[1] > 0.85 * true_first[-1], (
        "one deletion should account for nearly all the available damage"
    )


def test_a_curve_needs_at_least_two_points_to_have_an_area():
    with pytest.raises(ValueError, match="at least two points"):
        faithfulness.curve_auc([0.5])


def test_curve_area_is_normalised_so_it_does_not_depend_on_the_step_count():
    assert faithfulness.curve_auc([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert faithfulness.curve_auc([0.0, 1.0]) == pytest.approx(0.5)
    assert faithfulness.curve_auc([0.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.875)


def test_the_mean_score_would_have_measured_almost_nothing(dominant, true_ranking):
    """The defect this module was rewritten to avoid, kept as a live check.

    Ablation pushes rows in opposite directions and the average already sits
    between them, so the mean predicted score separates "delete the true driver
    first" from "delete an inert feature first" by a factor of about four, on
    movements of 0.016 against 0.004. Per-row absolute shift separates the same
    two orderings by a factor of thirty-one. The mean is not inert - it is an
    order of magnitude less discriminating, on a scale small enough for anything
    to swamp it, which is the more precise and more useful statement.
    """
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]
    before = float(np.mean(model.decision_scores(sample)))

    def mean_score_move(ranking):
        frame = sample.copy()
        frame[ranking[0]] = baseline[ranking[0]]
        return abs(float(np.mean(model.decision_scores(frame))) - before)

    mean_driver = mean_score_move(true_ranking)
    mean_inert = mean_score_move(true_ranking[::-1])

    shift_driver = faithfulness.deletion_curve(model, sample, true_ranking, baseline)[1]
    shift_inert = faithfulness.deletion_curve(
        model, sample, true_ranking[::-1], baseline
    )[1]

    mean_ratio = mean_driver / mean_inert
    shift_ratio = shift_driver / shift_inert

    assert shift_ratio > 5 * mean_ratio, (
        f"per-row shift separates the orderings {shift_ratio:.1f}-fold and the "
        f"mean separates them {mean_ratio:.1f}-fold; the gap has closed enough "
        "that the simpler statistic is worth reconsidering"
    )
    assert mean_driver < 0.05, (
        f"the mean moved {mean_driver:.4f}, large enough to be read directly; "
        "the cancellation argument in faithfulness.py needs re-measuring"
    )


# =================================================== comprehensiveness / sufficiency

def test_removing_the_driver_costs_more_than_removing_three_inert_features(dominant):
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]

    with_driver = faithfulness.comprehensiveness(
        model, sample, ("GenHlth", "Education", "PhysActivity"), baseline, 3
    )
    without = faithfulness.comprehensiveness(
        model, sample, ("Education", "PhysActivity", "DiffWalk"), baseline, 3
    )

    assert with_driver > 5 * without


def test_keeping_only_the_driver_preserves_most_of_the_score(dominant, true_ranking):
    """Sufficiency: near zero means the shortlist carried the model alone."""
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]

    with_driver = faithfulness.sufficiency(model, sample, true_ranking, baseline, 1)
    without = faithfulness.sufficiency(
        model, sample, ("Education", *[f for f in FEATURES if f != "Education"]),
        baseline, 1,
    )

    assert with_driver < without


@pytest.mark.parametrize("k", [0, -1])
def test_a_shortlist_of_no_features_is_refused(dominant, true_ranking, k):
    model, _, X_eval, baseline, _ = dominant

    with pytest.raises(ValueError, match="k must be positive"):
        faithfulness.comprehensiveness(model, X_eval.iloc[:5], true_ranking, baseline, k)
    with pytest.raises(ValueError, match="k must be positive"):
        faithfulness.sufficiency(model, X_eval.iloc[:5], true_ranking, baseline, k)


# ============================================================ the random control

def test_a_shuffled_ranking_also_produces_a_dramatic_looking_curve(dominant, true_ranking):
    """Why no curve in this package is ever reported without its control.

    A random ranking still destroys most of the score, because deleting features
    one after another eventually deletes the one that matters. The curve alone
    proves nothing; only the gap between a ranking and the shuffled distribution
    says the *order* carried information.
    """
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]
    rng = np.random.default_rng(0)

    true_curve = faithfulness.deletion_curve(model, sample, true_ranking, baseline)
    shuffled_curve = faithfulness.deletion_curve(
        model, sample, list(rng.permutation(list(FEATURES))), baseline
    )

    assert shuffled_curve[-1] == pytest.approx(true_curve[-1], abs=1e-9), (
        "both rankings must end at the same fully-ablated score"
    )
    assert shuffled_curve[-2] > 0.8 * shuffled_curve[-1], (
        "a shuffled ranking still reaches most of the total damage before the "
        "end; the curve alone therefore proves nothing"
    )
    assert faithfulness.curve_auc(true_curve) > faithfulness.curve_auc(shuffled_curve)


def test_the_true_ranking_beats_shuffling_and_its_inverse_loses(dominant, true_ranking):
    """The two-sided check. Beating random is only meaningful if losing is possible.

    A module that scored an inverted ranking well would be measuring the act of
    deleting features rather than the order they were deleted in, and would call
    every ranking faithful.
    """
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:60]

    good = faithfulness.evaluate(model, sample, true_ranking, baseline, seed=1)
    bad = faithfulness.evaluate(model, sample, true_ranking[::-1], baseline, seed=1)

    assert good.beats_random, f"the true ranking scored {good.deletion_gap:+.5f}"
    assert not bad.beats_random, f"an inverted ranking scored {bad.deletion_gap:+.5f}"
    assert good.deletion_gap > bad.deletion_gap


def test_a_result_reports_its_gap_and_the_baseline_it_was_measured_against(
    dominant, true_ranking
):
    model, _, X_eval, baseline, _ = dominant
    result = faithfulness.evaluate(model, X_eval.iloc[:40], true_ranking, baseline, seed=2)
    payload = result.as_dict()

    assert payload["rows"] == 40
    assert payload["top_k"] == 3
    assert payload["random_deletion_auc"] is not None
    assert payload["full_ablation_shift"] > 0.0
    assert payload["deletion_gap"] == pytest.approx(
        result.deletion_auc - result.random_deletion_auc
    )
    assert payload["insertion_gap"] == pytest.approx(
        result.random_insertion_auc - result.insertion_auc
    )


def test_evaluation_is_reproducible_from_its_seed(dominant, true_ranking):
    """The random control is random, so it has to be pinned to be reportable."""
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:40]

    first = faithfulness.evaluate(model, sample, true_ranking, baseline, seed=5)
    again = faithfulness.evaluate(model, sample, true_ranking, baseline, seed=5)

    assert first.random_deletion_auc == pytest.approx(again.random_deletion_auc)
    assert first.deletion_gap == pytest.approx(again.deletion_gap)


# ============================================================ end to end

def test_a_real_explanation_of_a_real_model_is_faithful_on_a_known_world(dominant):
    """Not a hand-built ranking: the one permutation importance actually produced."""
    model, _, X_eval, baseline, y_eval = dominant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributions = classical.permutation_importance(model, X_eval, y_eval, seed=7)
    ranking = tuple(FEATURES[i] for i in np.argsort(-np.abs(attributions)))

    result = faithfulness.evaluate(model, X_eval.iloc[:60], ranking, baseline, seed=3)

    assert ranking[0] == "GenHlth"
    assert result.beats_random


# ================================================================== summaries

def test_summarising_nothing_reports_nothing_rather_than_crashing():
    summary = faithfulness.summarise({})

    assert summary["evaluated"] == 0
    assert summary["best"] is None
    assert summary["failed_random"] == []


def test_a_summary_names_the_methods_that_failed_their_control(dominant, true_ranking):
    """Failures stay in the table. A results file of only what worked flatters."""
    model, _, X_eval, baseline, _ = dominant
    sample = X_eval.iloc[:40]

    results = {
        "true": faithfulness.evaluate(model, sample, true_ranking, baseline, seed=1),
        "inverted": faithfulness.evaluate(model, sample, true_ranking[::-1], baseline, seed=1),
    }
    summary = faithfulness.summarise(results)

    assert summary["evaluated"] == 2
    assert summary["beat_random"] == ["true"]
    assert summary["failed_random"] == ["inverted"]
    assert summary["best"]["name"] == "true"
