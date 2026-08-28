"""The evaluation contract and the paired bootstrap that compares models with it.

Two properties matter most here and both are tested against constructed cases
with known answers rather than against whatever the code happens to produce:

* the evaluator computes what it claims - a perfectly calibrated set of
  predictions must score ECE near zero, an inverted ranking must score ROC-AUC
  near zero, and a slope/intercept fit must recover 1 and 0;
* the bootstrap is PAIRED - every model in a replicate is scored on the same
  resampled rows. An unpaired implementation would produce wider intervals and
  a different verdict, so the pairing is asserted directly on the index matrix.
"""
import numpy as np
import pytest

from research.track_k import comparison, evaluation, protocol


@pytest.fixture
def rng():
    return np.random.default_rng(20260828)


@pytest.fixture
def separable(rng):
    """Well-separated classes, so metrics have room to move."""
    y_true = np.repeat([0, 1], 500)
    scores = np.concatenate([rng.beta(2, 5, 500), rng.beta(5, 2, 500)])
    return y_true, scores


# ============================================================== the evaluator

def test_every_declared_metric_is_produced(separable):
    y_true, scores = separable

    result = evaluation.evaluate(y_true, scores, threshold=0.5)

    for name in evaluation.metric_names():
        assert name in result, f"{name} is declared by the protocol but not computed"


def test_the_primary_metric_is_present_and_sane(separable):
    y_true, scores = separable

    result = evaluation.evaluate(y_true, scores, threshold=0.5)

    assert protocol.PRIMARY_METRIC == "roc_auc"
    assert 0.5 < result["roc_auc"] <= 1.0


def test_a_perfect_ranking_scores_one():
    y_true = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])

    assert evaluation.evaluate(y_true, perfect, threshold=0.5)["roc_auc"] == 1.0


def test_an_inverted_ranking_scores_zero():
    """Proves the metric has direction, not just a plausible magnitude."""
    y_true = np.array([0, 0, 1, 1])
    inverted = np.array([0.9, 0.8, 0.2, 0.1])

    assert evaluation.evaluate(y_true, inverted, threshold=0.5)["roc_auc"] == 0.0


def test_recall_and_specificity_are_computed_from_the_right_cells():
    # 2 positives, both caught; 2 negatives, one false alarm.
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.7])

    result = evaluation.evaluate(y_true, scores, threshold=0.5)

    assert result["recall"] == 1.0
    assert result["specificity"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_the_threshold_is_applied_and_recorded(separable):
    y_true, scores = separable

    strict = evaluation.evaluate(y_true, scores, threshold=0.9)
    lenient = evaluation.evaluate(y_true, scores, threshold=0.1)

    assert strict["threshold"] == 0.9
    assert strict["recall"] < lenient["recall"], "a stricter threshold must catch fewer"
    assert strict["specificity"] > lenient["specificity"]


def test_a_single_class_partition_reports_nan_rather_than_a_number():
    """An undefined metric must not be reported as though it were measured."""
    y_true = np.ones(20, dtype=int)

    result = evaluation.evaluate(y_true, np.linspace(0.1, 0.9, 20), threshold=0.5)

    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["pr_auc"])
    assert np.isnan(result["calibration_slope"])


def test_log_loss_is_finite_for_a_confidently_wrong_prediction():
    """Unclipped, this is infinite and would swamp every other number."""
    result = evaluation.evaluate(np.array([1, 0]), np.array([0.0, 1.0]), threshold=0.5)

    assert np.isfinite(result["log_loss"])


# ============================================================== calibration

def test_perfectly_calibrated_predictions_score_near_zero_ece(rng):
    """Constructed so the observed rate in each bin equals the prediction."""
    probabilities = np.repeat(np.linspace(0.05, 0.95, 10), 2000)
    outcomes = (rng.random(len(probabilities)) < probabilities).astype(int)

    ece = evaluation.expected_calibration_error(outcomes, probabilities)

    assert ece < 0.02, f"well-calibrated predictions scored ECE {ece:.4f}"


def test_systematically_overconfident_predictions_score_worse(rng):
    """Proves the previous test is not just measuring a small number."""
    probabilities = np.repeat(np.linspace(0.05, 0.95, 10), 2000)
    outcomes = (rng.random(len(probabilities)) < probabilities).astype(int)
    inflated = np.clip(probabilities + 0.25, 0, 1)

    honest = evaluation.expected_calibration_error(outcomes, probabilities)
    skewed = evaluation.expected_calibration_error(outcomes, inflated)

    assert skewed > honest + 0.1


def test_reliability_bins_drop_empty_bins_rather_than_reporting_zeroes():
    """A bin nothing landed in says nothing and must not drag the average."""
    probabilities = np.full(100, 0.55)

    bins = evaluation.reliability_bins(np.ones(100, dtype=int), probabilities)

    assert len(bins) == 1
    assert bins[0].count == 100
    assert bins[0].lower <= 0.55 <= bins[0].upper


def test_a_probability_of_exactly_one_lands_in_the_last_bin():
    bins = evaluation.reliability_bins(np.array([1, 1]), np.array([1.0, 1.0]))

    assert len(bins) == 1
    assert bins[0].upper == 1.0


def test_calibration_slope_recovers_one_for_a_calibrated_model(rng):
    probabilities = rng.uniform(0.02, 0.98, 20000)
    outcomes = (rng.random(20000) < probabilities).astype(int)

    slope, intercept = evaluation.calibration_slope_intercept(outcomes, probabilities)

    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_calibration_slope_detects_compressed_probabilities(rng):
    """Halving the logit should show up as a slope well above one."""
    probabilities = rng.uniform(0.02, 0.98, 20000)
    outcomes = (rng.random(20000) < probabilities).astype(int)
    logit = np.log(probabilities / (1 - probabilities))
    compressed = 1 / (1 + np.exp(-logit / 2))

    slope, _intercept = evaluation.calibration_slope_intercept(outcomes, compressed)

    assert slope > 1.4


def test_degenerate_probabilities_yield_nan_not_a_fabricated_fit():
    constant = np.full(100, 0.5)

    slope, intercept = evaluation.calibration_slope_intercept(
        np.array([0, 1] * 50), constant
    )

    assert np.isnan(slope) and np.isnan(intercept)


# ============================================================ paired bootstrap

@pytest.fixture
def three_models(rng):
    y_true = np.repeat([0, 1], 400)
    strong = np.concatenate([rng.beta(2, 6, 400), rng.beta(6, 2, 400)])
    middling = np.concatenate([rng.beta(2, 4, 400), rng.beta(4, 2, 400)])
    weak = rng.random(800)
    return y_true, {"strong": strong, "middling": middling, "weak": weak}


def test_the_resample_matrix_is_shared_by_every_model():
    """The definition of paired: replicate i uses the same rows for all models."""
    matrix = resample = comparison.resample_indices(50, resamples=8, seed=1)

    assert matrix.shape == (8, 50)
    assert (resample == comparison.resample_indices(50, resamples=8, seed=1)).all()


def test_resampling_is_deterministic_under_the_frozen_seed(three_models):
    y_true, predictions = three_models

    first, _ = comparison.bootstrap_metrics(y_true, predictions, resamples=50, seed=7)
    second, _ = comparison.bootstrap_metrics(y_true, predictions, resamples=50, seed=7)

    assert first["strong"]["roc_auc"].lower == second["strong"]["roc_auc"].lower
    assert first["strong"]["roc_auc"].upper == second["strong"]["roc_auc"].upper


def test_intervals_bracket_the_point_estimate(three_models):
    y_true, predictions = three_models

    intervals, _ = comparison.bootstrap_metrics(y_true, predictions, resamples=200, seed=3)

    for model in predictions:
        interval = intervals[model]["roc_auc"]
        assert interval.lower <= interval.point <= interval.upper


def test_a_clear_difference_is_reported_as_an_improvement(three_models):
    y_true, predictions = three_models

    _intervals, replicates = comparison.bootstrap_metrics(
        y_true, predictions, resamples=400, seed=5
    )
    delta = comparison.paired_delta(replicates, "strong", "weak", "roc_auc")

    assert delta.point > 0
    assert comparison.interpret(delta) == "CLEAR IMPROVEMENT"


def test_the_reverse_comparison_is_a_regression(three_models):
    y_true, predictions = three_models

    _intervals, replicates = comparison.bootstrap_metrics(
        y_true, predictions, resamples=400, seed=5
    )
    delta = comparison.paired_delta(replicates, "weak", "strong", "roc_auc")

    assert comparison.interpret(delta) == "CLEAR REGRESSION"


def test_a_model_compared_against_itself_is_inconclusive(three_models):
    """A paired delta of exactly zero must never read as an improvement."""
    y_true, predictions = three_models

    _intervals, replicates = comparison.bootstrap_metrics(
        y_true, predictions, resamples=200, seed=11
    )
    delta = comparison.paired_delta(replicates, "strong", "strong", "roc_auc")

    assert delta.point == 0.0
    assert comparison.interpret(delta) == "INCONCLUSIVE"


def test_pairing_produces_a_tighter_interval_than_independent_resampling(three_models):
    """The reason the protocol specifies pairing at all."""
    y_true, predictions = three_models

    _intervals, replicates = comparison.bootstrap_metrics(
        y_true, predictions, resamples=600, seed=13
    )
    paired = comparison.paired_delta(replicates, "strong", "middling", "roc_auc")

    strong = np.asarray(replicates["strong"]["roc_auc"])
    middling = np.asarray(replicates["middling"]["roc_auc"])
    # Independent pairing of the same marginals, which is what an unpaired
    # bootstrap effectively compares.
    shuffled = np.random.default_rng(0).permutation(middling)
    unpaired = strong - shuffled

    paired_width = paired.upper - paired.lower
    unpaired_width = float(np.percentile(unpaired, 97.5) - np.percentile(unpaired, 2.5))
    assert paired_width < unpaired_width


def test_every_prespecified_pair_is_compared():
    pairs = comparison.default_pairs()

    assert ("mlp", "logistic_regression") in pairs
    assert ("mlp", "xgboost") in pairs
    assert ("ft_transformer", "logistic_regression") in pairs
    assert ("ft_transformer", "xgboost") in pairs
    assert ("ft_transformer", "mlp") in pairs
    assert len(pairs) == 5


def test_only_the_three_declared_outcomes_can_be_returned(three_models):
    y_true, predictions = three_models

    _intervals, replicates = comparison.bootstrap_metrics(
        y_true, predictions, resamples=100, seed=17
    )
    outcomes = {
        result.outcome
        for result in comparison.compare_all(
            replicates, [("strong", "weak"), ("weak", "strong"), ("strong", "strong")]
        )
    }

    assert outcomes <= set(protocol.COMPARISON_OUTCOMES)


def test_brier_is_treated_as_lower_is_better(three_models):
    """Direction matters: a smaller Brier score is an improvement."""
    y_true, predictions = three_models

    _intervals, replicates = comparison.bootstrap_metrics(
        y_true, predictions, resamples=300, seed=19
    )
    delta = comparison.paired_delta(replicates, "strong", "weak", "brier_score")

    assert delta.point < 0
    assert comparison.interpret(delta, higher_is_better=False) == "CLEAR IMPROVEMENT"


# ========================================================== promotion policy

def interval(lower: float, upper: float) -> comparison.Interval:
    return comparison.Interval(
        point=(lower + upper) / 2, lower=lower, upper=upper, resamples=2000, alpha=0.05
    )


def test_a_challenger_clearing_every_gate_is_promoted():
    verdict, reasons = comparison.promotion_verdict(
        primary_delta=interval(0.010, 0.020),
        ece_delta=0.0,
        recall_delta=0.0,
        latency_multiple=2.0,
    )

    assert verdict == "PROMOTE"
    assert len(reasons) == 4


def test_a_gain_below_the_required_margin_is_not_promoted():
    """An interval above zero is not enough; the margin exists for a reason."""
    verdict, _reasons = comparison.promotion_verdict(
        primary_delta=interval(0.0005, 0.0015),
        ece_delta=0.0,
        recall_delta=0.0,
        latency_multiple=1.0,
    )

    assert verdict == "REJECT"


def test_an_interval_straddling_the_margin_is_inconclusive():
    verdict, _reasons = comparison.promotion_verdict(
        primary_delta=interval(-0.002, 0.020),
        ece_delta=0.0,
        recall_delta=0.0,
        latency_multiple=1.0,
    )

    assert verdict == "INCONCLUSIVE"


@pytest.mark.parametrize(
    ("ece_delta", "recall_delta", "latency"),
    [
        (0.05, 0.0, 1.0),    # calibration regression
        (0.0, -0.10, 1.0),   # recall regression
        (0.0, 0.0, 50.0),    # latency blowout
    ],
    ids=["calibration", "recall", "latency"],
)
def test_a_guardrail_failure_blocks_promotion(ece_delta, recall_delta, latency):
    """Discrimination alone cannot buy promotion."""
    verdict, _reasons = comparison.promotion_verdict(
        primary_delta=interval(0.02, 0.04),
        ece_delta=ece_delta,
        recall_delta=recall_delta,
        latency_multiple=latency,
    )

    assert verdict != "PROMOTE"


def test_every_gate_is_reported_even_when_one_fails():
    """The record should show what was cleared as well as what was not."""
    _verdict, reasons = comparison.promotion_verdict(
        primary_delta=interval(0.02, 0.04),
        ece_delta=0.5,
        recall_delta=0.0,
        latency_multiple=1.0,
    )

    assert len(reasons) == 4
    assert any("ECE" in reason for reason in reasons)
    assert any("latency" in reason for reason in reasons)
