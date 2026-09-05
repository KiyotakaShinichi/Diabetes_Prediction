"""Rank agreement: the arithmetic, the groupings, and one end-to-end check.

Most of this file is unglamorous: a correlation of a vector with itself is 1, a
constant vector has no ordering to correlate, an empty set of comparisons
summarises to nothing rather than to a crash. Those matter because every
headline number in Track M is an aggregate over thousands of these, and an
aggregate is exactly where a NaN, a silent zero, or a pair quietly dropped from
a mean stops being visible.

Two tests carry more weight than the rest.

`test_a_high_mean_agreement_cannot_hide_a_badly_disagreeing_pair` guards the
claim the track exists to make. "The methods agree" is a sentence that can be
true of an average and false of the evidence, and the summary is required to
surface the worst pair by name so the sentence cannot be written without it.

`test_methods_agree_where_there_is_something_to_agree_about` runs three genuinely
different explainers over a world whose driver is known and checks they converge
- and then checks they stop converging on labels that contain nothing. Agreement
that appears in both cases would be an artefact of the comparison rather than a
property of the models.
"""
import warnings

import numpy as np
import pytest

from ml_core import feature_contract
from research.model_zoo.registry import REGISTRY
from research.xai import agreement, worlds
from research.xai.contracts import (
    AgreementLabel,
    DisagreementLabel,
    ExplanationRecord,
    Scope,
    normalize_attributions,
    rank_features,
)
from research.xai.explainers import classical
from research.xai.worlds import XaiWorld

FEATURES = feature_contract.FEATURE_NAMES


def _record(model_id, family, method, values, *, sample_id=None):
    values = np.asarray(values, dtype=float)
    return ExplanationRecord(
        explanation_id=f"{model_id}-{method}-{sample_id}",
        model_id=model_id,
        model_family=family,
        method=method,
        method_version="1.0.0",
        scope=Scope.GLOBAL,
        feature_names=FEATURES,
        raw_attributions=tuple(float(v) for v in values),
        normalized_attributions=tuple(float(v) for v in normalize_attributions(values)),
        ranking=rank_features(FEATURES, values),
        baseline_reference="not_applicable",
        sample_id=sample_id,
    )


def _descending(shift=0):
    """A clean ordering, optionally rotated so two records disagree by a known amount."""
    values = np.arange(len(FEATURES), 0, -1, dtype=float)
    return np.roll(values, shift)


# =========================================================== the correlations

def test_an_explanation_agrees_perfectly_with_itself():
    values = _descending()
    assert agreement.spearman(values, values) == pytest.approx(1.0)
    assert agreement.kendall_tau(values, values) == pytest.approx(1.0)


def test_a_reversed_ranking_is_perfect_disagreement():
    values = _descending()
    assert agreement.spearman(values, values[::-1]) == pytest.approx(-1.0)
    assert agreement.kendall_tau(values, values[::-1]) == pytest.approx(-1.0)


def test_sign_is_discarded_so_opposite_directions_look_identical():
    """A documented limit of the module, asserted so it cannot surprise anyone.

    Permutation importance has no sign to compare against a coefficient's, so
    everything here ranks by magnitude. The cost is real: two methods that agree
    a feature matters and disagree about which way it pushes are indistinguish-
    able from two that agree completely.
    """
    values = _descending()
    assert agreement.spearman(values, -values) == pytest.approx(1.0)


@pytest.mark.parametrize("constant", [np.zeros(len(FEATURES)), np.ones(len(FEATURES))])
def test_a_constant_attribution_reports_no_agreement_rather_than_a_nan(constant):
    """A NaN here would vanish from a mean instead of counting against it.

    A model that attributed nothing, or spread its attribution perfectly evenly,
    has no ordering to correlate. That is a real result about the model and it
    has to survive aggregation.
    """
    assert agreement.spearman(constant, _descending()) == 0.0
    assert agreement.kendall_tau(constant, _descending()) == 0.0


def test_shortlist_overlap_counts_what_a_reader_would_notice():
    left = list(FEATURES)
    right = [FEATURES[1], FEATURES[0], *FEATURES[2:]]

    assert agreement.top_k_overlap(left, right, 3) == pytest.approx(1.0)
    assert agreement.top_k_overlap(left, list(reversed(left)), 3) == pytest.approx(0.0)
    assert agreement.jaccard(left, right, 3) == pytest.approx(1.0)
    assert agreement.jaccard(left, list(reversed(left)), 3) == pytest.approx(0.0)


def test_a_shortlist_of_no_features_is_refused():
    for size in (0, -1):
        with pytest.raises(ValueError, match="k must be positive"):
            agreement.top_k_overlap(list(FEATURES), list(FEATURES), size)
        with pytest.raises(ValueError, match="k must be positive"):
            agreement.jaccard(list(FEATURES), list(FEATURES), size)


# ================================================================== the bands

@pytest.mark.parametrize(
    ("correlation", "expected"),
    [
        (1.0, AgreementLabel.HIGH),
        (agreement.HIGH_AGREEMENT, AgreementLabel.HIGH),
        (agreement.HIGH_AGREEMENT - 1e-9, AgreementLabel.MODERATE),
        (agreement.MODERATE_AGREEMENT, AgreementLabel.MODERATE),
        (agreement.MODERATE_AGREEMENT - 1e-9, AgreementLabel.LOW),
        (-1.0, AgreementLabel.LOW),
        (float("nan"), AgreementLabel.LOW),
    ],
)
def test_the_bands_are_closed_at_their_lower_edge(correlation, expected):
    assert agreement.label_for(correlation) is expected


def test_the_disagreement_bands_are_the_agreement_bands_read_backwards():
    assert agreement.disagreement_label_for(0.95) is DisagreementLabel.LOW
    assert agreement.disagreement_label_for(0.6) is DisagreementLabel.MODERATE
    assert agreement.disagreement_label_for(0.1) is DisagreementLabel.HIGH


def test_a_band_is_an_adjective_and_never_a_probability():
    """The vocabulary boundary, enforced where the label is produced.

    Nothing in this package estimates the chance an explanation is correct, and
    a band that started calling itself a confidence would be claiming a
    calibration nobody measured.
    """
    for label in (*AgreementLabel, *DisagreementLabel):
        assert "confidence" not in label.value
        assert "probability" not in label.value


# ============================================================== the comparison

def test_comparing_an_explanation_with_itself_agrees_on_every_reading():
    record = _record("logistic_l2", "linear", "coefficients", _descending())
    result = agreement.compare(record, record)

    assert result.spearman == pytest.approx(1.0)
    assert result.kendall_tau == pytest.approx(1.0)
    assert result.top_1_agreement
    assert result.top_3_overlap == pytest.approx(1.0)
    assert result.top_5_jaccard == pytest.approx(1.0)
    assert result.label is AgreementLabel.HIGH


def test_the_headline_correlation_can_be_high_while_the_top_feature_differs():
    """Why top-1 agreement is reported separately rather than inferred.

    "The most important feature is X" is the sentence a reader takes away, and
    it can be wrong while the full ordering looks excellent.
    """
    left = _record("a", "linear", "coefficients", [10, 9.9, 8, 7, 6, 5, 4, 3, 2, 1])
    right = _record("b", "tree", "coefficients", [9.9, 10, 8, 7, 6, 5, 4, 3, 2, 1])

    result = agreement.compare(left, right)

    assert result.spearman > agreement.HIGH_AGREEMENT
    assert not result.top_1_agreement


def test_comparing_explanations_over_different_features_is_refused():
    left = _record("a", "linear", "coefficients", _descending())
    right = ExplanationRecord(
        explanation_id="odd",
        model_id="b",
        model_family="tree",
        method="coefficients",
        method_version="1.0.0",
        scope=Scope.GLOBAL,
        feature_names=("only", "two"),
        raw_attributions=(1.0, 2.0),
        normalized_attributions=(0.33, 0.67),
        ranking=("two", "only"),
        baseline_reference="not_applicable",
    )

    with pytest.raises(ValueError, match="different feature sets"):
        agreement.compare(left, right)


# ================================================================== groupings

def _mixed_records():
    return [
        _record("logistic_l2", "linear", "coefficients", _descending()),
        _record("logistic_l2", "linear", "permutation_importance", _descending(1)),
        _record("logistic_l1", "linear", "coefficients", _descending(2)),
        _record("random_forest", "tree", "coefficients", _descending(3)),
        _record("random_forest", "tree", "permutation_importance", _descending(4)),
    ]


def test_within_model_compares_methods_and_never_crosses_a_model():
    results = agreement.within_model(_mixed_records())

    assert results
    for result in results:
        assert result.left_model == result.right_model
        assert result.left_method != result.right_method


def test_within_family_compares_models_and_holds_the_method_fixed():
    """Holding the method fixed is what makes this measure models, not methods."""
    results = agreement.within_family(_mixed_records())

    assert results
    for result in results:
        assert result.left_method == result.right_method
        assert result.left_model != result.right_model


def test_between_families_only_pairs_models_from_different_families():
    records = _mixed_records()
    results = agreement.between_families(records)

    families = {r.model_id: r.model_family for r in records}
    assert results
    for result in results:
        assert families[result.left_model] != families[result.right_model]
        assert result.left_method == result.right_method


def test_a_single_model_produces_no_cross_family_comparisons():
    records = [_record("logistic_l2", "linear", "coefficients", _descending())]

    assert agreement.between_families(records) == []
    assert agreement.within_family(records) == []
    assert agreement.within_model(records) == []


# ================================================================== consensus

def test_a_consensus_orders_features_by_mean_rank():
    records = [
        _record("a", "linear", "coefficients", [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        _record("b", "tree", "coefficients", [8, 9, 7, 6, 5, 4, 3, 2, 1, 0]),
    ]

    consensus = agreement.consensus_ranking(records)
    ranks = agreement.mean_ranks(records)

    assert set(consensus) == set(FEATURES)
    assert ranks[FEATURES[0]] == pytest.approx(1.5)
    assert ranks[FEATURES[1]] == pytest.approx(1.5)
    assert consensus[2] == FEATURES[2]


def test_a_tied_consensus_is_broken_by_name_so_it_never_changes_between_runs():
    """A different answer on every run would make the stability study measure a sort."""
    records = [_record("a", "linear", "coefficients", np.ones(len(FEATURES)))]

    first = agreement.consensus_ranking(records)
    again = agreement.consensus_ranking(records)

    assert first == again
    assert list(first) == sorted(FEATURES)


def test_a_consensus_needs_explanations_and_one_shared_feature_set():
    with pytest.raises(ValueError, match="at least one explanation"):
        agreement.consensus_ranking([])
    with pytest.raises(ValueError, match="at least one explanation"):
        agreement.mean_ranks([])

    odd = ExplanationRecord(
        explanation_id="odd", model_id="b", model_family="tree", method="m",
        method_version="1.0.0", scope=Scope.GLOBAL, feature_names=("one",),
        raw_attributions=(1.0,), normalized_attributions=(1.0,), ranking=("one",),
        baseline_reference="not_applicable",
    )
    with pytest.raises(ValueError, match="one shared feature set"):
        agreement.consensus_ranking(
            [_record("a", "linear", "coefficients", _descending()), odd]
        )


def test_each_family_gets_its_own_consensus():
    consensus = agreement.family_consensus(_mixed_records())

    assert set(consensus) == {"linear", "tree"}
    assert all(set(order) == set(FEATURES) for order in consensus.values())


# =================================================================== summaries

def test_an_empty_comparison_set_summarises_to_nothing_rather_than_crashing():
    summary = agreement.summarise([])

    assert summary["pairs"] == 0
    assert summary["mean_spearman"] is None
    assert summary["worst_pair"] is None
    assert summary["label_counts"] == {}


def test_a_high_mean_agreement_cannot_hide_a_badly_disagreeing_pair():
    """The guard on the sentence this whole track is trying to earn.

    An average of 0.85 built from nine pairs at 0.92 and one at 0.2 is a
    different finding from ten pairs at 0.85, and only the second licenses "the
    methods agree". The summary must therefore carry the median and the worst
    pair by name, not just the mean.
    """
    values = _descending()
    agreeing = [
        agreement.compare(
            _record(f"a{i}", "linear", "coefficients", values),
            _record(f"b{i}", "tree", "coefficients", values),
        )
        for i in range(9)
    ]
    dissenting = agreement.compare(
        _record("odd", "linear", "coefficients", values),
        _record("out", "kernel", "coefficients", values[::-1]),
    )

    summary = agreement.summarise([*agreeing, dissenting])

    assert summary["pairs"] == 10
    assert summary["median_spearman"] == pytest.approx(1.0)
    assert summary["min_spearman"] == pytest.approx(-1.0)
    assert summary["worst_pair"]["left"] == "odd/coefficients"
    assert summary["worst_pair"]["right"] == "out/coefficients"
    assert summary["label_counts"][AgreementLabel.LOW.value] == 1


def test_the_summary_reports_how_often_the_top_feature_actually_matched():
    results = [
        agreement.compare(
            _record("a", "linear", "coefficients", _descending()),
            _record("b", "tree", "coefficients", _descending()),
        ),
        agreement.compare(
            _record("c", "linear", "coefficients", _descending()),
            _record("d", "tree", "coefficients", _descending(3)),
        ),
    ]

    assert agreement.summarise(results)["top_1_agreement_rate"] == pytest.approx(0.5)


# ============================================================ end to end

def _summary_for(world, seed):
    """Three genuinely different explainers over one world, summarised."""
    dataset = worlds.make(world, rows=600, seed=seed)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linear = REGISTRY.build("logistic_l2").fit(X_train, y_train)
        forest = REGISTRY.build("random_forest").fit(X_train, y_train)

    records = [
        _record("logistic_l2", "linear", "coefficients",
                classical.coefficient_attributions(linear)),
        _record("random_forest", "tree", "native_importance",
                classical.native_importance_attributions(forest)),
        _record("random_forest", "tree", "permutation_importance",
                classical.permutation_importance(forest, X_eval, y_eval, seed=7)),
    ]
    return agreement.summarise(
        [
            agreement.compare(left, right)
            for index, left in enumerate(records)
            for right in records[index + 1:]
        ]
    )


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_methods_agree_about_the_driver_and_stop_agreeing_without_one(seed):
    """Three unrelated explainers, one known driver, and a control.

    A coefficient, an impurity decrease and a permutation drop are three
    different quantities computed from two different models, so converging on
    the feature that generated the label is real evidence - but only if the same
    three stop converging when the labels contain nothing. Otherwise the
    agreement would be a property of the comparison rather than of the models.
    """
    signal = _summary_for(XaiWorld.ONE_DOMINANT_FEATURE, seed)
    noise = _summary_for(XaiWorld.PURE_NOISE, seed)

    assert signal["top_1_agreement_rate"] == pytest.approx(1.0), (
        "three methods disagreed about the feature that generated the label"
    )
    assert noise["top_1_agreement_rate"] == pytest.approx(0.0), (
        "three methods agreed on a top feature where there is nothing to find"
    )


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_the_shortlist_reading_only_separates_once_there_are_several_drivers(seed):
    """Top-3 overlap is informative in proportion to how many features matter.

    With a single dominant driver, ranks two and three are noise in the signal
    world exactly as they are in the control, and at seed 3 the two worlds score
    an identical 0.444 - the shortlist reading carries nothing there and only
    top-1 separates them. Give the world two real drivers and the same statistic
    starts working, because there is now a second position with something in it.
    """
    additive = _summary_for(XaiWorld.ADDITIVE_TWO_FEATURE, seed)
    noise = _summary_for(XaiWorld.PURE_NOISE, seed)

    assert additive["mean_top_3_overlap"] > noise["mean_top_3_overlap"]


def test_full_ranking_correlation_is_dominated_by_the_uninformative_tail():
    """A finding about the metric, not a defect - and the reason for reporting four.

    When one feature takes nearly all the attribution, nine of ten ranks are
    ordering noise, so Spearman over the full ranking is mostly a measurement of
    that noise. Measured across seeds, the correlation on a world with a real
    driver (0.20 to 0.54) overlaps the correlation on pure noise (-0.01 to 0.41)
    while top-1 agreement separates the two perfectly. Any Track M sentence of
    the form "the methods agree, Spearman 0.5" has to survive this test's
    existence.
    """
    signal = _summary_for(XaiWorld.ONE_DOMINANT_FEATURE, 5)
    noise = _summary_for(XaiWorld.PURE_NOISE, 3)

    assert signal["top_1_agreement_rate"] > noise["top_1_agreement_rate"]
    assert signal["mean_spearman"] < noise["mean_spearman"], (
        "full-ranking correlation has started separating signal from noise on "
        "this data; the caveat in agreement.py should be re-measured"
    )
