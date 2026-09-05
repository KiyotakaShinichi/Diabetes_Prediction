"""Do two explanations say the same thing, and in what sense of "same"?

Track L established that twenty-nine algorithms make nearly identical
predictions on this dataset. The obvious follow-up - do they do it for the same
reasons? - only becomes answerable once "the same" is pinned to something
measurable, because two explanations can agree completely on one reading and
disagree completely on another.

So agreement is reported four ways, and they are not redundant:

* **Spearman** over the full ranking answers "is the whole ordering similar?".
  It is the headline number, and it is also the most forgiving: two methods can
  correlate at 0.9 while disagreeing about which feature comes first.
* **Kendall's tau** counts discordant pairs instead of squaring rank distances,
  so a single feature thrown from second place to last hurts it less than it
  hurts Spearman. Where the two diverge, one method has moved one feature a
  long way rather than shuffling everything a little.
* **Top-1 agreement** is the one a reader actually acts on. "The most important
  feature is X" is the sentence that escapes into a slide deck, and it can be
  false while Spearman looks excellent.
* **Top-k overlap and Jaccard** sit between the two: which features made the
  shortlist, ignoring their order inside it.

Which of the four is the headline depends on the data, and on this data it is
not Spearman. Measured on a world with one feature generating the label, three
different explainers agree on the top feature at every seed tried and correlate
over the full ranking at only 0.20 to 0.54 - a band that overlaps what the same
three methods score on labels containing no signal at all (-0.01 to 0.41). The
reason is arithmetic rather than surprising: when one feature takes almost all
the attribution, nine of ten ranks are ordering noise, and a full-ranking
correlation is mostly a measurement of that noise.

Top-1 agreement separates those two worlds perfectly - 1.0 against 0.0 at every
seed. Top-3 overlap does not: with a single driver, ranks two and three are
noise on both sides, and at one seed the two worlds score an identical 0.444.
Give the world a second real driver and top-3 starts working again. So the
shortlist readings are informative in proportion to how many features genuinely
carry signal, which is itself worth reporting rather than assuming.

Three deliberate restrictions.

**Only ranks are compared, never raw values.** A logistic coefficient, an
impurity decrease and a permutation drop in ROC-AUC points are different
physical quantities. Correlating them directly would produce a number that
looks like a comparison and is not one.

**Sign is discarded**, because it does not survive the translation. Permutation
importance has no sign to compare against a coefficient's, so the ranking is by
magnitude throughout - which means this module cannot detect two methods that
agree on a feature's importance and disagree about its direction. That is a
real limit and the report states it.

**The bands are frozen here, before any aggregate was computed**, and they are
adjectives rather than probabilities. "High agreement" is a description of a
correlation. Nothing in this module estimates the chance that an explanation is
correct, and no threshold here should ever be read as one.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from research.xai.contracts import AgreementLabel, DisagreementLabel, ExplanationRecord

#: Either a stored record's tuple of floats or the raw array an explainer just
#: produced. Both arrive here and both are read only for their ordering.
AttributionVector = Sequence[float] | np.ndarray

#: Spearman bands, fixed before any result was looked at. Chosen so that the
#: middle band is genuinely ambiguous rather than a narrow strip nothing lands
#: in: below 0.5 the orderings share little, above 0.8 they differ mainly in
#: the tail nobody reads.
HIGH_AGREEMENT: float = 0.80
MODERATE_AGREEMENT: float = 0.50

#: Shortlist sizes. Three is roughly what a person remembers from a chart; five
#: is half the feature set, past which "overlap" stops discriminating because
#: ten-choose-eight leaves little room to disagree.
TOP_K_SMALL: int = 3
TOP_K_LARGE: int = 5


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """One pairwise comparison, on all four readings at once."""

    left: str
    right: str
    left_model: str
    right_model: str
    left_method: str
    right_method: str
    spearman: float
    kendall_tau: float
    top_1_agreement: bool
    top_3_overlap: float
    top_5_jaccard: float
    label: AgreementLabel

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "left_model": self.left_model,
            "right_model": self.right_model,
            "left_method": self.left_method,
            "right_method": self.right_method,
            "spearman": self.spearman,
            "kendall_tau": self.kendall_tau,
            "top_1_agreement": self.top_1_agreement,
            "top_3_overlap": self.top_3_overlap,
            "top_5_jaccard": self.top_5_jaccard,
            "label": self.label.value,
        }


def label_for(correlation: float) -> AgreementLabel:
    """Band a rank correlation. A description, never a confidence."""
    if not np.isfinite(correlation):
        return AgreementLabel.LOW
    if correlation >= HIGH_AGREEMENT:
        return AgreementLabel.HIGH
    if correlation >= MODERATE_AGREEMENT:
        return AgreementLabel.MODERATE
    return AgreementLabel.LOW


def disagreement_label_for(correlation: float) -> DisagreementLabel:
    """The same bands read from the other end, for the disagreement summary."""
    band = label_for(correlation)
    if band is AgreementLabel.HIGH:
        return DisagreementLabel.LOW
    if band is AgreementLabel.MODERATE:
        return DisagreementLabel.MODERATE
    return DisagreementLabel.HIGH


def spearman(left: AttributionVector, right: AttributionVector) -> float:
    """Rank correlation between two attribution magnitude vectors.

    Returns 0.0 rather than NaN when either side is constant - a model that
    spread its attribution perfectly evenly, or one that attributed nothing at
    all. There is no ordering to correlate in that case, and propagating a NaN
    into an aggregate would quietly drop the pair from a mean instead of
    recording that it carried no information.
    """
    from scipy.stats import spearmanr

    a, b = _magnitudes(left), _magnitudes(right)
    if a.size < 2 or _is_constant(a) or _is_constant(b):
        return 0.0
    value = float(spearmanr(a, b).statistic)
    return 0.0 if not np.isfinite(value) else value


def kendall_tau(left: AttributionVector, right: AttributionVector) -> float:
    """Concordant minus discordant pairs, normalised. Tie-aware (tau-b)."""
    from scipy.stats import kendalltau

    a, b = _magnitudes(left), _magnitudes(right)
    if a.size < 2 or _is_constant(a) or _is_constant(b):
        return 0.0
    value = float(kendalltau(a, b).statistic)
    return 0.0 if not np.isfinite(value) else value


def top_k_overlap(left: Sequence[str], right: Sequence[str], k: int) -> float:
    """Fraction of each shortlist the two rankings share."""
    if k <= 0:
        raise ValueError("k must be positive")
    head_a, head_b = set(left[:k]), set(right[:k])
    return len(head_a & head_b) / float(k)


def jaccard(left: Sequence[str], right: Sequence[str], k: int) -> float:
    """Intersection over union of two shortlists.

    Harsher than overlap when the lists are different lengths, which happens
    whenever a ranking is shorter than k - a model with fewer features scored.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    head_a, head_b = set(left[:k]), set(right[:k])
    union = head_a | head_b
    return len(head_a & head_b) / float(len(union)) if union else 0.0


def compare(left: ExplanationRecord, right: ExplanationRecord) -> AgreementResult:
    """Compare two explanations on all four readings.

    Both records must describe the same features in the same order; comparing
    rankings over different feature sets would produce a number with no meaning
    and no obvious sign of trouble.
    """
    if left.feature_names != right.feature_names:
        raise ValueError(
            "cannot compare explanations over different feature sets: "
            f"{left.feature_names} against {right.feature_names}"
        )

    correlation = spearman(left.normalized_attributions, right.normalized_attributions)
    return AgreementResult(
        left=left.explanation_id,
        right=right.explanation_id,
        left_model=left.model_id,
        right_model=right.model_id,
        left_method=left.method,
        right_method=right.method,
        spearman=correlation,
        kendall_tau=kendall_tau(
            left.normalized_attributions, right.normalized_attributions
        ),
        top_1_agreement=left.ranking[0] == right.ranking[0],
        top_3_overlap=top_k_overlap(left.ranking, right.ranking, TOP_K_SMALL),
        top_5_jaccard=jaccard(left.ranking, right.ranking, TOP_K_LARGE),
        label=label_for(correlation),
    )


# ================================================================ groupings

def within_model(records: Iterable[ExplanationRecord]) -> list[AgreementResult]:
    """Every pair of *methods* applied to the same model.

    Answers "does this model get a consistent story told about it?". A model
    whose methods disagree with each other is not a model whose explanation can
    be quoted without naming the method that produced it.
    """
    grouped: dict[str, list[ExplanationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.model_id].append(record)

    return [
        compare(left, right)
        for group in grouped.values()
        for left, right in combinations(group, 2)
        if left.method != right.method
    ]


def within_family(records: Iterable[ExplanationRecord]) -> list[AgreementResult]:
    """Every pair of *models* inside one family, holding the method fixed.

    Holding the method fixed is what makes this measure the models rather than
    the methods. Two random forests explained by permutation importance that
    disagree are telling you something about forests; two different methods
    disagreeing tells you nothing about them.
    """
    grouped: dict[tuple[str, str], list[ExplanationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.model_family, record.method)].append(record)

    return [
        compare(left, right)
        for group in grouped.values()
        for left, right in combinations(group, 2)
        if left.model_id != right.model_id
    ]


def between_families(records: Iterable[ExplanationRecord]) -> list[AgreementResult]:
    """Every cross-family pair, holding the method fixed.

    The headline question of the track: given that these families predict alike,
    do they attribute alike?
    """
    grouped: dict[str, list[ExplanationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.method].append(record)

    return [
        compare(left, right)
        for group in grouped.values()
        for left, right in combinations(group, 2)
        if left.model_family != right.model_family
    ]


# ================================================================ consensus

def consensus_ranking(records: Sequence[ExplanationRecord]) -> tuple[str, ...]:
    """Features ordered by mean rank across a set of explanations.

    Mean rank rather than mean attribution, for the same reason the pairwise
    comparisons use ranks: the underlying magnitudes are in incompatible units,
    and averaging them would let whichever method happens to produce the
    largest numbers decide the answer.

    Ties break by feature name, so a set of explanations that genuinely cannot
    order two features produces the same consensus every run rather than a
    different one each time.
    """
    if not records:
        raise ValueError("a consensus needs at least one explanation")

    features = records[0].feature_names
    for record in records:
        if record.feature_names != features:
            raise ValueError("consensus requires one shared feature set")

    mean_rank = {
        feature: float(np.mean([record.rank_of(feature) for record in records]))
        for feature in features
    }
    return tuple(sorted(features, key=lambda f: (mean_rank[f], f)))


def mean_ranks(records: Sequence[ExplanationRecord]) -> dict[str, float]:
    """The mean rank behind a consensus, so a near-tie is visible as one.

    A consensus ordering alone hides how close the contest was; two features
    separated by 0.05 of a rank are reported as first and second exactly as
    firmly as two separated by six.
    """
    if not records:
        raise ValueError("a consensus needs at least one explanation")
    features = records[0].feature_names
    return {
        feature: float(np.mean([record.rank_of(feature) for record in records]))
        for feature in features
    }


def family_consensus(
    records: Iterable[ExplanationRecord],
) -> dict[str, tuple[str, ...]]:
    """One consensus ranking per model family."""
    grouped: dict[str, list[ExplanationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.model_family].append(record)
    return {family: consensus_ranking(group) for family, group in grouped.items()}


def summarise(results: Sequence[AgreementResult]) -> dict[str, Any]:
    """Aggregate a set of comparisons without hiding the spread.

    Reports the median alongside the mean and keeps the worst pair by name. An
    average agreement of 0.85 built from one pair at 0.2 and nine at 0.92 is a
    different finding from ten pairs at 0.85, and only the second one licenses
    "the methods agree".
    """
    if not results:
        return {
            "pairs": 0,
            "mean_spearman": None,
            "median_spearman": None,
            "min_spearman": None,
            "top_1_agreement_rate": None,
            "mean_top_3_overlap": None,
            "label_counts": {},
            "worst_pair": None,
        }

    correlations = np.array([r.spearman for r in results], dtype=float)
    worst = min(results, key=lambda r: r.spearman)
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result.label.value] += 1

    return {
        "pairs": len(results),
        "mean_spearman": float(np.mean(correlations)),
        "median_spearman": float(np.median(correlations)),
        "min_spearman": float(correlations.min()),
        "top_1_agreement_rate": float(np.mean([r.top_1_agreement for r in results])),
        "mean_top_3_overlap": float(np.mean([r.top_3_overlap for r in results])),
        "label_counts": dict(counts),
        "worst_pair": {
            "left": f"{worst.left_model}/{worst.left_method}",
            "right": f"{worst.right_model}/{worst.right_method}",
            "spearman": worst.spearman,
        },
    }


def _magnitudes(values: AttributionVector) -> np.ndarray:
    magnitudes: np.ndarray = np.abs(np.asarray(values, dtype=float))
    return magnitudes


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.ptp(values) == 0.0)
