"""Paired bootstrap intervals, and the verdicts derived from them.

Every model in Track K predicts the SAME test rows. That makes the comparison
paired, and pairing is not a detail: an unpaired bootstrap resamples each model's
rows independently, so the interval it produces includes variance from which
rows landed in the test split - variance shared by every model and therefore
irrelevant to which model is better. Pairing removes it.

The implementation draws one index resample per replicate and applies that same
index set to every model, then computes each metric and each pairwise delta
inside that replicate.

No p-values. A percentile bootstrap interval is not a hypothesis test, and
calling one significant would overclaim. Verdicts are the three the protocol
allows, decided by whether the interval contains zero.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.track_k import protocol
from research.track_k.deep.training import roc_auc


@dataclass(frozen=True, slots=True)
class Interval:
    """A percentile bootstrap interval."""

    point: float
    lower: float
    upper: float
    resamples: int
    alpha: float

    @property
    def excludes_zero_above(self) -> bool:
        return self.lower > 0.0

    @property
    def excludes_zero_below(self) -> bool:
        return self.upper < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "resamples": self.resamples,
            "alpha": self.alpha,
        }


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """One challenger measured against one baseline on the primary metric."""

    challenger: str
    baseline: str
    metric: str
    interval: Interval
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "challenger": self.challenger,
            "baseline": self.baseline,
            "metric": self.metric,
            "delta": self.interval.as_dict(),
            "outcome": self.outcome,
        }


#: Metrics the bootstrap computes per replicate. Deliberately few: each one costs
#: 2,000 recomputations per model, and the promotion policy reads only these.
def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision, computed directly to stay fast inside the loop.

    Tied scores are collapsed before precision and recall are read off, which
    is not a nicety here. Isotonic calibration is a step function: it maps this
    test set's 13,376 predictions onto roughly a hundred distinct values, so
    almost every row is tied with hundreds of others. Walking the sorted array
    row by row silently assumes an ordering inside each tie group, and because
    ``argsort`` is stable that assumption is systematically favourable -
    measured on this run it inflated average precision by +0.007 to +0.009 for
    every calibrated model, while the one uncalibrated model, whose scores are
    nearly all distinct, was unaffected. Precision and recall are therefore
    evaluated once per distinct score, exactly as
    ``sklearn.metrics.average_precision_score`` defines them; a test asserts the
    two agree on heavily tied input.
    """
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]
    true_positive = np.cumsum(labels)
    seen = np.arange(1, len(labels) + 1)

    # The last index of each run of equal scores: the only points at which a
    # threshold can actually be placed.
    boundaries = np.r_[np.nonzero(np.diff(scores))[0], len(scores) - 1]
    precision = true_positive[boundaries] / seen[boundaries]
    recall = true_positive[boundaries] / labels.sum()
    gained = np.diff(recall, prepend=0.0)
    return float(np.sum(gained * precision))


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


BOOTSTRAP_METRICS: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc,
    "pr_auc": _pr_auc,
    "brier_score": _brier,
}


def resample_indices(
    n_rows: int, *, resamples: int, seed: int
) -> np.ndarray:
    """One index matrix, drawn once and shared by every model.

    Shape is (resamples, n_rows). Materialising it makes the pairing explicit
    and auditable: every model is scored on row set ``i`` for replicate ``i``,
    and a test asserts exactly that.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=(resamples, n_rows), dtype=np.int64)


def bootstrap_metrics(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    alpha: float = protocol.BOOTSTRAP_ALPHA,
    seed: int = protocol.BOOTSTRAP_SEED,
    metrics: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> tuple[dict[str, dict[str, Interval]], dict[str, dict[str, np.ndarray]]]:
    """Per-model intervals, plus the raw replicate values for pairing.

    Returns ``(intervals, replicates)`` where ``replicates[model][metric]`` is
    the array of per-replicate values. Deltas are computed from those arrays so
    a comparison never re-draws its own resample.
    """
    selected = dict(metrics or BOOTSTRAP_METRICS)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    index_matrix = resample_indices(len(y_true), resamples=resamples, seed=seed)

    replicates: dict[str, dict[str, np.ndarray]] = {
        name: {metric: np.empty(resamples) for metric in selected}
        for name in predictions
    }

    for replicate, rows in enumerate(index_matrix):
        truth = y_true[rows]
        for name, proba in predictions.items():
            scores = np.asarray(proba, dtype=np.float64)[rows]
            for metric, function in selected.items():
                replicates[name][metric][replicate] = function(truth, scores)

    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    intervals: dict[str, dict[str, Interval]] = {}
    for name, proba in predictions.items():
        intervals[name] = {}
        for metric, function in selected.items():
            values = replicates[name][metric]
            finite = values[np.isfinite(values)]
            intervals[name][metric] = Interval(
                point=float(function(y_true, np.asarray(proba, dtype=np.float64))),
                lower=float(np.percentile(finite, lower_percentile)) if finite.size else float("nan"),
                upper=float(np.percentile(finite, upper_percentile)) if finite.size else float("nan"),
                resamples=resamples,
                alpha=alpha,
            )
    return intervals, replicates


def paired_delta(
    replicates: Mapping[str, Mapping[str, np.ndarray]],
    challenger: str,
    baseline: str,
    metric: str,
    *,
    alpha: float = protocol.BOOTSTRAP_ALPHA,
) -> Interval:
    """Interval for (challenger - baseline) within each shared replicate."""
    differences = np.asarray(replicates[challenger][metric]) - np.asarray(
        replicates[baseline][metric]
    )
    finite = differences[np.isfinite(differences)]
    return Interval(
        point=float(np.mean(finite)) if finite.size else float("nan"),
        lower=float(np.percentile(finite, 100 * (alpha / 2))) if finite.size else float("nan"),
        upper=float(np.percentile(finite, 100 * (1 - alpha / 2))) if finite.size else float("nan"),
        resamples=len(differences),
        alpha=alpha,
    )


def interpret(interval: Interval, *, higher_is_better: bool = True) -> str:
    """One of the three outcomes the protocol allows.

    An interval containing zero is INCONCLUSIVE - not "promising", not "a slight
    edge". That restriction is the point: it is the sentence that stops a
    difference the data cannot resolve from being reported as a finding.
    """
    if interval.excludes_zero_above:
        return "CLEAR IMPROVEMENT" if higher_is_better else "CLEAR REGRESSION"
    if interval.excludes_zero_below:
        return "CLEAR REGRESSION" if higher_is_better else "CLEAR IMPROVEMENT"
    return "INCONCLUSIVE"


def compare_all(
    replicates: Mapping[str, Mapping[str, np.ndarray]],
    pairs: Sequence[tuple[str, str]],
    *,
    metric: str = protocol.PRIMARY_METRIC,
    higher_is_better: bool = True,
) -> list[PairedComparison]:
    """Every prespecified pairwise comparison, in one pass."""
    results = []
    for challenger, baseline in pairs:
        interval = paired_delta(replicates, challenger, baseline, metric)
        results.append(
            PairedComparison(
                challenger=challenger,
                baseline=baseline,
                metric=metric,
                interval=interval,
                outcome=interpret(interval, higher_is_better=higher_is_better),
            )
        )
    return results


def default_pairs() -> tuple[tuple[str, str], ...]:
    """The comparisons the protocol names.

    Each deep model against each classical baseline - the question Track K was
    opened to answer - plus every deep model against every other, which is what
    distinguishes "no architecture helps" from "one architecture helps and the
    others do not". Derived from the family roster rather than listed by hand,
    so adding a challenger cannot silently omit its comparisons.

    Deep-vs-deep pairs are ordered by the roster, so each unordered pair appears
    once and the direction of every reported delta is predictable.
    """
    pairs = [
        (deep, classical)
        for deep in protocol.DEEP_FAMILIES
        for classical in protocol.CLASSICAL_FAMILIES
    ]
    deep = protocol.DEEP_FAMILIES
    pairs.extend(
        (deep[later], deep[earlier])
        for earlier in range(len(deep))
        for later in range(earlier + 1, len(deep))
    )
    return tuple(pairs)


def promotion_verdict(
    *,
    primary_delta: Interval,
    ece_delta: float,
    recall_delta: float,
    latency_multiple: float,
    policy: protocol.PromotionPolicy = protocol.PROMOTION_POLICY,
) -> tuple[str, list[str]]:
    """Apply the frozen promotion policy. Returns (verdict, reasons).

    Every gate is evaluated and reported, not short-circuited, so the record
    shows which ones a challenger cleared as well as which it failed.
    """
    reasons: list[str] = []

    clears_primary = primary_delta.lower > policy.min_primary_delta
    reasons.append(
        f"primary delta 95% CI lower bound {primary_delta.lower:+.5f} "
        f"{'>' if clears_primary else '<='} required {policy.min_primary_delta:+.5f}"
    )

    clears_calibration = ece_delta <= policy.max_ece_regression
    reasons.append(
        f"ECE change {ece_delta:+.5f} "
        f"{'<=' if clears_calibration else '>'} allowed {policy.max_ece_regression:+.5f}"
    )

    clears_recall = recall_delta >= -policy.max_recall_regression
    reasons.append(
        f"recall change {recall_delta:+.5f} "
        f"{'>=' if clears_recall else '<'} allowed {-policy.max_recall_regression:+.5f}"
    )

    clears_latency = latency_multiple <= policy.max_latency_multiple
    reasons.append(
        f"latency {latency_multiple:.2f}x baseline "
        f"{'<=' if clears_latency else '>'} allowed {policy.max_latency_multiple:.2f}x"
    )

    if clears_primary and clears_calibration and clears_recall and clears_latency:
        return "PROMOTE", reasons
    # A challenger whose primary interval sits entirely below the required
    # margin has been measured and found wanting; one whose interval straddles
    # it has not been resolved by the data.
    if primary_delta.upper <= policy.min_primary_delta:
        return "REJECT", reasons
    return "INCONCLUSIVE", reasons
