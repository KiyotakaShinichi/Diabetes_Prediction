"""Do thirty different algorithms make thirty different mistakes?

Track K found that four models spanning linear, boosted and attention-based
designs produced *the same* errors: false positives with the classic risk
markers and no diagnosis, false negatives with the diagnosis and no markers,
and 15-19% of the test set where every model was near chance. It concluded the
limit was the ten-feature information set rather than the model class.

Four models is a small sample from which to claim that. Thirty is better, and
this module is the test. If a nearest-centroid classifier, an RBF kernel, a
gradient-boosted ensemble and a transformer all fail on the same patients, then
the shared factor is the data, and no amount of further architecture search will
help. If instead the families disagree substantially, the picture is different
and there is something for a future ensemble to exploit.

Three measurements, each answering a different question:

* **Error overlap** - of the rows model A gets wrong, what fraction does model B
  also get wrong? Directly tests the information-bottleneck hypothesis.
* **Score correlation** - do the models rank patients the same way, whatever
  their errors? Two models can agree on every label and still rank differently.
* **Family diversity** - is disagreement larger *between* families than *within*
  them? If not, then "thirty models" is not thirty independent pieces of
  evidence, and reporting them as if it were would overstate the case.

Nothing here fits anything. It reads the per-row predictions a completed run
already wrote, so it costs seconds and can be re-run against any past run.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from research.model_zoo.contracts import Family
from research.model_zoo.registry import REGISTRY


@dataclass(frozen=True, slots=True)
class OverlapPair:
    """How two models' mistakes relate."""

    left: str
    right: str
    #: Rows both models got wrong, over rows at least one got wrong. Jaccard,
    #: so it is symmetric and insensitive to one model simply erring more.
    jaccard: float
    #: Of the rows the left model got wrong, the share the right also missed.
    left_errors_shared: float
    right_errors_shared: float
    disagreement_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "jaccard": self.jaccard,
            "left_errors_shared": self.left_errors_shared,
            "right_errors_shared": self.right_errors_shared,
            "disagreement_rate": self.disagreement_rate,
        }


def error_overlap(
    y_true: np.ndarray, predictions: dict[str, np.ndarray]
) -> list[OverlapPair]:
    """Pairwise error agreement for every pair of models."""
    truth = np.asarray(y_true).astype(int)
    wrong = {name: (np.asarray(p).astype(int) != truth) for name, p in predictions.items()}

    pairs = []
    for left, right in combinations(sorted(wrong), 2):
        a, b = wrong[left], wrong[right]
        both = float((a & b).sum())
        either = float((a | b).sum())
        pairs.append(
            OverlapPair(
                left=left,
                right=right,
                jaccard=both / either if either else 1.0,
                left_errors_shared=both / float(a.sum()) if a.sum() else float("nan"),
                right_errors_shared=both / float(b.sum()) if b.sum() else float("nan"),
                disagreement_rate=float(
                    (np.asarray(predictions[left]) != np.asarray(predictions[right])).mean()
                ),
            )
        )
    return pairs


def score_correlation(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Spearman rank correlation between every pair of models' scores.

    Rank rather than Pearson: the zoo's scores live on different scales - a
    probability, a vote share, a signed margin squashed through a logistic - and
    the question is whether they order patients alike, not whether the numbers
    are linearly related.
    """
    from scipy.stats import rankdata

    names = sorted(scores)
    ranked = {name: rankdata(np.asarray(scores[name], dtype=float)) for name in names}
    matrix: dict[str, dict[str, float]] = {}
    for left in names:
        matrix[left] = {}
        for right in names:
            if left == right:
                matrix[left][right] = 1.0
                continue
            matrix[left][right] = float(np.corrcoef(ranked[left], ranked[right])[0, 1])
    return matrix


def family_diversity(
    y_true: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Is disagreement larger between families than within them?

    The comparison that decides whether the zoo's breadth is real. If the
    between-family mean disagreement is no higher than the within-family mean,
    then the families are not exploring meaningfully different hypotheses on
    this problem, and thirty models are close to one model counted thirty times.
    """
    families = {
        model_id: REGISTRY.get(model_id).family
        for model_id in predictions
        if model_id in REGISTRY
    }
    within: list[float] = []
    between: list[float] = []
    for pair in error_overlap(y_true, predictions):
        left_family = families.get(pair.left)
        right_family = families.get(pair.right)
        if left_family is None or right_family is None:
            continue
        (within if left_family is right_family else between).append(pair.disagreement_rate)

    by_family: dict[str, float] = {}
    for family in Family:
        members = [m for m, f in families.items() if f is family]
        rates = [
            pair.disagreement_rate
            for pair in error_overlap(y_true, {m: predictions[m] for m in members})
        ] if len(members) > 1 else []
        if rates:
            by_family[family.value] = float(np.mean(rates))

    return {
        "mean_within_family_disagreement": float(np.mean(within)) if within else None,
        "mean_between_family_disagreement": float(np.mean(between)) if between else None,
        "within_family_by_family": by_family,
        "interpretation": _diversity_reading(within, between),
    }


def _diversity_reading(within: list[float], between: list[float]) -> str:
    if not within or not between:
        return "not enough models to compare within- and between-family diversity"
    inner, outer = float(np.mean(within)), float(np.mean(between))
    gap = outer - inner
    if gap > 0.02:
        return (
            f"between-family disagreement ({outer:.4f}) exceeds within-family "
            f"({inner:.4f}) by {gap:.4f}: the families do explore different "
            "hypotheses, and an ensemble across them has something to work with"
        )
    return (
        f"between-family disagreement ({outer:.4f}) is close to within-family "
        f"({inner:.4f}), a gap of {gap:+.4f}: the algorithmic families are "
        "converging on the same predictions, which supports the view that the "
        "feature set rather than the model class is the binding constraint"
    )


def hardest_rows(
    y_true: np.ndarray, predictions: dict[str, np.ndarray], *, top: int = 20
) -> dict[str, Any]:
    """Rows most models get wrong - the population Track K identified.

    If the same patients defeat every family, that is the information
    bottleneck made concrete rather than inferred.
    """
    truth = np.asarray(y_true).astype(int)
    stacked = np.vstack([np.asarray(p).astype(int) for p in predictions.values()])
    wrong_count = (stacked != truth).sum(axis=0)
    n_models = len(predictions)

    order = np.argsort(-wrong_count)[:top]
    return {
        "model_count": n_models,
        "rows_every_model_got_wrong": int((wrong_count == n_models).sum()),
        "rows_every_model_got_right": int((wrong_count == 0).sum()),
        "share_wrong_by_at_least_half": float((wrong_count >= n_models / 2).mean()),
        "hardest_row_indices": [int(i) for i in order],
        "hardest_row_error_counts": [int(wrong_count[i]) for i in order],
    }


def _optional(value: float | None) -> str:
    """Format a statistic that may be undefined.

    Within- and between-family disagreement are both undefined for some model
    sets - a run of three models from three different families has no
    within-family pair at all. Formatting None with a float spec is a
    TypeError, which would crash the summary of a perfectly valid run.
    """
    return f"{value:.4f}" if value is not None else "undefined (too few pairs)"


def summarise(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> str:
    """A short readable account of how much the zoo actually disagrees."""
    pairs = error_overlap(y_true, predictions)
    diversity = family_diversity(y_true, predictions)
    hardest = hardest_rows(y_true, predictions)

    jaccards = [p.jaccard for p in pairs]
    lines = [
        f"Error overlap across {len(predictions)} models "
        f"({len(pairs)} pairs, {len(y_true):,} test rows)",
        "",
        f"  mean pairwise error Jaccard      {np.mean(jaccards):.4f}",
        f"  min / max                        {np.min(jaccards):.4f} / {np.max(jaccards):.4f}",
        f"  rows every model got wrong       {hardest['rows_every_model_got_wrong']:,}",
        f"  rows every model got right       {hardest['rows_every_model_got_right']:,}",
        f"  share wrong by at least half     {hardest['share_wrong_by_at_least_half']:.4f}",
        "",
        f"  within-family disagreement       "
        f"{_optional(diversity['mean_within_family_disagreement'])}",
        f"  between-family disagreement      "
        f"{_optional(diversity['mean_between_family_disagreement'])}",
        "",
        f"  {diversity['interpretation']}",
    ]

    if scores:
        matrix = score_correlation(scores)
        names = sorted(matrix)
        off_diagonal = [
            matrix[a][b] for a, b in combinations(names, 2)
        ]
        lines += [
            "",
            f"Score rank correlation across {len(names)} models with a ranking score",
            f"  mean                             {np.mean(off_diagonal):.4f}",
            f"  min / max                        "
            f"{np.min(off_diagonal):.4f} / {np.max(off_diagonal):.4f}",
        ]
        least = min(combinations(names, 2), key=lambda p: matrix[p[0]][p[1]])
        lines.append(
            f"  least correlated pair            {least[0]} vs {least[1]} "
            f"({matrix[least[0]][least[1]]:.4f})"
        )
    return "\n".join(lines)
