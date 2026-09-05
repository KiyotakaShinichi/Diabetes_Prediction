"""Does the ranking actually describe what the model uses?

Agreement between methods is not correctness. Every method in this package
could rank BMI first, agree with each other perfectly, and all be wrong
together - they share assumptions, and two methods built on the same assumption
will agree about the same mistake. Faithfulness asks the separate question: if
this ranking is right, then removing the features it calls important should
damage the model more than removing the ones it calls unimportant. That is
checkable without knowing the true answer.

**Everything here measures per-row score shift, not the mean score, and that is
not a stylistic choice.** The obvious implementation - watch the average
predicted probability move as features are ablated - barely registers on a
balanced dataset, because ablation pushes rows in opposite directions toward the
same central prediction and they cancel in the average.

Measured on a world where one feature generates the label, a random forest's
mean score crept from 0.523 to 0.597 across all ten deletions. Removing the true
driver first moved that mean by 0.016; removing an inert feature first moved it
by 0.004. A ratio of about four, on movements small enough to be swamped by
anything. The mean absolute per-row shift over the same two runs separates them
by a factor of thirty-one: deleting the driver alone carries it to 0.310 of a
possible 0.325 - 95% of the total available damage in one step - while deleting
the driver last leaves it at 0.010. Taking the absolute value per row before
averaging is what turns a statistic that technically responds into one that
actually discriminates.

Four measurements, and the fourth is the one that keeps the other three honest.

**Deletion** replaces features with their training baseline in ranked order,
most important first, and watches how far the model's opinion of each patient
moves. A faithful ranking moves it fast, so a *higher* deletion area is better.
**Insertion** starts from the fully ablated row and restores features in the
same order, watching the model return to its original opinion. A faithful
ranking returns fast, so a *lower* insertion area is better. They are not
redundant: deletion is dominated by what happens while the model still has most
of its inputs, insertion by what happens when it has almost none, and a ranking
can look good on one and poor on the other.

**Comprehensiveness and sufficiency** are the same idea at a fixed k, which is
what a reader actually sees - a chart of the top three. Comprehensiveness is how
far the score moves when the top k are removed together; sufficiency is how far
it moves when *only* the top k are kept, so near zero is the good result there.

**The random baseline** is what makes any of it interpretable. A deletion curve
that climbs to its maximum looks impressive until the same curve for a shuffled
ranking climbs almost as fast, which happens whenever a model is dominated by
one feature: delete any three of ten and there is a good chance the important
one went with them. So every score here is reported against random rankings over
the same rows, and the number that means something is the *gap*. A method that
cannot beat shuffling has not been shown to be faithful, however clean its curve.

Two things this module does not claim. It cannot distinguish a faithful ranking
from one that is merely correlated with a faithful one; and every deletion moves
the input off the data manifold, so part of every score is the model reacting to
an implausible patient rather than to a missing feature. Both limits are stated
on the method cards, and neither is fixable by measuring harder.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#: Random rankings drawn per comparison. Five is enough to place a method
#: against the shuffled distribution without the baseline costing more than the
#: measurement it contextualises.
RANDOM_RANKING_DRAWS: int = 5


def _scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.decision_scores(X), dtype=float)


def _shift(model: Any, X: pd.DataFrame, reference: np.ndarray) -> float:
    """Mean per-row distance between the model's current and original opinion.

    Absolute value taken per row before averaging. Without it, a row the
    ablation pushed up cancels a row it pushed down and the statistic reports
    that nothing happened.
    """
    return float(np.mean(np.abs(_scores(model, X) - reference)))


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    """One ranking's faithfulness, always beside what shuffling would score.

    Sign conventions, stated once: a **higher** deletion area is better (the
    ranking damaged the model faster) and a **lower** insertion area is better
    (the model recovered its original opinion sooner). Both gaps are defined so
    that positive means "beat the shuffled control".
    """

    deletion_auc: float
    insertion_auc: float
    random_deletion_auc: float
    random_insertion_auc: float
    comprehensiveness: float
    sufficiency: float
    full_ablation_shift: float
    rows: int
    top_k: int

    @property
    def deletion_gap(self) -> float:
        """How much faster this ranking damages the model than a shuffle does."""
        return self.deletion_auc - self.random_deletion_auc

    @property
    def insertion_gap(self) -> float:
        """How much sooner this ranking restores the model than a shuffle does."""
        return self.random_insertion_auc - self.insertion_auc

    @property
    def beats_random(self) -> bool:
        return self.deletion_gap > 0.0 and self.insertion_gap > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "deletion_auc": self.deletion_auc,
            "insertion_auc": self.insertion_auc,
            "random_deletion_auc": self.random_deletion_auc,
            "random_insertion_auc": self.random_insertion_auc,
            "deletion_gap": self.deletion_gap,
            "insertion_gap": self.insertion_gap,
            "comprehensiveness": self.comprehensiveness,
            "sufficiency": self.sufficiency,
            "full_ablation_shift": self.full_ablation_shift,
            "beats_random": self.beats_random,
            "rows": self.rows,
            "top_k": self.top_k,
        }


def deletion_curve(
    model: Any, X: pd.DataFrame, ranking: Sequence[str], baseline: pd.Series
) -> list[float]:
    """How far the model's per-row opinion has moved after removing the top k.

    Index k is the mean absolute score shift with the top k features replaced by
    their training baseline, so the curve starts at exactly zero and ends at the
    full-ablation distance - the same value for every ranking, which is what
    makes two curves comparable. A faithful ranking gets there sooner.
    """
    reference = _scores(model, X)
    remaining = X.copy()
    curve = [0.0]

    for feature in ranking:
        remaining[feature] = baseline[feature]
        curve.append(_shift(model, remaining, reference))
    return curve


def insertion_curve(
    model: Any, X: pd.DataFrame, ranking: Sequence[str], baseline: pd.Series
) -> list[float]:
    """How far the model's opinion still is from the original after restoring k.

    The mirror of deletion, and not a restatement of it: this one spends most of
    its length describing a model that has almost no information, which is
    exactly where a ranking's leading choices show their worth. Starts at the
    full-ablation distance and ends at zero.
    """
    reference = _scores(model, X)
    restored = X.copy()
    for feature in X.columns:
        restored[feature] = baseline[feature]

    curve = [_shift(model, restored, reference)]
    for feature in ranking:
        restored[feature] = X[feature].to_numpy()
        curve.append(_shift(model, restored, reference))
    return curve


def curve_auc(curve: Sequence[float]) -> float:
    """Normalised area under a deletion or insertion curve.

    Trapezoidal, divided by the number of steps, so curves over different
    feature counts stay comparable.
    """
    values = np.asarray(curve, dtype=float)
    if values.size < 2:
        raise ValueError("a curve needs at least two points")
    return float(np.trapezoid(values) / (values.size - 1))


def comprehensiveness(
    model: Any, X: pd.DataFrame, ranking: Sequence[str], baseline: pd.Series, k: int
) -> float:
    """How far the score moves when the top k features are removed together.

    Removed together rather than one at a time, which is what makes this
    different from the head of the deletion curve: two features that only matter
    jointly are both gone here, and a one-at-a-time measurement would miss it.
    Higher is better.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ablated = X.copy()
    for feature in ranking[:k]:
        ablated[feature] = baseline[feature]
    return _shift(model, ablated, _scores(model, X))


def sufficiency(
    model: Any, X: pd.DataFrame, ranking: Sequence[str], baseline: pd.Series, k: int
) -> float:
    """How far the score moves when *only* the top k features are kept.

    Near zero is the good result and it means the shortlist carried the model on
    its own. Reported alongside comprehensiveness because the two can disagree:
    a ranking whose top three are individually redundant scores well there and
    poorly here.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    kept = set(ranking[:k])
    reduced = X.copy()
    for feature in X.columns:
        if feature not in kept:
            reduced[feature] = baseline[feature]
    return _shift(model, reduced, _scores(model, X))


def evaluate(
    model: Any,
    X: pd.DataFrame,
    ranking: Sequence[str],
    baseline: pd.Series,
    *,
    top_k: int = 3,
    random_draws: int = RANDOM_RANKING_DRAWS,
    seed: int = 0,
) -> FaithfulnessResult:
    """Score one ranking, always beside what a shuffled ranking would score.

    The shuffled comparison is not optional decoration. On a model dominated by
    a single feature, almost any ranking eventually produces a dramatic curve,
    because deleting features one after another eventually deletes the one that
    matters. Only the gap between the ranking and the shuffle says whether the
    *order* carried information.
    """
    rng = np.random.default_rng(seed)
    features = list(ranking)

    deletion = deletion_curve(model, X, features, baseline)
    insertion = insertion_curve(model, X, features, baseline)

    random_deletions: list[float] = []
    random_insertions: list[float] = []
    for _ in range(random_draws):
        shuffled = list(rng.permutation(features))
        random_deletions.append(curve_auc(deletion_curve(model, X, shuffled, baseline)))
        random_insertions.append(curve_auc(insertion_curve(model, X, shuffled, baseline)))

    return FaithfulnessResult(
        deletion_auc=curve_auc(deletion),
        insertion_auc=curve_auc(insertion),
        random_deletion_auc=float(np.mean(random_deletions)),
        random_insertion_auc=float(np.mean(random_insertions)),
        comprehensiveness=comprehensiveness(model, X, features, baseline, top_k),
        sufficiency=sufficiency(model, X, features, baseline, top_k),
        full_ablation_shift=float(deletion[-1]),
        rows=len(X),
        top_k=top_k,
    )


def summarise(results: dict[str, FaithfulnessResult]) -> dict[str, Any]:
    """Aggregate several rankings, naming any that failed to beat shuffling."""
    if not results:
        return {"evaluated": 0, "beat_random": [], "failed_random": [], "best": None}

    beat = sorted(name for name, r in results.items() if r.beats_random)
    failed = sorted(name for name, r in results.items() if not r.beats_random)
    best = max(results.items(), key=lambda item: item[1].deletion_gap)

    return {
        "evaluated": len(results),
        "beat_random": beat,
        "failed_random": failed,
        "best": {"name": best[0], "deletion_gap": best[1].deletion_gap},
        "mean_deletion_gap": float(np.mean([r.deletion_gap for r in results.values()])),
    }
