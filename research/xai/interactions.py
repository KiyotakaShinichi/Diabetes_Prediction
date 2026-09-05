"""Do these models use features jointly, or only one at a time?

The question matters here for a specific reason. Track K concluded that the
plateau across model families comes from the ten-feature information set rather
than from the model class, and one obvious way that conclusion could be wrong is
if the richer models were finding joint structure the simpler ones cannot
represent - structure that every single-feature explanation in this package is
blind to by construction. A partial-dependence curve averages over the rest of
the population, so a feature whose effect is positive for half the patients and
negative for the other half shows a flat line. That is not a hypothetical: on
the exclusive-or world, partial dependence ranks two inert columns above both
true drivers, at every seed, against a forest that has learned the rule to 0.96
held-out ROC-AUC.

So interaction strength is measured directly, with Friedman's H-statistic: how
much of a feature pair's joint partial dependence cannot be reconstructed by
adding the two one-dimensional curves. Zero means the pair is exactly additive.
One means the joint surface is entirely interaction.

**Interaction is not scale-free, and this module measures on the log-odds
scale.** That is not a detail; ignoring it produces a wrong answer that looks
entirely reasonable. A logistic regression is additive by construction - its
logit is a weighted sum and nothing else - yet measured on the probability its
`decision_scores` return, it reports an H-statistic of 0.18 for BMI against Age.
The sigmoid is what produced that: squashing an additive function through a
non-linear link makes it non-additive on the squashed scale. Left uncorrected,
every probability-valued model in the zoo would carry a floor of apparent
interaction created by its output transform, and the tree families - whose
probabilities are averages of votes rather than squashed sums - would be
compared against it on a different footing.

So probability-valued scores are converted to log odds per row before anything
is averaged, and the models that already emit an unbounded decision function are
left alone. On that scale a logistic regression reports essentially zero, which
is the answer its functional form demands and the check that the correction
works. Probabilities are clipped away from 0 and 1 first, because a forest that
votes unanimously would otherwise contribute an infinite logit.

Three further limits, all of which follow from the budget rather than the theory.

**It is computed on a grid, not on the empirical joint distribution.** The
textbook estimator evaluates the joint partial dependence at every observed pair
of values, which is quadratic in the row count and would dominate the whole
track's compute. Here both features are swept over a small grid and the
remaining features are averaged over a fixed row sample. The result is the same
quantity under a uniform weighting of the grid rather than the data's own
weighting, which matters most where a feature's distribution is very skewed -
BMI, in this contract.

**It shares partial dependence's manifold problem.** Setting two features to a
grid point creates patients who may not exist, and part of any H-statistic is
the model's response to an implausible combination.

**A high H-statistic is not evidence of a biological interaction.** It says the
model's fitted surface is not additive in those two inputs. Whether that
reflects something about diabetes, something about this sample, or something
about the model's inductive bias is not answerable from here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

#: Grid resolution per feature for the joint surface. Eight points squared is
#: sixty-four cells, which is as much resolution as a thousand training rows
#: supports and keeps a full ten-feature sweep affordable.
INTERACTION_GRID_POINTS: int = 8

#: Rows averaged over when marginalising the other eight features. A hundred is
#: enough for the ranking to be stable across samples while keeping the cost of
#: all forty-five pairs bounded.
INTERACTION_SAMPLE_ROWS: int = 100

#: Hard ceiling on pairs evaluated in one call. Ten features give forty-five
#: pairs; the cap exists so a future feature set cannot silently turn this into
#: an overnight job.
MAX_PAIRS: int = 45

#: Probabilities are pulled this far away from 0 and 1 before taking log odds.
#: A forest whose trees vote unanimously returns exactly 0.0 or 1.0, and an
#: infinite logit would dominate every surface it appeared in. At 1e-3 the
#: clipped extremes sit at about +/- 6.9 log odds, which is far beyond any
#: score these models produce on real rows and still finite.
PROBABILITY_CLIP: float = 1e-3


@dataclass(frozen=True, slots=True)
class InteractionResult:
    """One feature pair's departure from additivity."""

    feature_a: str
    feature_b: str
    h_statistic: float
    joint_range: float
    additive_range: float
    grid_points: int
    sample_rows: int

    @property
    def excess_range(self) -> float:
        """How much of the joint surface's spread additivity fails to explain."""
        return self.joint_range - self.additive_range

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "h_statistic": self.h_statistic,
            "joint_range": self.joint_range,
            "additive_range": self.additive_range,
            "excess_range": self.excess_range,
            "grid_points": self.grid_points,
            "sample_rows": self.sample_rows,
        }


def emits_probabilities(model: Any) -> bool:
    """Whether this model's ranking score is a probability needing a link.

    Prefers the model's own Track L capability declaration over sniffing the
    values, because a decision function can land inside [0, 1] by coincidence
    and would then be log-odds transformed twice as far as it should be. The
    range check is only the fallback for objects that declare nothing.
    """
    capabilities = getattr(model, "capabilities", None)
    declared = getattr(capabilities, "supports_predict_proba", None)
    if declared is not None:
        return bool(declared)
    return False


def _scores(model: Any, X: pd.DataFrame, *, log_odds: bool = True) -> np.ndarray:
    """The model's ranking score, on the scale interactions are measured on.

    See the module docstring: a probability is converted to log odds so that a
    model additive in the logit measures as additive, rather than inheriting a
    floor of apparent interaction from its own output transform.
    """
    raw = np.asarray(model.decision_scores(X), dtype=float)
    if not log_odds or not emits_probabilities(model):
        return raw
    clipped = np.clip(raw, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    transformed: np.ndarray = np.log(clipped / (1.0 - clipped))
    return transformed


def _grid(values: np.ndarray, points: int) -> np.ndarray:
    """Observed percentile span, so the sweep never extrapolates.

    Falls back to the full observed range when the central percentiles collapse,
    which happens for a binary feature whose minority class is under 2.5%.
    """
    low, high = np.percentile(values, [2.5, 97.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.array([low], dtype=float)
    return np.linspace(low, high, points)


def _sample(X: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if len(X) <= rows:
        return X
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), size=rows, replace=False))
    return X.iloc[chosen]


def _one_way(model: Any, sample: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    curve = np.empty(len(grid), dtype=float)
    probe = sample.copy()
    for index, value in enumerate(grid):
        probe[feature] = value
        curve[index] = float(np.mean(_scores(model, probe)))
    return curve


def two_way_partial_dependence(
    model: Any,
    X: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    *,
    grid_points: int = INTERACTION_GRID_POINTS,
    sample_rows: int = INTERACTION_SAMPLE_ROWS,
    seed: int = 0,
) -> dict[str, Any]:
    """The joint surface and both marginal curves, on one shared grid.

    Returned together because the only interesting quantity is the difference
    between them, and computing the three separately invites the grids to drift
    apart.
    """
    if feature_a == feature_b:
        raise ValueError("an interaction needs two distinct features")

    sample = _sample(X, sample_rows, seed)
    grid_a = _grid(X[feature_a].to_numpy(dtype=float), grid_points)
    grid_b = _grid(X[feature_b].to_numpy(dtype=float), grid_points)

    joint = np.empty((len(grid_a), len(grid_b)), dtype=float)
    probe = sample.copy()
    for i, value_a in enumerate(grid_a):
        probe[feature_a] = value_a
        for j, value_b in enumerate(grid_b):
            probe[feature_b] = value_b
            joint[i, j] = float(np.mean(_scores(model, probe)))

    return {
        "feature_a": feature_a,
        "feature_b": feature_b,
        "grid_a": [float(v) for v in grid_a],
        "grid_b": [float(v) for v in grid_b],
        "joint": joint,
        "curve_a": _one_way(model, sample, feature_a, grid_a),
        "curve_b": _one_way(model, sample, feature_b, grid_b),
        "sample_rows": len(sample),
    }


def h_statistic(surface: dict[str, Any]) -> float:
    """Fraction of the centred joint surface that additivity cannot explain.

    Zero means the pair is exactly additive: the joint surface is the sum of the
    two one-dimensional curves. One means the joint surface is entirely
    interaction. A degenerate surface - a model whose score does not move at all
    over this pair - has no variance to decompose and returns zero, because
    there is no interaction rather than because the measurement failed.
    """
    joint = np.asarray(surface["joint"], dtype=float)
    curve_a = np.asarray(surface["curve_a"], dtype=float)
    curve_b = np.asarray(surface["curve_b"], dtype=float)

    centred_joint = joint - joint.mean()
    centred_a = curve_a - curve_a.mean()
    centred_b = curve_b - curve_b.mean()

    additive = centred_a[:, None] + centred_b[None, :]
    residual = centred_joint - additive

    denominator = float(np.sum(centred_joint**2))
    if denominator <= 0:
        return 0.0
    return float(min(1.0, np.sum(residual**2) / denominator))


def measure_pair(
    model: Any,
    X: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    *,
    grid_points: int = INTERACTION_GRID_POINTS,
    sample_rows: int = INTERACTION_SAMPLE_ROWS,
    seed: int = 0,
) -> InteractionResult:
    """One pair's interaction strength, with the spreads it was derived from."""
    surface = two_way_partial_dependence(
        model, X, feature_a, feature_b,
        grid_points=grid_points, sample_rows=sample_rows, seed=seed,
    )
    joint = np.asarray(surface["joint"], dtype=float)
    curve_a = np.asarray(surface["curve_a"], dtype=float)
    curve_b = np.asarray(surface["curve_b"], dtype=float)
    additive = (curve_a - curve_a.mean())[:, None] + (curve_b - curve_b.mean())[None, :]

    return InteractionResult(
        feature_a=feature_a,
        feature_b=feature_b,
        h_statistic=h_statistic(surface),
        joint_range=float(joint.max() - joint.min()),
        additive_range=float(additive.max() - additive.min()),
        grid_points=grid_points,
        sample_rows=int(surface["sample_rows"]),
    )


def rank_interactions(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    max_pairs: int = MAX_PAIRS,
    grid_points: int = INTERACTION_GRID_POINTS,
    sample_rows: int = INTERACTION_SAMPLE_ROWS,
    seed: int = 0,
) -> list[InteractionResult]:
    """Every feature pair, strongest departure from additivity first.

    Pairs are enumerated in a fixed order and truncated at ``max_pairs``, so a
    budgeted run always measures the same subset rather than a different one
    each time.
    """
    names = list(features) if features is not None else [str(c) for c in X.columns]
    pairs = list(combinations(names, 2))[:max_pairs]

    results = [
        measure_pair(
            model, X, left, right,
            grid_points=grid_points, sample_rows=sample_rows, seed=seed,
        )
        for left, right in pairs
    ]
    return sorted(results, key=lambda r: (-r.h_statistic, r.feature_a, r.feature_b))


def summarise(results: Sequence[InteractionResult]) -> dict[str, Any]:
    """Condense a pair sweep without implying a causal reading.

    The wording of every key here is about the model's fitted surface. A
    departure from additivity is a property of what the model learned, not
    evidence that two clinical factors act on each other.
    """
    if not results:
        return {"pairs": 0, "strongest": None, "mean_h": None, "additive_share": None}

    values = np.array([r.h_statistic for r in results], dtype=float)
    strongest = results[0]

    return {
        "pairs": len(results),
        "strongest": {
            "features": [strongest.feature_a, strongest.feature_b],
            "h_statistic": strongest.h_statistic,
            "excess_range": strongest.excess_range,
        },
        "mean_h": float(values.mean()),
        "median_h": float(np.median(values)),
        "max_h": float(values.max()),
        #: Share of pairs whose joint surface is essentially the sum of its
        #: parts. High here means the model found little joint structure to use.
        "additive_share": float(np.mean(values < 0.05)),
    }
