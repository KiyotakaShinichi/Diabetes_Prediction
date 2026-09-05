"""Does an explanation survive a small change to the data it was computed from?

An explanation that reorders itself when a patient's BMI moves by a tenth of a
standard deviation is not a finding about the model; it is a finding about the
noise. Track M cannot report which features a family relies on without first
establishing that the answer is stable enough to be worth reporting.

Two kinds of instability are separated here, because they have different causes
and different remedies.

**Perturbation stability** asks what happens when the *data* moves. One rule is
applied to every feature: add Gaussian noise scaled to that feature's training
standard deviation, then coerce the result back into the served contract -
rounding where the contract says integer, clipping to the declared range. That
single rule covers the whole feature set without a special case, and it gives
binary features a principled treatment for free: a flag flips exactly when the
noise crosses a half, so at a magnitude of 0.5 training deviations a binary
feature flips about 5% of the time and at 1.0 about 32%. Perturbation strength
is therefore always quoted in training deviations, never in raw units, so a
number means the same thing for BMI and for PhysHlth.

**Seed stability** asks what happens when nothing moves but the method's own
randomness. It applies to permutation importance and to nothing else in this
package, and `seed_stability` refuses to run on a method the registry declares
deterministic. That refusal is the point: sweeping seeds over a deterministic
method would produce a variance of exactly zero and put it in a table beside a
real one, implying the two had been measured the same way.

**Every scale comes from TRAINING rows.** A perturbation sized by the test set's
spread would smuggle the evaluation distribution into the explanation of a model
being judged on it, and the leak would be invisible - the perturbations would
simply look slightly better calibrated. `fit_scale` records how many rows it saw
so a scale can be audited back to a partition.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_core import feature_contract
from research.xai.agreement import spearman, top_k_overlap
from research.xai.contracts import Determinism, XaiError, rank_features

#: Perturbation strengths, in training standard deviations. Chosen to span from
#: "a rounding error" to "a different patient": 0.05 should change nothing, and
#: if 1.0 changes nothing either then the explanation is not responding to the
#: data at all, which is its own finding.
STABILITY_MAGNITUDES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)

#: Perturbed replicates per magnitude. Enough to see a mean apart from its
#: spread without turning a diagnostic into the run's dominant cost.
STABILITY_REPEATS: int = 5


class DeterminismError(XaiError):
    """A seed sweep was requested for a method that has no randomness."""


@dataclass(frozen=True, slots=True)
class PerturbationScale:
    """Per-feature noise scale, fitted on training rows and nothing else."""

    feature_names: tuple[str, ...]
    deviations: tuple[float, ...]
    fitted_rows: int
    source: str = "training partition"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "deviations": [float(d) for d in self.deviations],
            "fitted_rows": self.fitted_rows,
            "source": self.source,
        }


def fit_scale(X: pd.DataFrame) -> PerturbationScale:
    """Measure each feature's spread on the rows handed in.

    Call this with TRAINING rows. Sizing a perturbation by the spread of the
    partition a model is evaluated on would leak that distribution into the
    explanation of the model, and the leak would show up as nothing more
    suspicious than unusually well-behaved perturbations.

    A degenerate column - one constant value in this partition - gets a
    deviation of zero and is therefore never perturbed, which is correct: there
    is no observed variation to imitate.
    """
    deviations = tuple(float(X[name].std(ddof=0)) for name in X.columns)
    return PerturbationScale(
        feature_names=tuple(str(name) for name in X.columns),
        deviations=tuple(0.0 if not np.isfinite(d) else d for d in deviations),
        fitted_rows=len(X),
    )


def perturb(
    X: pd.DataFrame, scale: PerturbationScale, magnitude: float, *, seed: int
) -> pd.DataFrame:
    """Add noise of ``magnitude`` training deviations, then re-enter the contract.

    The coercion step is what keeps this honest. Unconstrained Gaussian noise
    would produce a BMI of 91 and an Education level of 2.7 - inputs no model in
    the zoo was ever fitted on and no served request could contain - and the
    resulting instability would be a measurement of extrapolation rather than of
    the explanation. Rounding to integers and clipping to the declared range
    keeps every perturbed row inside the space the contract describes.

    A consequence worth naming: coercion makes the perturbation slightly weaker
    than requested, most of all for features near the edge of their range, where
    clipping can only push one way.
    """
    if magnitude < 0:
        raise ValueError("perturbation magnitude cannot be negative")
    if tuple(str(c) for c in X.columns) != scale.feature_names:
        raise ValueError(
            f"scale fitted on {scale.feature_names}, asked to perturb "
            f"{tuple(str(c) for c in X.columns)}"
        )

    rng = np.random.default_rng(seed)
    perturbed = X.copy()

    for position, name in enumerate(scale.feature_names):
        deviation = scale.deviations[position]
        if deviation <= 0 or magnitude == 0:
            continue
        spec = feature_contract.spec_for(name)
        values = perturbed[name].to_numpy(dtype=float)
        values = values + rng.normal(0.0, magnitude * deviation, len(values))
        if spec.dtype is int:
            values = np.rint(values)
        perturbed[name] = np.clip(values, spec.minimum, spec.maximum)

    return perturbed[list(X.columns)]


@dataclass(frozen=True, slots=True)
class StabilityPoint:
    """How much an explanation moved at one perturbation strength."""

    magnitude: float
    mean_spearman: float
    min_spearman: float
    top_1_retention: float
    mean_top_3_overlap: float
    replicates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "magnitude": self.magnitude,
            "mean_spearman": self.mean_spearman,
            "min_spearman": self.min_spearman,
            "top_1_retention": self.top_1_retention,
            "mean_top_3_overlap": self.mean_top_3_overlap,
            "replicates": self.replicates,
        }


def stability_curve(
    explain: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    scale: PerturbationScale,
    *,
    magnitudes: Sequence[float] = STABILITY_MAGNITUDES,
    repeats: int = STABILITY_REPEATS,
    seed: int = 0,
) -> list[StabilityPoint]:
    """Explanation agreement with itself as the data is perturbed harder.

    ``explain`` takes a frame and returns one attribution per feature; the model
    is *not* refitted. That is deliberate and it is the narrower of the two
    possible questions. This measures whether a fixed model's explanation is
    robust to the data it is explained on - not whether a model refitted on
    perturbed data would find the same structure, which is a question about
    training variance and belongs to a different study.

    Both the correlation and the top-1 retention are reported, because they can
    diverge sharply: an explanation whose leading feature never changes while its
    tail reshuffles is stable in the way that matters and unstable in the way a
    correlation measures.
    """
    features = tuple(str(column) for column in X.columns)
    reference = np.asarray(explain(X), dtype=float)
    reference_ranking = rank_features(features, reference)

    points: list[StabilityPoint] = []
    for magnitude in magnitudes:
        correlations: list[float] = []
        retained: list[bool] = []
        overlaps: list[float] = []

        for replicate in range(repeats):
            noisy = perturb(X, scale, magnitude, seed=seed + replicate)
            attributions = np.asarray(explain(noisy), dtype=float)
            ranking = rank_features(features, attributions)

            correlations.append(spearman(reference, attributions))
            retained.append(ranking[0] == reference_ranking[0])
            overlaps.append(top_k_overlap(reference_ranking, ranking, 3))

        points.append(
            StabilityPoint(
                magnitude=float(magnitude),
                mean_spearman=float(np.mean(correlations)),
                min_spearman=float(np.min(correlations)),
                top_1_retention=float(np.mean(retained)),
                mean_top_3_overlap=float(np.mean(overlaps)),
                replicates=repeats,
            )
        )
    return points


def seed_stability(
    explain: Callable[[int], np.ndarray],
    feature_names: tuple[str, ...],
    *,
    determinism: Determinism,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Spread of a stochastic method's answer across its own random seeds.

    Refuses a deterministic method rather than reporting zero variance for it.
    A row of exact zeros in a variance table reads as "measured and found
    stable", when the truth is that there was nothing to measure - and it would
    sit beside genuine measurements as though the two were comparable.
    """
    if determinism is not Determinism.STOCHASTIC:
        raise DeterminismError(
            "seed stability applies only to stochastic methods; running a seed "
            "sweep over a deterministic one would report a variance of zero as "
            "though it had been measured"
        )
    if len(seeds) < 2:
        raise ValueError("a seed sweep needs at least two seeds")

    attributions = [np.asarray(explain(seed), dtype=float) for seed in seeds]
    rankings = [rank_features(feature_names, values) for values in attributions]

    pairs = [
        spearman(attributions[i], attributions[j])
        for i in range(len(attributions))
        for j in range(i + 1, len(attributions))
    ]
    leaders = {ranking[0] for ranking in rankings}

    return {
        "seeds": list(seeds),
        "mean_spearman": float(np.mean(pairs)),
        "min_spearman": float(np.min(pairs)),
        "top_1_stable": len(leaders) == 1,
        "distinct_top_features": sorted(leaders),
        "per_feature_dispersion": {
            name: float(np.std([values[index] for values in attributions]))
            for index, name in enumerate(feature_names)
        },
    }


def summarise(points: Sequence[StabilityPoint]) -> dict[str, Any]:
    """Condense a stability curve, keeping the two readings separate.

    Reports the strongest perturbation at which the leading feature still always
    survives. That is the number a reader can act on - "this explanation's top
    feature holds up to a quarter of a standard deviation" - and it cannot be
    recovered from a mean correlation.
    """
    if not points:
        return {"measured": False, "points": [], "top_1_stable_through": None}

    stable_through = None
    for point in sorted(points, key=lambda p: p.magnitude):
        if point.top_1_retention < 1.0:
            break
        stable_through = point.magnitude

    return {
        "measured": True,
        "points": [point.as_dict() for point in points],
        "top_1_stable_through": stable_through,
        "weakest_mean_spearman": float(min(p.mean_spearman for p in points)),
    }
