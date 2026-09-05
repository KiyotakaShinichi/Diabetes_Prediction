"""Do the rows every model gets wrong have a different explanatory structure?

Track K found that four models spanning linear, boosted and attention-based
designs failed on the same patients, and concluded the limit was the ten-feature
information set rather than the model class. Track L widened that to twenty-nine
algorithms and found the same rows defeating all of them. Both stopped at
*which* rows. This module asks the next question: when the models fail together,
are they failing while looking at the same features?

Two answers are informative and they point in opposite directions.

If the hard rows produce the **same** attribution profile as the easy ones, then
the models are applying a rule that simply does not decide these patients - the
information is absent, and the bottleneck conclusion holds with an explanatory
mechanism attached rather than merely a correlation of errors.

If the hard rows produce a **different** profile - attribution spread thinly
across many features rather than concentrated on a few - then the models are not
confidently wrong so much as undecided, and something in the input space is
pulling them apart. That would be a different finding, and it would be visible
here and nowhere else in the track.

**Nothing here is causal.** A difference between the two profiles is an
association between error and attribution pattern. It does not say the
attribution pattern caused the error, that the features named are wrong for
these patients, or that any action on those features would change an outcome.
The partition is defined by model errors, so it is selected on the very thing
being explained, and every statement drawn from it inherits that selection.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.xai.agreement import spearman

#: Share of models that must be wrong for a row to count as majority-hard.
#: Half, so the band means "more models failed than succeeded" rather than an
#: arbitrary cut chosen after seeing the distribution.
MAJORITY_THRESHOLD: float = 0.5


@dataclass(frozen=True, slots=True)
class DifficultyPartition:
    """Rows grouped by how many models got them wrong."""

    universally_wrong: tuple[int, ...]
    majority_wrong: tuple[int, ...]
    universally_right: tuple[int, ...]
    model_count: int
    rows: int

    @property
    def universally_wrong_share(self) -> float:
        return len(self.universally_wrong) / self.rows if self.rows else 0.0

    @property
    def majority_wrong_share(self) -> float:
        return len(self.majority_wrong) / self.rows if self.rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_count": self.model_count,
            "rows": self.rows,
            "universally_wrong": len(self.universally_wrong),
            "majority_wrong": len(self.majority_wrong),
            "universally_right": len(self.universally_right),
            "universally_wrong_share": self.universally_wrong_share,
            "majority_wrong_share": self.majority_wrong_share,
        }


def partition(
    y_true: Sequence[int] | np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    majority_threshold: float = MAJORITY_THRESHOLD,
) -> DifficultyPartition:
    """Group rows by how many of the zoo's models got them wrong.

    Reads hard labels that a completed run already wrote, so it costs nothing
    and can be re-run against any past run. ``majority_wrong`` includes the
    universally wrong rows: it is a band of increasing difficulty, not a set of
    disjoint buckets, and treating it as disjoint would make the two shares
    incomparable.
    """
    if not predictions:
        raise ValueError("a difficulty partition needs at least one model")

    truth = np.asarray(y_true).astype(int)
    stacked = np.vstack([np.asarray(p).astype(int) for p in predictions.values()])
    if stacked.shape[1] != truth.size:
        raise ValueError(
            f"predictions cover {stacked.shape[1]} rows but {truth.size} labels were given"
        )

    wrong = (stacked != truth).sum(axis=0)
    count = len(predictions)

    return DifficultyPartition(
        universally_wrong=tuple(int(i) for i in np.flatnonzero(wrong == count)),
        majority_wrong=tuple(
            int(i) for i in np.flatnonzero(wrong >= majority_threshold * count)
        ),
        universally_right=tuple(int(i) for i in np.flatnonzero(wrong == 0)),
        model_count=count,
        rows=int(truth.size),
    )


def attribution_profile(attributions: np.ndarray) -> np.ndarray:
    """Mean absolute attribution per feature, normalised to shares.

    Absolute before averaging, for the same reason the faithfulness curves take
    it per row: a feature pushing one patient up and another down would
    otherwise average to nothing and look unused.
    """
    values = np.abs(np.asarray(attributions, dtype=float))
    if values.ndim == 1:
        values = values[None, :]
    means = values.mean(axis=0)
    total = means.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(means.shape, 1.0 / max(means.size, 1))
    shares: np.ndarray = means / total
    return shares


def concentration(shares: np.ndarray) -> float:
    """How much of the attribution sits on few features, from 0 to 1.

    One minus the normalised entropy of the share vector, so a model resting
    everything on one feature scores 1 and one spreading evenly scores 0. This
    is the statistic that distinguishes "confidently wrong" from "undecided",
    which a ranking alone cannot express - a ranking exists either way.
    """
    values = np.asarray(shares, dtype=float)
    values = values[values > 0]
    if values.size <= 1:
        return 1.0
    entropy = float(-np.sum(values * np.log(values)))
    return float(1.0 - entropy / np.log(len(shares)))


@dataclass(frozen=True, slots=True)
class RegionContrast:
    """The same explainer's profile on hard rows and on easy rows."""

    hard_rows: int
    easy_rows: int
    hard_profile: tuple[float, ...]
    easy_profile: tuple[float, ...]
    feature_names: tuple[str, ...]
    profile_agreement: float
    hard_concentration: float
    easy_concentration: float

    @property
    def hard_top_feature(self) -> str:
        return self.feature_names[int(np.argmax(self.hard_profile))]

    @property
    def easy_top_feature(self) -> str:
        return self.feature_names[int(np.argmax(self.easy_profile))]

    @property
    def largest_shift(self) -> tuple[str, float]:
        """The feature whose share moves most between the two regions."""
        deltas = np.asarray(self.hard_profile) - np.asarray(self.easy_profile)
        position = int(np.argmax(np.abs(deltas)))
        return self.feature_names[position], float(deltas[position])

    def as_dict(self) -> dict[str, Any]:
        feature, delta = self.largest_shift
        return {
            "hard_rows": self.hard_rows,
            "easy_rows": self.easy_rows,
            "feature_names": list(self.feature_names),
            "hard_profile": [float(v) for v in self.hard_profile],
            "easy_profile": [float(v) for v in self.easy_profile],
            "profile_agreement": self.profile_agreement,
            "hard_concentration": self.hard_concentration,
            "easy_concentration": self.easy_concentration,
            "hard_top_feature": self.hard_top_feature,
            "easy_top_feature": self.easy_top_feature,
            "largest_shift_feature": feature,
            "largest_shift_delta": delta,
            "reading": (
                "Association between a model's error and its attribution "
                "pattern. Not a cause of the error, and not a statement that "
                "these features are wrong for these patients."
            ),
        }


def contrast_regions(
    explain: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    hard_index: Sequence[int],
    easy_index: Sequence[int],
) -> RegionContrast:
    """Compare one explainer's attribution profile across the two regions.

    ``explain`` returns one attribution row per input row, so any local method
    fits: occlusion, integrated gradients, a linear contribution. Both regions
    go through the identical call, which is what makes the comparison about the
    rows rather than about the method's settings.
    """
    if len(hard_index) == 0 or len(easy_index) == 0:
        raise ValueError("both regions need at least one row to contrast")

    features = tuple(str(column) for column in X.columns)
    hard = X.iloc[list(hard_index)]
    easy = X.iloc[list(easy_index)]

    hard_profile = attribution_profile(explain(hard))
    easy_profile = attribution_profile(explain(easy))

    return RegionContrast(
        hard_rows=len(hard),
        easy_rows=len(easy),
        hard_profile=tuple(float(v) for v in hard_profile),
        easy_profile=tuple(float(v) for v in easy_profile),
        feature_names=features,
        profile_agreement=spearman(hard_profile, easy_profile),
        hard_concentration=concentration(hard_profile),
        easy_concentration=concentration(easy_profile),
    )


def summarise(contrast: RegionContrast) -> dict[str, Any]:
    """State which of the two readings the numbers support, and neither if unclear.

    The two hypotheses are distinguishable and the summary is required to name
    which one it is reporting, so a run cannot produce a table that quietly
    supports whichever story the reader arrived with.
    """
    payload = contrast.as_dict()
    concentration_drop = contrast.easy_concentration - contrast.hard_concentration

    if contrast.profile_agreement >= 0.8 and abs(concentration_drop) < 0.05:
        reading = (
            "same structure: the models attribute alike on rows they fail and "
            "rows they get right, consistent with the information being absent "
            "rather than the models being confused"
        )
    elif concentration_drop > 0.05:
        reading = (
            "diffuse on failure: attribution spreads across more features on the "
            "hard rows, consistent with the models being undecided there rather "
            "than confidently wrong"
        )
    elif concentration_drop < -0.05:
        reading = (
            "concentrated on failure: the models lean harder on fewer features "
            "on the rows they get wrong"
        )
    else:
        reading = (
            "no clear reading: the profiles differ in ordering without a "
            "consistent change in how concentrated they are"
        )

    payload["concentration_drop"] = float(concentration_drop)
    payload["summary"] = reading
    return payload
