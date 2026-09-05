"""Datasets whose true explanation is known before any model is fitted.

Every result in Track M is a statement about explanations, and an explanation
cannot be validated against the real data: nobody knows which features the
diabetes label actually depends on, which is why the models were built in the
first place. Measuring agreement between methods on real rows shows that they
agree, not that either is right.

These four worlds close that gap. The label comes from a rule this module
wrote, so the correct attribution is known in advance and an explainer can be
*scored* rather than merely inspected. They are the only place in the track
where the phrase "the right answer" is defensible.

They are also chosen to break things, not to flatter:

* `ONE_DOMINANT_FEATURE` is the easy case every method must pass. A method that
  cannot find a single dominant driver is broken, and the test says so.
* `ADDITIVE_TWO_FEATURE` separates two real drivers from eight inert columns,
  which is where impurity importance starts preferring the continuous one.
* `XOR_INTERACTION` is the trap, and it splits the methods cleanly. Neither
  driver has any marginal association with the label, so **partial dependence**
  reports approximately nothing for both - across five seeds its top two are
  always inert columns, never a driver. Permutation importance and row-wise
  occlusion both recover the pair decisively over the same models. A method
  being model-agnostic does not make it interaction-aware, and this world is
  where that distinction stops being theoretical.
* `PURE_NOISE` is the negative control. There is no true explanation, so any
  method that produces a confident-looking ranking here is ranking noise. A
  ranking will still exist - something has to come first - which is exactly why
  a rank is never reported in this track without a magnitude beside it. It also
  exposes the partition question: scored on the rows it was fitted to, a random
  forest reports permutation importances around 0.12 ROC-AUC points on labels
  that contain nothing. That number measures memorisation.

Prevalence is held near 50% by thresholding at the median rather than at zero,
so no world doubles as an accidental class-imbalance test; `model_zoo.synthetic`
already owns that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ml_core import feature_contract
from research.model_zoo.synthetic import contract_frame

FEATURES: tuple[str, ...] = feature_contract.FEATURE_NAMES


class XaiWorld(StrEnum):
    """The four ground-truth worlds."""

    ONE_DOMINANT_FEATURE = "one_dominant_feature"
    ADDITIVE_TWO_FEATURE = "additive_two_feature"
    XOR_INTERACTION = "xor_interaction"
    PURE_NOISE = "pure_noise"


@dataclass(frozen=True, slots=True)
class XaiWorldDataset:
    """A world, its rows, and the attribution a faithful method should give."""

    world: XaiWorld
    X: pd.DataFrame
    y: pd.Series
    #: Features the label-generating rule actually reads.
    driving_features: tuple[str, ...]
    #: True when the drivers carry no marginal signal, so single-feature
    #: methods are expected to miss them. Read by the interaction audit.
    interaction_only: bool
    #: What a correct method should report, in one sentence, for the report.
    expectation: str
    description: str

    @property
    def inert_features(self) -> tuple[str, ...]:
        """Columns the label-generating rule never reads."""
        return tuple(f for f in FEATURES if f not in self.driving_features)

    def is_driving(self, feature: str) -> bool:
        return feature in self.driving_features

    def as_dict(self) -> dict[str, Any]:
        return {
            "world": self.world.value,
            "rows": len(self.X),
            "prevalence": float(self.y.mean()),
            "driving_features": list(self.driving_features),
            "inert_features": list(self.inert_features),
            "interaction_only": self.interaction_only,
            "expectation": self.expectation,
            "description": self.description,
        }


def make(world: XaiWorld, *, rows: int = 600, seed: int = 0) -> XaiWorldDataset:
    """Build one ground-truth world deterministically."""
    rng = np.random.default_rng(seed)
    X = contract_frame(rows, rng)

    if world is XaiWorld.ONE_DOMINANT_FEATURE:
        score = 2.5 * _z(X["GenHlth"]) + rng.normal(0, 0.4, rows)
        return XaiWorldDataset(
            world, X, _balanced_labels(score), ("GenHlth",), False,
            "GenHlth first by a wide margin; the other nine near zero.",
            "One ordinal feature drives the label and nothing else contributes. "
            "The easiest possible attribution problem, and therefore the one a "
            "broken explainer fails first.",
        )

    if world is XaiWorld.ADDITIVE_TWO_FEATURE:
        score = 1.2 * _z(X["BMI"]) + 1.2 * _z(X["Age"]) + rng.normal(0, 0.35, rows)
        return XaiWorldDataset(
            world, X, _balanced_labels(score), ("BMI", "Age"), False,
            "BMI and Age in the top two in either order; no interaction term.",
            "Two features contribute additively and with equal weight. This "
            "separates methods that find both drivers from methods that split "
            "the credit unevenly because one input is continuous and the other "
            "is a thirteen-level ordinal.",
        )

    if world is XaiWorld.XOR_INTERACTION:
        parity = (X["HighBP"].astype(int) ^ X["HighChol"].astype(int)).to_numpy()
        flipped = rng.random(rows) < 0.03
        labels = pd.Series(np.where(flipped, 1 - parity, parity), name="target")
        return XaiWorldDataset(
            world, X, labels, ("HighBP", "HighChol"), True,
            "Partial dependence is blind - its top two are inert columns. "
            "Permutation importance and row-wise occlusion both recover the "
            "pair, because both intervene while the partner is left in place.",
            "The label is the exclusive-or of two binary features with 3% label "
            "noise. Neither driver has any marginal association with the label, "
            "so a population-averaged curve sees nothing and a greedy single "
            "tree splits on the inert columns instead. A random forest learns "
            "the rule at roughly 0.96 held-out ROC-AUC, which is what makes the "
            "world usable: there is a model that genuinely uses both features, "
            "so a method reporting otherwise is wrong rather than merely "
            "describing a model that failed.",
        )

    if world is XaiWorld.PURE_NOISE:
        labels = pd.Series(rng.integers(0, 2, rows), name="target")
        return XaiWorldDataset(
            world, X, labels, (), False,
            "Scored on held-out rows, no feature comes close to the magnitude a "
            "real driver reaches. Scored on the fitting rows, a high-capacity "
            "model reports large importances that measure memorisation.",
            "Labels independent of every feature. There is no true explanation, "
            "which makes this the control for methods that always produce a "
            "confident-looking answer whatever they are given.",
        )

    raise ValueError(f"unknown XAI world: {world!r}")


def _balanced_labels(score: np.ndarray) -> pd.Series:
    """Threshold at the median, so every world sits near 50% prevalence.

    Thresholding at zero would let the noise draw decide the base rate, and a
    shifting base rate moves every attribution magnitude for reasons that have
    nothing to do with the explainer under test.
    """
    return pd.Series((score > float(np.median(score))).astype(int), name="target")


def _z(column: pd.Series) -> np.ndarray:
    values = column.to_numpy(dtype=float)
    spread = float(values.std())
    if spread == 0:
        return np.zeros_like(values)
    standardised: np.ndarray = (values - values.mean()) / spread
    return standardised


def split(
    dataset: XaiWorldDataset, *, train_fraction: float = 0.7, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """A deterministic train/held-out split of a world.

    Explanations are computed on training rows throughout Track M, so the
    held-out part exists to confirm the model learned the rule at all before
    anything is asked about *why*. Explaining a model that never fit would
    measure the explainer against noise and call the result a disagreement.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.X))
    cut = int(len(order) * train_fraction)
    train, held_out = order[:cut], order[cut:]
    return (
        dataset.X.iloc[train].reset_index(drop=True),
        dataset.y.iloc[train].reset_index(drop=True),
        dataset.X.iloc[held_out].reset_index(drop=True),
        dataset.y.iloc[held_out].reset_index(drop=True),
    )


def baseline_row(X: pd.DataFrame) -> pd.Series:
    """The training-median reference that occlusion and IG measure against.

    Computed from whatever partition is handed in, and every call site hands in
    TRAINING rows. A baseline built from evaluation rows would leak the
    distribution a model is judged on into the explanation of it.
    """
    return X.median(axis=0)
