"""Decision-threshold selection.

Youden's J statistic (J = sensitivity + specificity - 1 = TPR - FPR) picks the
cut-point that maximises the balance between catching positives and avoiding
false alarms. That is the rule both maintained pipelines already used, and it is
unchanged here.

What changed is a correctness defect. ``sklearn.metrics.roc_curve`` prepends a
synthetic ``inf`` threshold to represent "classify everything as negative"; it
is not an observed score. J at that point is ``0 - 0 = 0``, so whenever no real
cut-point achieved ``J > 0`` the previous ``np.argmax`` landed on index 0 and
returned ``inf`` - a threshold that would classify every patient as negative.
Observed on three inputs: all-identical probabilities, scores no better than
random, and a single-class target.

The selection now considers only finite candidates, so a serving threshold can
never be ``inf``, ``-inf``, ``NaN`` or outside ``[0, 1]``. For every input where
the old implementation already returned a finite value, the selected threshold
is identical - the excluded entry could not have been the argmax in those cases.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_youden_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the probability threshold maximising Youden's J.

    The result is guaranteed to be a finite float within ``[0, 1]``: it is
    always one of the observed probabilities.

    Ties are broken deterministically toward the first maximum, which is the
    highest threshold among the tied candidates because ``roc_curve`` returns
    thresholds in decreasing order. That matches the previous ``np.argmax``
    behaviour exactly.

    Raises:
        ValueError: if the inputs are empty or mismatched, if ``y_proba``
            contains a non-finite value or a value outside ``[0, 1]``, or if
            ``y_true`` holds only one class. A single-class target makes TPR or
            FPR undefined, so there is no honest Youden answer - inventing a
            neutral 0.5 would hide a broken evaluation set.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"y_true and y_proba must be the same length, got "
            f"{y_true.shape[0]} and {y_proba.shape[0]}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_proba must not be empty")
    if not np.all(np.isfinite(y_proba)):
        raise ValueError("y_proba must contain only finite values")
    if y_proba.min() < 0.0 or y_proba.max() > 1.0:
        raise ValueError("y_proba must lie within [0, 1]; these are probabilities")

    classes = np.unique(y_true)
    if classes.size < 2:
        raise ValueError(
            "y_true must contain both classes; Youden's J is undefined for a "
            f"single-class target (found only {classes.tolist()})"
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Drop the synthetic leading threshold (inf) that roc_curve prepends.
    finite = np.isfinite(thresholds)
    if not finite.any():  # pragma: no cover - impossible with two classes present
        raise ValueError("roc_curve produced no finite threshold candidate")

    youden_j = tpr[finite] - fpr[finite]
    best = int(np.argmax(youden_j))
    threshold = float(thresholds[finite][best])

    # Every candidate is an observed probability, already validated above.
    return threshold
