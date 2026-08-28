"""One evaluator, applied identically to all four model families.

Every model is scored by this module and no other, so a difference in a reported
number is a difference between models rather than between measurement code.

Two things are deliberately separated:

* DISCRIMINATION - can the model rank a positive above a negative? ROC-AUC is
  the primary metric, PR-AUC the secondary one.
* PROBABILITY QUALITY - are the numbers it reports believable as probabilities?
  Brier score, log loss and expected calibration error answer that, and they
  matter here because the product this research feeds shows a visitor a
  percentage.

A caveat that belongs beside every calibration number produced here: the study
dataset is close to 50/50, so calibration is measured against that base rate.
These figures describe internal consistency on this dataset and are not evidence
about any population prevalence. See docs/research/track_k_protocol.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from research.track_k import protocol

#: Probabilities are clipped before log loss so a confident wrong prediction
#: cannot produce an infinite penalty that swamps every other number.
_LOG_LOSS_EPS = 1e-15


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_predicted": self.mean_predicted,
            "observed_rate": self.observed_rate,
        }


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate. Reported beside recall, which it trades against."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive = matrix[0, 0], matrix[0, 1]
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator else 0.0


def reliability_bins(
    y_true: np.ndarray, y_proba: np.ndarray, *, bins: int = protocol.ECE_BINS
) -> list[ReliabilityBin]:
    """Equal-width reliability bins over [0, 1].

    Empty bins are dropped rather than reported as 0/0: a bin nothing fell into
    says nothing about calibration, and including it as zero would drag a
    weighted average toward a value no prediction supports.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # The last bin is closed on the right so a prediction of exactly 1.0 lands
    # somewhere rather than falling off the end.
    indices = np.clip(np.digitize(y_proba, edges[1:-1], right=False), 0, bins - 1)

    result: list[ReliabilityBin] = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        if count == 0:
            continue
        result.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_predicted=float(y_proba[mask].mean()),
                observed_rate=float(y_true[mask].mean()),
            )
        )
    return result


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, *, bins: int = protocol.ECE_BINS
) -> float:
    """Count-weighted mean gap between predicted probability and observed rate.

    The standard ECE. It is a summary and hides direction and shape, which is
    why the bins themselves are persisted alongside it.
    """
    populated = reliability_bins(y_true, y_proba, bins=bins)
    if not populated:
        return 0.0
    total = sum(item.count for item in populated)
    return float(
        sum(
            item.count * abs(item.mean_predicted - item.observed_rate)
            for item in populated
        )
        / total
    )


def calibration_slope_intercept(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float]:
    """Slope and intercept of a logistic recalibration fit.

    Outcomes are regressed on the LOGIT of the predicted probability. A
    perfectly calibrated model gives slope 1 and intercept 0; slope below 1
    indicates over-confident spread, and a non-zero intercept indicates a
    systematic shift.

    Returns NaN when the fit cannot be identified - a single-class partition or
    degenerate probabilities - rather than returning a number that would look
    like a measurement.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    clipped = np.clip(y_proba, _LOG_LOSS_EPS, 1 - _LOG_LOSS_EPS)
    logit = np.log(clipped / (1 - clipped))
    if not np.isfinite(logit).all() or float(np.std(logit)) < 1e-12:
        return float("nan"), float("nan")

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logit.reshape(-1, 1), y_true)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def evaluate(
    y_true: np.ndarray, y_proba: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    """The full metric set for one model on one partition.

    ``threshold`` is supplied rather than chosen here: it is selected on
    validation and then frozen, so the test evaluation cannot quietly optimise
    its own operating point.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_proba = np.asarray(y_proba, dtype=np.float64).reshape(-1)
    y_pred = (y_proba >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    slope, intercept = calibration_slope_intercept(y_true, y_proba)

    return {
        # Discrimination
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float("nan")
        if single_class
        else float(average_precision_score(y_true, y_proba)),
        # Threshold-dependent
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        # Probability quality
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(
            log_loss(y_true, np.clip(y_proba, _LOG_LOSS_EPS, 1 - _LOG_LOSS_EPS), labels=[0, 1])
        ),
        "ece": expected_calibration_error(y_true, y_proba),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        # Context
        "threshold": float(threshold),
        "n_rows": len(y_true),
        "positive_rate": float(y_true.mean()),
        "mean_predicted": float(y_proba.mean()),
    }


def metric_names() -> tuple[str, ...]:
    """Every scalar metric ``evaluate`` produces, in report order."""
    return (
        protocol.PRIMARY_METRIC,
        *protocol.SECONDARY_METRICS,
        *protocol.CALIBRATION_METRICS,
    )
