"""Post-hoc probability calibration, fitted on validation only.

The protocol fits any calibrator on the validation partition and applies it to
test. Fitting on test would make the reported calibration a description of the
fit rather than a measurement of the model.

Calibration is applied only when it actually helps, and candidates are compared
OUT OF FOLD inside validation. Judging a candidate on the rows it was fitted on
would hand the comparison to whichever candidate can memorise best: measured
here, in-sample isotonic scored an ECE of 1.6e-17 against a sigmoid's 0.0188,
which says nothing about generalisation. Scored out of fold the same pair gives
0.0069 and 0.0177 against an uncalibrated 0.0171 - a real difference. The winner
is then refitted on the whole validation partition before being applied to test.

If no candidate beats the uncalibrated model, the raw probabilities are kept and
that decision is recorded. A transform that makes a model worse is not applied
merely because the protocol offers one.

The base-rate caveat from the protocol applies to everything here: this dataset
is close to 50/50, so these numbers describe internal consistency on this study
population and are not evidence about any wider prevalence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from research.track_k import evaluation, protocol

#: Isotonic regression needs enough validation rows to estimate a step function
#: without simply memorising it. Below this, only the two-parameter sigmoid is
#: considered.
MIN_ROWS_FOR_ISOTONIC: int = 1000

_EPS = 1e-12


class Calibrator(Protocol):
    """Maps raw probabilities to calibrated ones."""

    def __call__(self, proba: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    """What calibration was chosen for one model, and why."""

    method: str
    applied: bool
    validation_ece_before: float
    validation_ece_after: float
    candidates: dict[str, float]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "applied": self.applied,
            "validation_ece_before": self.validation_ece_before,
            "validation_ece_after": self.validation_ece_after,
            "candidates_validation_ece": self.candidates,
            "reason": self.reason,
            "fitted_on": "validation",
        }


def identity_calibrator() -> Calibrator:
    def apply(proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=np.float64)

    return apply


def fit_sigmoid(y_true: np.ndarray, y_proba: np.ndarray) -> Calibrator:
    """Platt scaling: a logistic fit on the logit of the raw probability."""
    clipped = np.clip(np.asarray(y_proba, dtype=np.float64), _EPS, 1 - _EPS)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logit, np.asarray(y_true).astype(int))

    def apply(proba: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(proba, dtype=np.float64), _EPS, 1 - _EPS)
        transformed = np.log(values / (1 - values)).reshape(-1, 1)
        return np.asarray(model.predict_proba(transformed)[:, 1], dtype=np.float64)

    return apply


def fit_isotonic(y_true: np.ndarray, y_proba: np.ndarray) -> Calibrator:
    """Isotonic regression: monotone, non-parametric, hungrier for data."""
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(y_proba, dtype=np.float64), np.asarray(y_true).astype(float))

    def apply(proba: np.ndarray) -> np.ndarray:
        return np.asarray(
            model.predict(np.asarray(proba, dtype=np.float64)), dtype=np.float64
        )

    return apply


#: Folds used to score calibrator candidates out of sample. See the note in
#: select_calibrator on why in-sample scoring cannot be used here.
SELECTION_FOLDS: int = 5


def _out_of_fold_ece(
    fitter: Any,
    y_val: np.ndarray,
    val_proba: np.ndarray,
    *,
    bins: int,
    folds: int,
    seed: int,
) -> float:
    """Score a calibrator on rows it was not fitted on.

    Selecting by in-sample ECE is not a fair comparison between a
    two-parameter sigmoid and a non-parametric isotonic fit: isotonic can
    reproduce the validation distribution almost exactly and scores an ECE
    near zero regardless of whether it generalises. Measured on this dataset,
    in-sample isotonic scored 1.6e-17 against a sigmoid's 0.0188 - not because
    it was better, but because it had memorised the rows it was judged on.

    Each fold fits on the other folds and scores on the held-out one, so the
    number returned describes behaviour on unseen rows. Everything still stays
    inside the validation partition; test is untouched.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    transformed = np.empty_like(val_proba, dtype=np.float64)
    for fit_index, score_index in splitter.split(val_proba.reshape(-1, 1), y_val):
        calibrator = fitter(y_val[fit_index], val_proba[fit_index])
        transformed[score_index] = calibrator(val_proba[score_index])
    return evaluation.expected_calibration_error(y_val, transformed, bins=bins)


def select_calibrator(
    y_val: np.ndarray,
    val_proba: np.ndarray,
    *,
    bins: int = protocol.ECE_BINS,
    min_rows_for_isotonic: int = MIN_ROWS_FOR_ISOTONIC,
    folds: int = SELECTION_FOLDS,
    seed: int = protocol.SPLIT_SEED,
) -> tuple[Calibrator, CalibrationOutcome]:
    """Choose the calibration that most improves OUT-OF-FOLD validation ECE, or none.

    Candidates are scored out of fold within the validation partition, then the
    winner is refitted on all of validation before being applied to test. The
    test partition takes no part in the decision; it only ever sees the winner
    applied.
    """
    y_val = np.asarray(y_val).astype(int)
    val_proba = np.asarray(val_proba, dtype=np.float64)
    baseline = evaluation.expected_calibration_error(y_val, val_proba, bins=bins)

    fitters: dict[str, Any] = {"sigmoid": fit_sigmoid}
    if len(y_val) >= min_rows_for_isotonic:
        fitters["isotonic"] = fit_isotonic

    candidates: dict[str, tuple[Calibrator, float]] = {}
    for name, fitter in fitters.items():
        try:
            score = _out_of_fold_ece(
                fitter, y_val, val_proba, bins=bins, folds=folds, seed=seed
            )
            # The applied calibrator is refitted on the whole validation
            # partition; only its SELECTION was out of fold.
            candidates[name] = (fitter(y_val, val_proba), score)
        except (ValueError, RuntimeError):
            # A degenerate validation partition cannot support a fit; that is a
            # reason to skip the candidate, not to abort the benchmark.
            continue

    scores = {name: score for name, (_fn, score) in candidates.items()}
    if not candidates:
        return identity_calibrator(), CalibrationOutcome(
            method="none",
            applied=False,
            validation_ece_before=baseline,
            validation_ece_after=baseline,
            candidates=scores,
            reason="no calibrator could be fitted on the validation partition",
        )

    best_name = min(scores, key=lambda name: scores[name])
    best_score = scores[best_name]

    if best_score >= baseline:
        return identity_calibrator(), CalibrationOutcome(
            method="none",
            applied=False,
            validation_ece_before=baseline,
            validation_ece_after=baseline,
            candidates=scores,
            reason=(
                f"best candidate {best_name} scored {best_score:.5f} out of fold "
                f"against an uncalibrated {baseline:.5f}; calibration was not "
                "applied because it did not improve the criterion"
            ),
        )

    return candidates[best_name][0], CalibrationOutcome(
        method=best_name,
        applied=True,
        validation_ece_before=baseline,
        validation_ece_after=best_score,
        candidates=scores,
        reason=(
            f"{best_name} improved out-of-fold validation ECE from {baseline:.5f} "
            f"to {best_score:.5f}; refitted on the full validation partition"
        ),
    )


def select_threshold(y_val: np.ndarray, val_proba: np.ndarray) -> float:
    """Operating threshold from Youden's J on VALIDATION.

    Reuses the shared implementation the production pipelines use, so the
    operating point is chosen the same way here as it is there - and chosen on
    validation, so the test evaluation cannot tune its own threshold.
    """
    from ml_core.thresholds import compute_youden_threshold

    return float(compute_youden_threshold(np.asarray(y_val).astype(int), np.asarray(val_proba)))
