"""Canonical prediction-evaluation metrics.

Extracted verbatim from the byte-identical copies that previously lived in both
logisticregression_only.py and boostedtrees_ab.py. The metric set, the
zero-division policy and the JSON shape are unchanged, so metrics.json and
boosted_metrics.json keep exactly the schema already committed.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Exact key set of an evaluate_predictions() result. Part of the committed
#: metrics JSON schema, so it must not change without regenerating artifacts.
EVALUATION_KEYS = frozenset({
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "brier_score", "cohen_kappa", "mcc", "confusion_matrix",
})


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    """Compute the standard evaluation metrics for a binary classifier.

    ``zero_division=0`` is applied to precision, recall and F1 so a degenerate
    prediction (for example all-negative) yields 0.0 rather than raising - this
    matters inside the bootstrap, where resamples can be degenerate.

    The confusion matrix is returned as a nested list, so the whole result is
    JSON-serializable exactly as written.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
