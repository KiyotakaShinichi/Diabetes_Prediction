"""Bootstrap confidence intervals for evaluation metrics.

Extracted from the two pipeline copies, which differed only cosmetically
(tuple unpacking and two inlined locals) and were semantically identical. The
resampling semantics are preserved exactly so previously reported intervals stay
reproducible:

* the sampling unit is one test-set row, drawn with replacement, ``n`` draws per
  resample where ``n`` is the test-set size;
* randomness comes from a local ``numpy.random.RandomState`` seeded per call, so
  the global NumPy RNG is never read or mutated;
* a resample that happens to contain a single class is skipped, because ROC AUC
  is undefined for it;
* intervals are percentile intervals at ``alpha``.

``RandomState`` is kept deliberately rather than modernised to
``default_rng``: the two produce different draw sequences, which would silently
change every published interval.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Metrics carried through the bootstrap. Narrower than EVALUATION_KEYS on
#: purpose: confusion_matrix is not a scalar, and kappa/MCC were never reported
#: with intervals. Part of the committed metrics JSON schema.
BOOTSTRAP_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier_score")


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap intervals for the scalar evaluation metrics.

    Returns ``{metric: {"mean": .., "ci_lower": .., "ci_upper": ..}}``.

    Raises:
        ValueError: if every resample was degenerate, so no interval can be
            computed. That is deliberate: the previous implementation reached
            ``np.percentile`` on an empty array and failed with an obscure
            IndexError, which hid the real cause.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_boot: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p = y_proba[idx]
        y_pred = (y_p >= threshold).astype(int)

        # Skip degenerate resamples (single class): ROC AUC is undefined.
        if len(np.unique(y_t)) < 2:
            continue

        metrics_boot["accuracy"].append(float(accuracy_score(y_t, y_pred)))
        metrics_boot["precision"].append(float(precision_score(y_t, y_pred, zero_division=0)))
        metrics_boot["recall"].append(float(recall_score(y_t, y_pred, zero_division=0)))
        metrics_boot["f1"].append(float(f1_score(y_t, y_pred, zero_division=0)))
        metrics_boot["roc_auc"].append(float(roc_auc_score(y_t, y_p)))
        metrics_boot["brier_score"].append(float(brier_score_loss(y_t, y_p)))

    if not metrics_boot["accuracy"]:
        raise ValueError(
            "every bootstrap resample contained a single class, so no interval "
            "could be estimated; check that y_true holds both classes"
        )

    result = {}
    for metric, values in metrics_boot.items():
        arr = np.array(values)
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        result[metric] = {"mean": float(arr.mean()), "ci_lower": lo, "ci_upper": hi}
    return result
