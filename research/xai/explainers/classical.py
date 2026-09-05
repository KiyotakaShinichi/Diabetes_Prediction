"""Coefficient, native-importance and model-agnostic explainers.

Three method groups live here because they share one requirement: a fitted
scikit-learn-compatible model and nothing else. The gradient methods need a
torch module and live next door.

The subtlety that matters throughout is **feature space**. Every linear model in
the zoo is wrapped in a pipeline that standardises first, so its coefficients
describe standardised features, not the raw ones a clinician would recognise.
A coefficient of 0.4 on BMI and 0.4 on HighBP are comparable *only* because both
were standardised; on raw units the BMI coefficient would be divided by roughly
seven points of BMI and the comparison would be meaningless. Every function here
states which space it works in, and the records say so too.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from research.model_zoo.contracts import ProbabilityBehavior
from research.model_zoo.registry import REGISTRY
from research.xai.contracts import (
    CapabilityError,
    ExplanationRecord,
    Scope,
    explanation_id,
    normalize_attributions,
    rank_features,
)

#: Shuffles averaged per feature for permutation importance. Five is enough to
#: damp the noise of a single shuffle without turning a global method into the
#: run's dominant cost; the seed is recorded and the spread is measured by the
#: seed-stability study rather than assumed.
PERMUTATION_REPEATS: int = 5

#: Grid resolution for partial dependence. Twenty points across the observed
#: range is enough to see a shape without pretending to a resolution 1,000
#: training rows cannot support.
PDP_GRID_POINTS: int = 20


def _inner_estimator(model: Any) -> Any:
    """The estimator itself, past any preprocessing pipeline."""
    estimator = getattr(model, "estimator", model)
    steps = getattr(estimator, "named_steps", None)
    if steps is not None and "model" in steps:
        return steps["model"]
    return estimator


def _scaler(model: Any) -> Any | None:
    """The fitted preprocessing step, when the pipeline has one."""
    estimator = getattr(model, "estimator", model)
    steps = getattr(estimator, "named_steps", None)
    if steps is not None and "prepare" in steps:
        return steps["prepare"]
    return None


def _scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.decision_scores(X), dtype=float)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


# ============================================================ coefficients

def coefficient_attributions(model: Any) -> np.ndarray:
    """Linear weights in the model's own (standardised) feature space.

    Returned signed: direction is real information for a linear model, and the
    local case reports use it. The global comparison normalises magnitude.
    """
    estimator = _inner_estimator(model)
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None:
        raise CapabilityError(
            f"{getattr(model, 'model_id', estimator)} exposes no coef_; its "
            "capability profile should not claim native coefficients"
        )
    values = np.asarray(coefficients, dtype=float)
    flat: np.ndarray = values.reshape(-1) if values.ndim == 1 else values[0]
    return flat


def native_importance_attributions(model: Any) -> np.ndarray:
    """The estimator's own importance statistic, whatever it calls it."""
    estimator = _inner_estimator(model)
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        raise CapabilityError(
            f"{getattr(model, 'model_id', estimator)} exposes no "
            "feature_importances_; its capability profile should not claim one"
        )
    return np.asarray(values, dtype=float).reshape(-1)


def local_linear_contribution(model: Any, row: pd.DataFrame) -> np.ndarray:
    """Coefficient times the row's standardised value: a per-row contribution.

    This is the linear model's exact local attribution - the terms literally
    sum to the logit minus the intercept - which makes it the one method in the
    zoo whose faithfulness is guaranteed by construction rather than measured.
    """
    coefficients = coefficient_attributions(model)
    scaler = _scaler(model)
    transformed = (
        np.asarray(scaler.transform(row), dtype=float)
        if scaler is not None
        else np.asarray(row, dtype=float)
    )
    contributions: np.ndarray = coefficients * transformed[0]
    return contributions


# ================================================== model-agnostic methods

def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    repeats: int = PERMUTATION_REPEATS,
    seed: int = 0,
) -> np.ndarray:
    """Drop in ROC-AUC when each feature's values are shuffled.

    Measured in ROC-AUC points, so a value is interpretable on its own and
    comparable across models in a way native importances are not - this is why
    permutation serves as the cross-family baseline.

    **Which rows you pass decides what this measures**, and the difference is
    not small. On `PURE_NOISE`, where the label is independent of every column,
    a random forest scored on the rows it was fitted to reports importances of
    roughly 0.12 ROC-AUC points; scored on held-out rows the same forest reports
    0.02-0.06. The first number is memorisation - shuffling a feature breaks the
    lookup the forest built - and it is indistinguishable in the results table
    from a real finding. Logistic regression shows no such gap, because it has
    nothing to memorise with. Every Track M call site therefore passes an
    evaluation partition the model was not fitted on, drawn from TRAIN.

    The correlation caveat is real and is recorded on the method card: shuffling
    ``HighBP`` while ``HighChol`` remains intact lets a correlated partner carry
    the signal, so both can look less important than either is. Note that this
    is not the same as interaction blindness - on an exclusive-or rule this
    method recovers both drivers decisively, because permuting one member while
    the other stays in place destroys the joint structure.
    """
    rng = np.random.default_rng(seed)
    baseline = _roc_auc(y.to_numpy(), _scores(model, X))

    drops = np.zeros(X.shape[1], dtype=float)
    for index, column in enumerate(X.columns):
        losses = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            losses.append(baseline - _roc_auc(y.to_numpy(), _scores(model, shuffled)))
        drops[index] = float(np.mean(losses))
    return drops


def occlusion_attributions(
    model: Any, row: pd.DataFrame, baseline: pd.Series
) -> np.ndarray:
    """Score change when one feature is replaced by its training baseline.

    A single-feature intervention on a *specific row*, which is why - contrary
    to the usual summary of it - this does detect interacting features: on an
    exclusive-or rule, averaged over rows, it puts both drivers on top by a wide
    margin, because moving one bit while the partner stays put flips the parity.
    What it cannot do is say the effect *belongs to the pair*: it hands each
    member the full swing separately, so two features worth one joint effect are
    reported as two independent effects. Attributing the effect to the pair is
    the interaction audit's job, not this one's.

    Its real blind spot is narrower and easy to miss: a feature whose value in
    this row already equals the baseline gets exactly zero, however much the
    model depends on it. For a binary feature with a median baseline that is
    roughly half the rows, which is why local occlusion is read across a case
    sample rather than from any single row.
    """
    deltas: np.ndarray = occlusion_matrix(model, row, baseline)[0]
    return deltas


def occlusion_matrix(
    model: Any, frame: pd.DataFrame, baseline: pd.Series
) -> np.ndarray:
    """Occlusion for every row at once: one array of rows by features.

    Identical arithmetic to calling `occlusion_attributions` per row, and about
    forty times faster, which is the difference between the stability sweep
    costing six minutes per model and costing ten seconds. A one-row DataFrame
    carries roughly 87ms of scikit-learn call overhead on this zoo's pipelines,
    so occluding forty rows one at a time spends almost all of its time in
    dispatch rather than in the model. Setting one column to its baseline for
    the whole frame and scoring once needs one call per feature instead of one
    per feature per row.
    """
    reference = _scores(model, frame)
    deltas = np.zeros((len(frame), frame.shape[1]), dtype=float)
    for index, column in enumerate(frame.columns):
        occluded = frame.copy()
        occluded[column] = baseline[column]
        deltas[:, index] = reference - _scores(model, occluded)
    return deltas


def partial_dependence(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    *,
    grid_points: int = PDP_GRID_POINTS,
) -> dict[str, Any]:
    """Average model score as one feature is swept across its observed range.

    **Not a causal effect and not an individual effect.** It is the model's mean
    response when that input is set to each grid value, averaged over the other
    features as they actually occur. With correlated features it averages over
    combinations that do not exist in the data, which is why the correlation
    audit is reported alongside it.

    The averaging is also what makes this the one method here that genuinely
    fails on interactions. On an exclusive-or rule, a forest that has learned
    the rule to 0.96 held-out ROC-AUC produces PD curves whose two largest
    ranges belong to inert columns, at every seed tried - the driver's effect is
    positive for half the population and negative for the other half, and the
    mean of the two is nothing. A flat PD curve is evidence about the average,
    never evidence that the model ignores the feature.

    The grid spans observed percentiles rather than the full contract range, so
    the curve never extrapolates into territory the model never saw.
    """
    values = X[feature].to_numpy(dtype=float)
    low, high = np.percentile(values, [2.5, 97.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    grid = np.linspace(low, high, grid_points)

    averages = []
    for point in grid:
        probe = X.copy()
        probe[feature] = point
        averages.append(float(np.mean(_scores(model, probe))))

    return {
        "feature": feature,
        "grid": [float(g) for g in grid],
        "average_score": averages,
        "range": float(max(averages) - min(averages)),
        "support": "observed 2.5th-97.5th percentile; no extrapolation",
    }


def individual_conditional_expectation(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    *,
    rows: int = 20,
    grid_points: int = PDP_GRID_POINTS,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-row versions of the partial-dependence curve.

    PDP averages away heterogeneity; ICE shows whether the average describes
    anybody. Curves that fan out mean the average curve is hiding an
    interaction, which makes this a cheap cross-check on the interaction audit.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(X), size=min(rows, len(X)), replace=False)
    sample = X.iloc[np.sort(chosen)]

    values = X[feature].to_numpy(dtype=float)
    low, high = np.percentile(values, [2.5, 97.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    grid = np.linspace(low, high, grid_points)

    curves = []
    for point in grid:
        probe = sample.copy()
        probe[feature] = point
        curves.append(_scores(model, probe))
    matrix = np.vstack(curves).T  # rows x grid

    spans = matrix.max(axis=1) - matrix.min(axis=1)
    return {
        "feature": feature,
        "grid": [float(g) for g in grid],
        "curves": [[float(v) for v in row] for row in matrix],
        "row_count": int(matrix.shape[0]),
        "mean_span": float(np.mean(spans)),
        "span_dispersion": float(np.std(spans)),
    }


# ================================================================ records

def build_record(
    model_id: str,
    method: str,
    method_version: str,
    scope: Scope,
    feature_names: tuple[str, ...],
    raw: np.ndarray,
    *,
    baseline_reference: str,
    sample_id: int | None = None,
    seed: int | None = None,
    runtime_seconds: float | None = None,
    prediction: float | None = None,
    prediction_probability: float | None = None,
    notes: str = "",
) -> ExplanationRecord:
    """Assemble a record from a raw attribution vector."""
    spec = REGISTRY.get(model_id)
    raw = np.asarray(raw, dtype=float)
    return ExplanationRecord(
        explanation_id=explanation_id(model_id, method, sample_id=sample_id, seed=seed),
        model_id=model_id,
        model_family=spec.family.value,
        method=method,
        method_version=method_version,
        scope=scope,
        feature_names=feature_names,
        raw_attributions=tuple(float(v) for v in raw),
        normalized_attributions=tuple(float(v) for v in normalize_attributions(raw)),
        ranking=rank_features(feature_names, raw),
        baseline_reference=baseline_reference,
        seed=seed,
        sample_id=sample_id,
        prediction=prediction,
        prediction_probability=prediction_probability,
        runtime_seconds=runtime_seconds,
        notes=notes,
    )


def timed(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call something and report how long it took, for the cost telemetry."""
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def has_probability(model_id: str) -> bool:
    return (
        REGISTRY.get(model_id).probability_behavior
        is not ProbabilityBehavior.HARD_LABELS_ONLY
    )
