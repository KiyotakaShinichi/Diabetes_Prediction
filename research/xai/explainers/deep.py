"""Gradient-based explanations for the zoo's torch models.

A neural network is the one model family here that can be asked *why* by
differentiation rather than by re-running it on altered inputs. That is cheaper
and more precise where it applies - and it applies to fewer of these models than
"it is a neural network" suggests.

The zoo's modules all take ``(numeric, levels)``: a standardised float vector and
a vector of integer codes for the discrete features. Most of them delete
``levels`` on the first line of ``forward`` and treat the ordinal codes as
numbers, so every feature reaches the network through the differentiable input
and a gradient describes the whole model. Two do not. `ft_transformer` tokenises
each discrete feature through an embedding table, and `tab_transformer` does the
same for nine of the ten; an embedding lookup is a table index, and the gradient
of a table index is not small - it does not exist. Ask those two for an input
gradient and nine of ten features come back as exactly 0.0, which in a results
table is indistinguishable from "the model ignores this feature".

So gradient support here is declared per architecture and verified by
measurement in `tests/test_xai_deep.py`, which builds every torch model and
checks which feature slots actually carry a derivative. Excluding a model that
cannot be differentiated is not a gap in the study; it is one of its findings,
and `capabilities.py` records the reason beside it.

The other thing this module owns is the neural additive model's **exact**
attribution. That architecture computes one subnetwork per feature and sums
them, so its per-feature terms are not an approximation of an explanation - they
are the model. It is the only deep model in the zoo whose attribution needs no
faithfulness proxy, and it sits here rather than with the classical explainers
because reaching it means reaching into a torch module.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.xai.contracts import CapabilityError

#: Riemann steps along the baseline-to-input path for integrated gradients.
#:
#: Set from measurement, not convention. The completeness gap - how far the
#: summed attribution falls short of the score difference the axiom promises -
#: was measured for every gradient-capable architecture at 8, 32, 128 and 512
#: midpoint steps. Seven of the eight are already under 0.007 logits at 32.
#: `feature_token_mixer` is not: it sits at 1.34 logits at 8 steps and 0.105 at
#: 32, because its token-mixing blocks make the path from baseline to input far
#: more curved than the others'. At 128 steps every architecture is under 0.004
#: and the worst case is `feature_token_mixer` at 0.0036.
#:
#: So the budget is set by the hardest architecture rather than the typical one,
#: and the per-record gap is stored anyway - a step count chosen once cannot
#: guarantee an axiom holds for a model nobody has run yet.
INTEGRATED_GRADIENT_STEPS: int = 128


def _module(model: Any) -> Any:
    """The fitted ``nn.Module``, in eval mode.

    Eval mode is not a detail. These architectures contain batch normalisation
    and dropout; in training mode a row's output would depend on the other rows
    in the batch and on a random mask, so an "explanation" would change every
    time it was computed and the seed-stability study would be measuring
    dropout.
    """
    module = getattr(model, "model", None)
    if module is None:
        raise CapabilityError(
            f"{getattr(model, 'model_id', model)} has no torch module; gradient "
            "methods apply to the deep families only"
        )
    module.eval()
    return module


def _encoded(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    encode = getattr(model, "encode", None)
    if encode is None:
        raise CapabilityError(
            f"{getattr(model, 'model_id', model)} exposes no encoder; gradient "
            "methods need the standardised input the module actually consumes"
        )
    numeric, levels = encode(frame)
    return np.asarray(numeric, dtype=np.float32), np.asarray(levels, dtype=np.int64)


def _logit_gradients(model: Any, numeric: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """d(logit)/d(numeric) for a batch, one row of derivatives per input row.

    Summing the batch's logits before differentiating is safe only because the
    module is in eval mode, where each row's output depends on its own input
    alone. That is asserted rather than assumed: `_module` puts it in eval mode
    and `test_xai_deep.py` checks a row's gradient is unchanged by its
    neighbours.
    """
    import torch

    module = _module(model)
    inputs = torch.tensor(numeric, dtype=torch.float32, requires_grad=True)
    codes = torch.tensor(levels, dtype=torch.int64)

    logits = module(inputs, codes)
    (gradients,) = torch.autograd.grad(logits.sum(), inputs)
    return np.asarray(gradients.detach().numpy(), dtype=float)


def _logits(model: Any, numeric: np.ndarray, levels: np.ndarray) -> np.ndarray:
    import torch

    module = _module(model)
    with torch.no_grad():
        values = module(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(levels, dtype=torch.int64),
        )
    return np.asarray(values.numpy(), dtype=float).reshape(-1)


# ============================================================ the three methods

def input_gradient(model: Any, rows: pd.DataFrame) -> np.ndarray:
    """Local sensitivity: how the logit responds to an infinitesimal nudge.

    The slope at one point, in standardised units, and nothing more. Two
    consequences worth stating because both look like findings when they are
    not: a feature the model is already saturated on has a near-zero gradient
    however decisive it was, and the gradient ignores how far the feature
    actually is from anywhere interesting, so a steep slope on a feature that
    barely varies outranks a gentle one on a feature that spans its range.
    """
    _module(model)  # refuse a non-deep model here, where the message is clearest
    numeric, levels = _encoded(model, rows)
    return _logit_gradients(model, numeric, levels)


def gradient_x_input(model: Any, rows: pd.DataFrame) -> np.ndarray:
    """Sensitivity scaled by the encoded value: a first-order contribution.

    Multiplying by the input silently chooses a reference point, and in this
    zoo that reference is **not one thing**. Track K's standardiser deliberately
    leaves binary indicators on their raw 0/1 scale and standardises only the
    five continuous and ordinal features. So the implicit zero is:

    * the **training mean** for GenHlth, BMI, Age, PhysHlth and Education;
    * the literal value **0 - "No"** for HighBP, HighChol, DiffWalk,
      HeartDiseaseorAttack and PhysActivity.

    The consequence is concrete and easy to misread as a finding. Every patient
    without high blood pressure receives exactly zero attribution on HighBP,
    at every model, however heavily the model relies on that feature - not
    because the feature is unimportant but because "No" is the origin. Half the
    zeros this method produces on binary features are structural.

    A row at the encoder's centre therefore gets no attribution anywhere, and
    the *data* mean is not that row: with binary means near 0.5, the mean row
    encodes to 0.5 on five of ten features rather than to zero.
    """
    numeric, levels = _encoded(model, rows)
    gradients = _logit_gradients(model, numeric, levels)
    scaled: np.ndarray = gradients * np.asarray(numeric, dtype=float)
    return scaled


def integrated_gradients(
    model: Any,
    rows: pd.DataFrame,
    baseline: pd.Series,
    *,
    steps: int = INTEGRATED_GRADIENT_STEPS,
) -> np.ndarray:
    """Attribution accumulated along the straight path from a baseline row.

    Integrated gradients exists to fix gradient-times-input's saturation
    problem: instead of the slope at the endpoint it averages the slope along
    the whole path, so a feature the model has already saturated on still
    collects the attribution it earned on the way. The price is that the answer
    is entirely relative to the baseline, and no baseline is neutral. This one
    is the training median, declared on the method and recorded on every result.

    Uses the midpoint rule rather than either endpoint, which for the same step
    count gives a materially smaller completeness gap - and the gap is measured
    rather than assumed; see `completeness_gap`.

    The ``levels`` codes are held at the row's own values along the whole path.
    For every model this method is offered on, ``forward`` discards them, so
    that choice changes nothing; the two architectures where it would change
    something are the two excluded from gradient methods entirely.
    """
    numeric, levels = _encoded(model, rows)
    frame = pd.DataFrame([baseline], columns=rows.columns)
    base_numeric, _ = _encoded(model, frame)

    difference = np.asarray(numeric, dtype=float) - np.asarray(base_numeric, dtype=float)
    # Midpoint Riemann sum: alphas at the centre of each of ``steps`` intervals.
    alphas = (np.arange(steps, dtype=float) + 0.5) / steps

    accumulated = np.zeros_like(difference)
    for alpha in alphas:
        point = np.asarray(base_numeric, dtype=float) + alpha * difference
        accumulated += _logit_gradients(model, point.astype(np.float32), levels)

    attributions: np.ndarray = difference * accumulated / steps
    return attributions


def completeness_gap(
    model: Any,
    rows: pd.DataFrame,
    baseline: pd.Series,
    *,
    steps: int = INTEGRATED_GRADIENT_STEPS,
) -> np.ndarray:
    """How far integrated gradients falls short of the axiom it claims.

    The method promises that the attributions sum to ``f(x) - f(baseline)``.
    With a finite number of Riemann steps they do not, quite, and the shortfall
    is the honest measure of the approximation error in an individual
    explanation. Reported per row rather than averaged, because an average gap
    hides the rows where the path was hardest to integrate - which tend to be
    exactly the rows a case study would pick.
    """
    numeric, levels = _encoded(model, rows)
    frame = pd.DataFrame([baseline], columns=rows.columns)
    base_numeric, base_levels = _encoded(model, frame)

    attributions = integrated_gradients(model, rows, baseline, steps=steps)
    at_input = _logits(model, numeric, levels)
    at_baseline = float(_logits(model, base_numeric, base_levels)[0])

    gap: np.ndarray = attributions.sum(axis=1) - (at_input - at_baseline)
    return gap


# ================================================ the exact additive attribution

def additive_contributions(model: Any, rows: pd.DataFrame) -> np.ndarray:
    """Per-feature logit terms from a neural additive model, exactly.

    The architecture computes one subnetwork per feature and sums the results,
    so these terms are the model rather than a description of it: they add up to
    the logit minus the bias by construction, not by approximation. That makes
    this the deep counterpart of the linear model's coefficient-times-value
    decomposition, and the only deep attribution in the zoo that needs no
    faithfulness proxy.
    """
    import torch

    module = _module(model)
    contributions = getattr(module, "feature_contributions", None)
    if contributions is None:
        raise CapabilityError(
            f"{getattr(model, 'model_id', model)} is not additive by construction; "
            "its capability profile should not claim a native importance"
        )

    numeric, _ = _encoded(model, rows)
    with torch.no_grad():
        terms = contributions(torch.tensor(numeric, dtype=torch.float32))
    return np.asarray(terms.numpy(), dtype=float)


# ======================================================= capability validation

def gradient_reachable_features(model: Any, rows: pd.DataFrame) -> tuple[bool, ...]:
    """Which feature slots actually carry a derivative in this architecture.

    The check behind the declaration. A feature routed through an embedding
    table has a numeric-input gradient of exactly zero at every row, which is
    why the two tokenising architectures are excluded from gradient methods
    rather than reported with nine zeros.

    A genuinely uninformative feature could also produce a zero gradient at one
    row by coincidence, so the caller passes several rows and this reports a
    slot as reachable if any of them moved.
    """
    gradients = np.abs(input_gradient(model, rows))
    return tuple(bool(v) for v in (gradients.max(axis=0) > 0.0))
