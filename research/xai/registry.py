"""The explanation-method registry: what each method needs, and what it gives.

The pairing problem in a cross-family XAI lab is combinatorial. Twenty-nine
models times a dozen methods is several hundred (model, method) pairs, most of
which are invalid — you cannot take a gradient through a random forest, and a
coefficient does not exist for an RBF kernel. Resolving that pairwise in the
runner would be a wall of conditionals that silently rots as models are added.

So each method declares the **capability** it requires, and the runner asks the
model's profile whether it has it. A pair is either supported, or it is recorded
as `UNSUPPORTED` with the missing capability named. Adding a method is one
registration; adding a model is one profile; neither touches the runner.

Each method also declares the three things that make its output interpretable
at all: whether it is local or global, what it measures against (its baseline
strategy), and whether repeating it gives the same answer. That last flag is
what stops the seed study from fabricating variance for a deterministic method.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from research.xai.capabilities import XaiCapability
from research.xai.contracts import BaselineStrategy, Determinism, Scope


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One explanation method and the contract it operates under."""

    method_id: str
    display_name: str
    version: str
    #: The single capability a model must have for this method to apply.
    required_capability: XaiCapability
    scope: Scope
    determinism: Determinism
    baseline_strategy: BaselineStrategy
    #: Rough cost class at the zoo's budget, used by the plan command.
    runtime_class: str
    #: What the method actually measures, in one sentence, for the method card.
    measures: str
    #: How it can mislead. Every method here has a real failure mode and the
    #: card states it; a method card that lists no limitation is a marketing
    #: document.
    failure_modes: tuple[str, ...]
    #: True when a causal reading is specifically tempting and specifically
    #: wrong - PDP being the classic case.
    causal_reading_invalid: bool = True
    optional_dependency: str | None = None
    builder: Callable[..., Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "version": self.version,
            "required_capability": self.required_capability.value,
            "scope": self.scope.value,
            "determinism": self.determinism.value,
            "baseline_strategy": self.baseline_strategy.value,
            "runtime_class": self.runtime_class,
            "measures": self.measures,
            "failure_modes": list(self.failure_modes),
            "causal_reading_invalid": self.causal_reading_invalid,
            "optional_dependency": self.optional_dependency,
        }

    def is_available(self) -> bool:
        """Whether this method's optional dependency can be imported."""
        if self.optional_dependency is None:
            return True
        import importlib.util

        return importlib.util.find_spec(self.optional_dependency) is not None


class MethodRegistry:
    """An ordered collection of method specs, addressable by id."""

    def __init__(self) -> None:
        self._methods: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec) -> MethodSpec:
        if spec.method_id in self._methods:
            raise ValueError(f"duplicate method_id: {spec.method_id!r}")
        self._methods[spec.method_id] = spec
        return spec

    def __contains__(self, method_id: object) -> bool:
        return method_id in self._methods

    def __len__(self) -> int:
        return len(self._methods)

    def __iter__(self) -> Iterator[MethodSpec]:
        return iter(self._methods.values())

    def get(self, method_id: str) -> MethodSpec:
        if method_id not in self._methods:
            raise KeyError(
                f"unknown method_id {method_id!r}; registered: {sorted(self._methods)}"
            )
        return self._methods[method_id]

    def ids(self) -> list[str]:
        return list(self._methods)

    def by_scope(self, scope: Scope) -> list[MethodSpec]:
        return [m for m in self if m.scope is scope]

    def available(self) -> list[MethodSpec]:
        return [m for m in self if m.is_available()]


METHODS = MethodRegistry()


def register(spec: MethodSpec) -> MethodSpec:
    return METHODS.register(spec)


# ===================================================== method declarations

register(MethodSpec(
    method_id="coefficients",
    display_name="Standardized coefficient magnitude",
    version="1.0.0",
    required_capability=XaiCapability.NATIVE_COEFFICIENTS,
    scope=Scope.GLOBAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.NOT_APPLICABLE,
    runtime_class="instant",
    measures=(
        "The weight each standardised feature carries in the model's linear "
        "decision function."
    ),
    failure_modes=(
        "Only comparable across features because the pipeline standardises "
        "them first; on raw features the magnitudes would reflect units.",
        "Correlated features share credit arbitrarily - an L1 penalty may zero "
        "one of a correlated pair and load the other, which says more about "
        "the penalty than about the features.",
        "Says nothing about non-linear dependence, because the model has none.",
    ),
))

register(MethodSpec(
    method_id="native_importance",
    display_name="Native feature importance",
    version="1.0.0",
    required_capability=XaiCapability.NATIVE_FEATURE_IMPORTANCE,
    scope=Scope.GLOBAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.NOT_APPLICABLE,
    runtime_class="instant",
    measures="The model's own internal importance statistic, as its library defines it.",
    failure_modes=(
        "For tree ensembles this is mean impurity decrease, which is BIASED "
        "toward high-cardinality and continuous features: BMI can outrank a "
        "binary flag simply by offering more split points.",
        "Computed on training data, so it reflects what the model used to fit, "
        "not what it needs to predict held-out rows.",
        "Not comparable in units across libraries; only its ranking is used.",
    ),
))

register(MethodSpec(
    method_id="permutation_importance",
    display_name="Permutation importance",
    version="1.0.0",
    required_capability=XaiCapability.PERMUTATION_IMPORTANCE,
    scope=Scope.GLOBAL,
    determinism=Determinism.STOCHASTIC,
    baseline_strategy=BaselineStrategy.NOT_APPLICABLE,
    runtime_class="moderate",
    measures=(
        "How much the model's ranking quality falls when one feature's values "
        "are shuffled, breaking its association with the target."
    ),
    failure_modes=(
        "Scored on the rows the model was fitted to, it measures memorisation: "
        "on labels independent of every feature, a random forest reports about "
        "0.12 ROC-AUC points where the truth is zero. Every call site passes an "
        "evaluation partition the model never saw, drawn from TRAIN.",
        "Shuffling breaks correlation structure, so a feature correlated with "
        "another can look unimportant because its partner still carries the "
        "signal - and both can look unimportant for the same reason.",
        "Permuting can create feature combinations that do not occur in "
        "reality, evaluating the model off its training manifold.",
        "Stochastic: a single shuffle is noisy, so several are averaged and "
        "the seed is recorded.",
    ),
))

register(MethodSpec(
    method_id="occlusion",
    display_name="Feature occlusion",
    version="1.0.0",
    required_capability=XaiCapability.OCCLUSION_COMPATIBLE,
    scope=Scope.LOCAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.TRAINING_MEDIAN,
    runtime_class="cheap",
    measures=(
        "How far one row's score moves when a single feature is replaced by "
        "its training-median value."
    ),
    failure_modes=(
        "A feature already sitting at its baseline value in this row scores "
        "exactly zero, however much the model depends on it - for a binary "
        "feature with a median baseline that is about half the rows, so a "
        "single row's occlusion vector is never read on its own.",
        "It detects interacting features but cannot attribute the effect to "
        "the pair: each member is credited with the whole joint swing "
        "separately, so one effect is reported twice.",
        "The median substitute may be implausible for a particular row, again "
        "moving the input off the data manifold.",
        "Sensitive to the baseline choice, which is why it is declared.",
    ),
))

register(MethodSpec(
    method_id="gradient",
    display_name="Input gradient",
    version="1.0.0",
    required_capability=XaiCapability.GRADIENT_COMPATIBLE,
    scope=Scope.LOCAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.NOT_APPLICABLE,
    runtime_class="cheap",
    measures="Local sensitivity: how the logit responds to an infinitesimal nudge.",
    failure_modes=(
        "Purely local - it describes the slope at one point and says nothing "
        "about what happens a realistic distance away.",
        "Saturates: a feature the model is already certain about has near-zero "
        "gradient despite driving the decision.",
        "Ignores the feature's actual value, so a large slope on a feature "
        "that barely varies is overweighted.",
    ),
))

register(MethodSpec(
    method_id="gradient_x_input",
    display_name="Gradient x input",
    version="1.0.0",
    required_capability=XaiCapability.GRADIENT_X_INPUT_COMPATIBLE,
    scope=Scope.LOCAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.ZERO_STANDARDIZED,
    runtime_class="cheap",
    measures=(
        "Local sensitivity scaled by the encoded feature value - a first-order "
        "estimate of each feature's contribution to the logit."
    ),
    failure_modes=(
        "Inherits the gradient's saturation problem.",
        "The implicit reference is not one thing. The encoder standardises the "
        "five continuous and ordinal features but leaves binary indicators on "
        "their raw 0/1 scale, so zero means 'the training mean' for GenHlth, "
        "BMI, Age, PhysHlth and Education and 'No' for the five binary flags.",
        "Consequently every patient without high blood pressure gets exactly "
        "zero attribution on HighBP, at every model, whatever the model's "
        "reliance on it. Structural zeros, not measured ones.",
        "A row at the encoder's centre gets zero attribution everywhere; note "
        "that the data mean is not that row, because binary means sit near 0.5.",
    ),
))

register(MethodSpec(
    method_id="integrated_gradients",
    display_name="Integrated gradients",
    version="1.0.0",
    required_capability=XaiCapability.INTEGRATED_GRADIENTS_COMPATIBLE,
    scope=Scope.LOCAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.TRAINING_MEDIAN,
    runtime_class="moderate",
    measures=(
        "Each feature's share of the score difference between a training-median "
        "baseline and the actual row, integrated along the straight path."
    ),
    failure_modes=(
        "The attribution is entirely relative to the baseline; a different "
        "baseline gives different answers, and no baseline is neutral.",
        "The straight-line path may traverse implausible inputs.",
        "Riemann approximation with a finite step count introduces error, and "
        "how much depends sharply on the architecture: at 32 midpoint steps "
        "seven of the eight gradient-capable models leave under 0.007 logits "
        "unattributed while feature_token_mixer leaves 0.105. The budget is set "
        "from the worst case and the per-row gap is recorded regardless.",
    ),
))

register(MethodSpec(
    method_id="tree_shap",
    display_name="Tree SHAP",
    version="1.0.0",
    required_capability=XaiCapability.SHAP_COMPATIBLE,
    scope=Scope.LOCAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.TRAINING_MEAN,
    runtime_class="moderate",
    measures=(
        "Each feature's Shapley value for one prediction, computed exactly for "
        "tree ensembles."
    ),
    failure_modes=(
        "Shapley values assume feature independence when marginalising; with "
        "correlated features the attributions are not uniquely determined.",
        "Exact for trees only - the same guarantee does not extend to the "
        "other families in this zoo.",
        "Optional dependency: absent, the pair is recorded as skipped rather "
        "than dropped.",
    ),
    optional_dependency="shap",
))

register(MethodSpec(
    method_id="partial_dependence",
    display_name="Partial dependence",
    version="1.0.0",
    required_capability=XaiCapability.PARTIAL_DEPENDENCE_COMPATIBLE,
    scope=Scope.GLOBAL,
    determinism=Determinism.DETERMINISTIC,
    baseline_strategy=BaselineStrategy.NOT_APPLICABLE,
    runtime_class="moderate",
    measures=(
        "The model's average predicted score as one feature is varied across "
        "its observed range, holding the others at their observed values."
    ),
    failure_modes=(
        "THE classic misreading: a partial-dependence curve is not a causal "
        "effect and not an individual-level effect. It is an average model "
        "response under an intervention on the input, nothing more.",
        "Blind to pure interaction, and measurably so: on an exclusive-or rule "
        "a forest at 0.96 held-out ROC-AUC yields PD ranges whose top two are "
        "inert columns at every seed tried, because the driver's effect is "
        "positive for half the population and negative for the other half. A "
        "flat curve is evidence about the average, not about the model.",
        "With correlated features it averages over combinations that do not "
        "occur, which is why the correlation audit runs beside it.",
        "Evaluated only within the observed feature support; no extrapolation.",
    ),
))
