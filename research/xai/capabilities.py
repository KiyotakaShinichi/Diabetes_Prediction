"""Which explanations each model can genuinely provide.

Track L caught a real defect of exactly this kind: `hist_gradient_boosting` was
registered as having native feature importance, and scikit-learn's histogram
implementation does not expose one. The declaration was wrong, the test caught
it, and the declaration was fixed rather than the test weakened.

Track M inherits that discipline and extends it. A capability here is derived
from the Track L spec where the spec already knows the answer, refined where
XAI needs a finer distinction than prediction did, and then **validated against
the constructed model** by `tests/test_xai_capabilities.py`. A model that
claims gradients must actually produce them; a model that claims none must
raise when asked.

The derivation matters. Writing a second hand-maintained table beside the Track
L registry would guarantee the two drift apart, so the only facts stated here
are the ones the model registry cannot already answer:

* whether a model exposes coefficients as opposed to some other importance;
* whether it is a torch module, and therefore differentiable end to end;
* whether interaction analysis is meaningful for it.

Everything else - probabilities, native importance, scaling - is read from the
Track L spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from research.model_zoo.contracts import Family, Framework, ProbabilityBehavior
from research.model_zoo.registry import REGISTRY, ModelSpec


class XaiCapability(StrEnum):
    """One thing a model can be asked for, explanation-wise."""

    NATIVE_COEFFICIENTS = "native_coefficients"
    NATIVE_FEATURE_IMPORTANCE = "native_feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    OCCLUSION_COMPATIBLE = "occlusion_compatible"
    PARTIAL_DEPENDENCE_COMPATIBLE = "partial_dependence_compatible"
    ICE_COMPATIBLE = "ice_compatible"
    GRADIENT_COMPATIBLE = "gradient_compatible"
    GRADIENT_X_INPUT_COMPATIBLE = "gradient_x_input_compatible"
    INTEGRATED_GRADIENTS_COMPATIBLE = "integrated_gradients_compatible"
    SHAP_COMPATIBLE = "shap_compatible"
    INTERACTION_ANALYSIS_COMPATIBLE = "interaction_analysis_compatible"


#: Models whose fitted estimator exposes a linear coefficient vector in the
#: transformed feature space. Derived from the family plus the estimator type
#: rather than assumed from the family alone, because ``distance`` contains
#: both a centroid classifier (no coefficients) and k-NN (no coefficients),
#: while ``kernel`` contains a linear SVM (coefficients) and an RBF SVM (none:
#: its dual coefficients live in kernel space, not feature space).
_COEFFICIENT_MODELS: frozenset[str] = frozenset({
    "logistic_l2",
    "logistic_l1",
    "logistic_elasticnet",
    "sgd_logistic",
    "sgd_modified_huber",
    "lda",
    "linear_svm",
})

#: Tree-ensemble models a tree-aware SHAP implementation can explain exactly.
#: Deliberately narrow: SHAP is optional here, and claiming support for a model
#: its tree explainer cannot consume would turn an optional dependency into a
#: runtime failure.
_TREE_SHAP_MODELS: frozenset[str] = frozenset({
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
})

#: Torch architectures that route SOME feature through an embedding table
#: instead of through the differentiable numeric input.
#:
#: This is the correction that matters most in this file. "It is a neural
#: network, so it has gradients" is true of the parameters and false of the
#: inputs. Both models below tokenise discrete features by table lookup - all
#: nine of them for `ft_transformer`, nine of ten for `tab_transformer` - and
#: the derivative of a table index does not exist. Asking either for an input
#: gradient returns exactly 0.0 for those slots, which in the results table
#: reads as "the model ignores this feature" rather than "this method cannot
#: see it". Every other torch model in the zoo deletes ``levels`` on the first
#: line of ``forward`` and treats the ordinal codes as numbers, so its gradient
#: covers the whole model.
#:
#: `tests/test_xai_deep.py` builds every torch model and measures which slots
#: actually carry a derivative, so this set cannot drift away from the
#: architectures without a test failing.
_EMBEDDING_INPUT_MODELS: frozenset[str] = frozenset({
    "ft_transformer",
    "tab_transformer",
})


@dataclass(frozen=True, slots=True)
class XaiProfile:
    """Everything the XAI layer needs to know about one model."""

    model_id: str
    family: Family
    capabilities: frozenset[XaiCapability]
    #: Why certain capabilities are absent. Recorded so a gap in the results
    #: table can be read as a property of the model rather than an oversight.
    exclusions: tuple[str, ...] = ()

    def supports(self, capability: XaiCapability) -> bool:
        return capability in self.capabilities

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family.value,
            "capabilities": sorted(c.value for c in self.capabilities),
            "exclusions": list(self.exclusions),
        }


def profile_for(spec: ModelSpec) -> XaiProfile:
    """Derive one model's XAI capabilities from its Track L specification."""
    capabilities: set[XaiCapability] = set()
    exclusions: list[str] = []

    has_scores = spec.probability_behavior is not ProbabilityBehavior.HARD_LABELS_ONLY

    # Model-agnostic methods all need a ranking score to measure a change in.
    # A hard-label model gives only 0/1, so a permuted feature either flips a
    # label or does nothing - the measurement exists but is far coarser, and
    # occlusion/PDP become step functions. Excluded rather than reported as a
    # comparable number.
    if has_scores:
        capabilities.update({
            XaiCapability.PERMUTATION_IMPORTANCE,
            XaiCapability.OCCLUSION_COMPATIBLE,
            XaiCapability.PARTIAL_DEPENDENCE_COMPATIBLE,
            XaiCapability.ICE_COMPATIBLE,
            XaiCapability.INTERACTION_ANALYSIS_COMPATIBLE,
        })
    else:
        exclusions.append(
            "model-agnostic methods need a continuous ranking score; this model "
            "emits hard labels only, so permutation, occlusion, PDP/ICE and "
            "interaction analysis are undefined rather than merely unmeasured"
        )

    if spec.model_id in _COEFFICIENT_MODELS:
        capabilities.add(XaiCapability.NATIVE_COEFFICIENTS)

    if spec.capabilities.supports_feature_importance and (
        spec.model_id not in _COEFFICIENT_MODELS
    ):
        capabilities.add(XaiCapability.NATIVE_FEATURE_IMPORTANCE)

    if spec.framework is Framework.TORCH:
        if spec.model_id in _EMBEDDING_INPUT_MODELS:
            exclusions.append(
                "gradient methods are excluded: this architecture reaches its "
                "discrete features through an embedding lookup rather than "
                "through the differentiable numeric input, so an input gradient "
                "is exactly zero for them - which would read as 'the model "
                "ignores this feature' rather than 'this method cannot see it'"
            )
        else:
            # Every other torch model discards the ordinal codes and treats
            # them as numbers, so its logit is differentiable with respect to
            # every feature and a gradient describes the whole model.
            capabilities.update({
                XaiCapability.GRADIENT_COMPATIBLE,
                XaiCapability.GRADIENT_X_INPUT_COMPATIBLE,
                XaiCapability.INTEGRATED_GRADIENTS_COMPATIBLE,
            })

    if spec.model_id in _TREE_SHAP_MODELS:
        capabilities.add(XaiCapability.SHAP_COMPATIBLE)
    elif has_scores:
        exclusions.append(
            "tree-aware SHAP is not applicable; a model-agnostic SHAP variant "
            "would cost far more than the zoo's budget allows and is not run"
        )

    if not spec.capabilities.supports_feature_importance and (
        spec.model_id not in _COEFFICIENT_MODELS
    ):
        exclusions.append("no native importance of any kind is exposed by this estimator")

    return XaiProfile(
        model_id=spec.model_id,
        family=spec.family,
        capabilities=frozenset(capabilities),
        exclusions=tuple(exclusions),
    )


def build_profiles() -> dict[str, XaiProfile]:
    """Capability profiles for every registered model, keyed by id."""
    return {spec.model_id: profile_for(spec) for spec in REGISTRY}


def capability_matrix() -> list[dict[str, Any]]:
    """Machine-readable XAI capability matrix, one row per model."""
    rows = []
    for spec in REGISTRY:
        profile = profile_for(spec)
        row: dict[str, Any] = {
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "family": spec.family.value,
            "framework": spec.framework.value,
            "status": spec.effective_status().value,
        }
        for capability in XaiCapability:
            row[capability.value] = profile.supports(capability)
        row["exclusions"] = list(profile.exclusions)
        rows.append(row)
    return rows


def models_supporting(capability: XaiCapability) -> list[str]:
    """Active models that genuinely provide a given capability."""
    from research.model_zoo.contracts import ResearchStatus

    return [
        spec.model_id
        for spec in REGISTRY
        if spec.effective_status() is ResearchStatus.ACTIVE
        and profile_for(spec).supports(capability)
    ]
