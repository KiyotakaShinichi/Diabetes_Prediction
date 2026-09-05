"""The XAI contracts: records, capabilities and the method registry.

Track L caught a model overclaiming a capability it did not have, and fixed the
declaration rather than the test. Track M inherits that arrangement: every
capability declared here is asserted against the constructed model, so a
profile that drifts from reality fails rather than quietly producing a column
of numbers nobody can interpret.

The normalisation tests matter more than they look. Cross-model comparison is
only meaningful because raw attributions are converted to a common share-of-
magnitude scale, so the conversion is tested against inputs whose answer is
arithmetic.
"""
import numpy as np
import pytest

from research.model_zoo.contracts import (
    Family,
    Framework,
    ProbabilityBehavior,
    ResearchStatus,
)
from research.model_zoo.registry import REGISTRY
from research.xai import capabilities as caps
from research.xai.capabilities import XaiCapability
from research.xai.contracts import (
    BaselineStrategy,
    Determinism,
    ExplanationRecord,
    RunStatus,
    Scope,
    explanation_id,
    hash_payload,
    normalize_attributions,
    rank_features,
)
from research.xai.registry import METHODS

pytest.importorskip("torch", reason="the zoo's deep models need PyTorch")

FEATURES = ("GenHlth", "HighBP", "BMI", "HighChol", "Age")


# ======================================================== normalisation

def test_normalisation_produces_a_share_of_total_magnitude():
    shares = normalize_attributions(np.array([2.0, 1.0, 1.0]))

    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] == pytest.approx(0.5)


def test_normalisation_uses_magnitude_so_sign_does_not_change_the_share():
    """A negative coefficient carries as much weight as a positive one."""
    positive = normalize_attributions(np.array([3.0, 1.0]))
    negative = normalize_attributions(np.array([-3.0, 1.0]))

    assert np.allclose(positive, negative)


def test_an_all_zero_attribution_becomes_uniform_rather_than_dividing_by_zero():
    """A model that used nothing must not crash the harness."""
    shares = normalize_attributions(np.zeros(4))

    assert shares.sum() == pytest.approx(1.0)
    assert len(set(np.round(shares, 12))) == 1, "the degenerate case must be detectable"


def test_non_finite_attributions_degrade_to_uniform():
    shares = normalize_attributions(np.array([np.inf, 1.0]))

    assert np.isfinite(shares).all()
    assert shares.sum() == pytest.approx(1.0)


# ============================================================== ranking

def test_ranking_orders_by_magnitude_not_by_signed_value():
    ranking = rank_features(FEATURES[:3], np.array([0.1, -0.9, 0.5]))

    assert ranking == ("HighBP", "BMI", "GenHlth")


def test_ties_are_broken_deterministically_by_name():
    """An L1 fit that zeroed several features must still rank reproducibly.

    Without a deterministic tie-break the stability study would measure the
    sort order rather than the model.
    """
    first = rank_features(FEATURES[:4], np.array([1.0, 0.0, 0.0, 0.0]))
    second = rank_features(FEATURES[:4], np.array([1.0, 0.0, 0.0, 0.0]))

    assert first == second
    assert first[0] == "GenHlth"
    assert list(first[1:]) == sorted(first[1:])


def test_ranking_covers_every_feature_exactly_once():
    ranking = rank_features(FEATURES, np.array([5.0, 4.0, 3.0, 2.0, 1.0]))

    assert set(ranking) == set(FEATURES)
    assert len(ranking) == len(FEATURES)


# ====================================================== the record type

@pytest.fixture
def record():
    raw = np.array([0.5, -0.3, 0.2, 0.0, 0.0])
    return ExplanationRecord(
        explanation_id=explanation_id("logistic_l2", "coefficients", sample_id=None, seed=None),
        model_id="logistic_l2",
        model_family="linear",
        method="coefficients",
        method_version="1.0.0",
        scope=Scope.GLOBAL,
        feature_names=FEATURES,
        raw_attributions=tuple(raw),
        normalized_attributions=tuple(normalize_attributions(raw)),
        ranking=rank_features(FEATURES, raw),
        baseline_reference=BaselineStrategy.NOT_APPLICABLE.value,
    )


def test_a_record_is_immutable(record):
    """Evidence that can be edited after the fact is not evidence."""
    with pytest.raises((AttributeError, TypeError)):
        record.model_id = "something_else"  # type: ignore[misc]


def test_a_record_serialises_to_plain_json_data(record):
    import json

    payload = record.as_dict()
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["model_id"] == "logistic_l2"
    assert round_tripped["ranking"][0] == "GenHlth"
    assert isinstance(round_tripped["normalized_attributions"][0], float)


def test_a_record_reports_ranks_and_shares_per_feature(record):
    assert record.top_feature == "GenHlth"
    assert record.rank_of("GenHlth") == 1
    assert record.rank_of("HighBP") == 2
    assert record.attribution_for("GenHlth") == pytest.approx(0.5)


def test_records_carry_the_provenance_needed_to_reproduce_them(record):
    payload = record.as_dict()

    for field in ("source_sha", "model_config_hash", "training_subset_hash",
                  "data_fingerprint", "created_at", "schema_version"):
        assert field in payload


def test_explanation_ids_are_stable_and_distinct():
    a = explanation_id("mlp", "gradient", sample_id=7, seed=1)
    b = explanation_id("mlp", "gradient", sample_id=7, seed=1)
    c = explanation_id("mlp", "gradient", sample_id=8, seed=1)

    assert a == b
    assert a != c


def test_payload_hashing_is_order_independent():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


# =================================================== capability profiles

ACTIVE = [s.model_id for s in REGISTRY.active()]


def test_every_registered_model_has_a_profile():
    profiles = caps.build_profiles()

    assert set(profiles) == set(REGISTRY.ids())


@pytest.mark.parametrize("model_id", ACTIVE)
def test_a_hard_label_model_claims_no_score_based_method(model_id):
    """Permutation, occlusion and PDP all need a continuous score."""
    spec = REGISTRY.get(model_id)
    profile = caps.profile_for(spec)

    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        for capability in (
            XaiCapability.PERMUTATION_IMPORTANCE,
            XaiCapability.OCCLUSION_COMPATIBLE,
            XaiCapability.PARTIAL_DEPENDENCE_COMPATIBLE,
        ):
            assert not profile.supports(capability)
        assert profile.exclusions, "an excluded model must say why"


@pytest.mark.parametrize("model_id", ACTIVE)
def test_no_model_outside_the_deep_families_claims_gradients(model_id):
    spec = REGISTRY.get(model_id)
    profile = caps.profile_for(spec)

    if spec.framework is not Framework.TORCH:
        assert not profile.supports(XaiCapability.GRADIENT_COMPATIBLE)


DEEP = [
    model_id
    for model_id in ACTIVE
    if REGISTRY.get(model_id).framework is Framework.TORCH
]


@pytest.mark.parametrize("model_id", DEEP)
def test_a_torch_model_claims_gradients_unless_it_reaches_features_by_lookup(model_id):
    """Being a neural network is not the same as being differentiable in x.

    `ft_transformer` and `tab_transformer` tokenise discrete features through an
    embedding table, and the derivative of a table index does not exist - an
    input gradient returns exactly 0.0 for those slots. Reported as a number,
    that is indistinguishable from a model that ignores the feature, so the two
    are excluded and told to say why. `tests/test_xai_deep.py` measures which
    slots actually carry a derivative, so this declaration cannot drift away
    from the architectures.
    """
    spec = REGISTRY.get(model_id)
    profile = caps.profile_for(spec)

    excluded = spec.model_id in caps._EMBEDDING_INPUT_MODELS
    assert profile.supports(XaiCapability.GRADIENT_COMPATIBLE) is not excluded
    if excluded:
        assert any("embedding lookup" in reason for reason in profile.exclusions)


@pytest.mark.parametrize("model_id", ACTIVE)
def test_gradient_capabilities_travel_together(model_id):
    """A differentiable model supports all three gradient methods or none."""
    profile = caps.profile_for(REGISTRY.get(model_id))

    gradient_caps = {
        profile.supports(XaiCapability.GRADIENT_COMPATIBLE),
        profile.supports(XaiCapability.GRADIENT_X_INPUT_COMPATIBLE),
        profile.supports(XaiCapability.INTEGRATED_GRADIENTS_COMPATIBLE),
    }
    assert len(gradient_caps) == 1


@pytest.mark.parametrize("model_id", ACTIVE)
def test_a_model_never_claims_both_coefficients_and_native_importance(model_id):
    """They are different quantities; conflating them would mix scales."""
    profile = caps.profile_for(REGISTRY.get(model_id))

    assert not (
        profile.supports(XaiCapability.NATIVE_COEFFICIENTS)
        and profile.supports(XaiCapability.NATIVE_FEATURE_IMPORTANCE)
    )


def test_the_capability_matrix_covers_every_model_and_capability():
    rows = caps.capability_matrix()

    assert {row["model_id"] for row in rows} == set(REGISTRY.ids())
    for row in rows:
        for capability in XaiCapability:
            assert capability.value in row


def test_models_supporting_returns_only_active_models():
    for capability in XaiCapability:
        for model_id in caps.models_supporting(capability):
            assert REGISTRY.get(model_id).effective_status() is ResearchStatus.ACTIVE


def test_every_family_has_at_least_one_explainable_model():
    """A family with no available method would be invisible in the analysis."""
    for family in Family:
        members = [s.model_id for s in REGISTRY.by_family(family)
                   if s.effective_status() is ResearchStatus.ACTIVE]
        if not members:
            continue
        explainable = [
            m for m in members
            if caps.profile_for(REGISTRY.get(m)).capabilities
        ]
        assert explainable, f"no model in {family.value} supports any XAI method"


# ==================================================== the method registry

def test_the_method_registry_is_populated():
    assert len(METHODS) >= 8


def test_method_ids_are_unique():
    assert len(METHODS.ids()) == len(set(METHODS.ids()))


def test_a_duplicate_method_registration_is_refused():
    from research.xai.registry import MethodRegistry

    registry = MethodRegistry()
    spec = METHODS.get("coefficients")
    registry.register(spec)

    with pytest.raises(ValueError, match="duplicate method_id"):
        registry.register(spec)


def test_an_unknown_method_names_what_is_registered():
    with pytest.raises(KeyError, match="unknown method_id"):
        METHODS.get("telepathy")


def test_every_method_declares_what_it_measures_and_how_it_misleads():
    """A method card with no stated limitation is a marketing document."""
    for method in METHODS:
        assert len(method.measures) > 30, f"{method.method_id} does not say what it measures"
        assert method.failure_modes, f"{method.method_id} lists no failure modes"
        for mode in method.failure_modes:
            assert len(mode) > 20


def test_every_method_declares_a_baseline_strategy():
    """"Compared to what?" must never be implicit."""
    for method in METHODS:
        assert isinstance(method.baseline_strategy, BaselineStrategy)


def test_methods_that_need_a_reference_declare_a_real_one():
    """Occlusion and integrated gradients are meaningless without a baseline."""
    for method_id in ("occlusion", "integrated_gradients", "gradient_x_input", "tree_shap"):
        method = METHODS.get(method_id)
        assert method.baseline_strategy is not BaselineStrategy.NOT_APPLICABLE


def test_every_method_declares_its_determinism():
    for method in METHODS:
        assert isinstance(method.determinism, Determinism)


def test_permutation_importance_is_declared_stochastic():
    """The seed study reads this flag; a wrong value fabricates variance."""
    assert METHODS.get("permutation_importance").determinism is Determinism.STOCHASTIC


def test_gradient_methods_are_declared_deterministic():
    for method_id in ("gradient", "gradient_x_input", "integrated_gradients"):
        assert METHODS.get(method_id).determinism is Determinism.DETERMINISTIC


def test_partial_dependence_flags_the_causal_misreading():
    """It is the method most often read as a causal effect. It is not one."""
    method = METHODS.get("partial_dependence")

    assert method.causal_reading_invalid
    assert any("causal" in mode.lower() for mode in method.failure_modes)


def test_the_correlation_failure_mode_is_documented_where_it_applies():
    """Permutation and PDP both mislead under correlated features."""
    for method_id in ("permutation_importance", "partial_dependence"):
        modes = " ".join(METHODS.get(method_id).failure_modes).lower()
        assert "correlat" in modes


def test_impurity_bias_is_documented_for_native_importance():
    """Track L's tree family needs this caveat attached to its numbers."""
    modes = " ".join(METHODS.get("native_importance").failure_modes).lower()

    assert "impurity" in modes
    assert "bias" in modes


def test_every_method_requires_a_capability_some_model_provides():
    """A method no model can run would be dead weight in the registry."""
    for method in METHODS:
        supported = caps.models_supporting(method.required_capability)
        assert supported, f"{method.method_id} requires a capability no active model has"


def test_optional_methods_declare_their_dependency():
    for method in METHODS:
        if method.method_id == "tree_shap":
            assert method.optional_dependency == "shap"
        if method.optional_dependency is None:
            assert method.is_available()


def test_an_unavailable_optional_method_is_detected_not_assumed(monkeypatch):
    """Absence must be observable, so the runner can record it as skipped."""
    import importlib.util

    method = METHODS.get("tree_shap")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    assert method.is_available() is False


def test_the_registry_serialises_for_the_method_cards():
    for method in METHODS:
        payload = method.as_dict()
        assert payload["method_id"] == method.method_id
        assert payload["failure_modes"]
        assert payload["required_capability"] in {c.value for c in XaiCapability}


def test_run_status_covers_every_way_a_pair_can_fail():
    """Failures stay in the table; the vocabulary has to be complete."""
    values = {status.value for status in RunStatus}

    assert {"success", "unsupported", "optional_dependency_missing",
            "resource_limit", "numerical_failure", "invalid_capability"} <= values
