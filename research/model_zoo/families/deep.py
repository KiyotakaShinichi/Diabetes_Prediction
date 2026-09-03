"""Ten neural architectures: three reused from Track K, seven new.

The three Track K models are **imported, not copied**. `research.track_k.deep.
models` remains their only definition, so the zoo cannot drift away from the
evidence Track K published; if those classes change, both tracks change
together or the tests fail.

The seven new architectures live in `research.model_zoo.deep.architectures`,
each encoding a different structural hypothesis about where the remaining
signal might be. What the zoo adds beyond Track K is the *comparison*: ten
architectures under one budget, scored on the same rows, with parameter counts
and training times attached, so "bigger did not help" becomes a measurement
across ten points rather than an impression from three.

Every model here is CPU-only, epoch-bounded, early-stopped and seeded. None
requires CUDA.
"""
from __future__ import annotations

from typing import Any

from research.model_zoo.contracts import (
    Capabilities,
    Family,
    Framework,
    Preprocessing,
    ProbabilityBehavior,
    ResourceClass,
)
from research.model_zoo.registry import ModelSpec, register

#: Epoch ceiling for every deep model in the zoo. Track K measured both of its
#: networks reaching within 0.002 of their final validation ROC-AUC in a single
#: epoch, so a long budget here would buy noise; early stopping usually ends a
#: run before this.
MAX_EPOCHS: int = 40
BATCH_SIZE: int = 128


def _torch_adapter(spec: ModelSpec, factory: Any, **overrides: Any) -> Any:
    from research.model_zoo.adapters.torch_adapter import TorchAdapter

    return TorchAdapter(
        spec.model_id,
        factory,
        capabilities=spec.capabilities,
        probability_behavior=spec.probability_behavior,
        seed=spec.seed,
        max_epochs=overrides.pop("max_epochs", MAX_EPOCHS),
        batch_size=overrides.pop("batch_size", BATCH_SIZE),
        **overrides,
    )


# ------------------------------------------------- Track K models, reused

def build_mlp(*, spec: ModelSpec, hidden_dims: tuple[int, ...] = (128, 64), **kw: Any) -> Any:
    from research.track_k.deep.models import MLPConfig, TabularMLP

    config = MLPConfig(hidden_dims=tuple(hidden_dims), dropout=0.1)
    return _torch_adapter(
        spec, lambda n_features, vocab: TabularMLP(n_features, config), **kw
    )


def build_tabular_resnet(*, spec: ModelSpec, n_blocks: int = 3, **kw: Any) -> Any:
    from research.track_k.deep.models import TabularResNet, TabularResNetConfig

    config = TabularResNetConfig(d_hidden=64, n_blocks=n_blocks, dropout=0.1)
    return _torch_adapter(
        spec, lambda n_features, vocab: TabularResNet(n_features, config), **kw
    )


def build_ft_transformer(*, spec: ModelSpec, d_token: int = 16, n_blocks: int = 2, **kw: Any) -> Any:
    from research.track_k.deep.models import FTTransformer, FTTransformerConfig

    config = FTTransformerConfig(d_token=d_token, n_blocks=n_blocks, n_heads=4)
    return _torch_adapter(
        spec, lambda n_features, vocab: FTTransformer(vocab, config), **kw
    )


# ----------------------------------------------------- new architectures

def _new(architecture: str, **config_kw: Any) -> Any:
    """Factory for the zoo's own architectures, by name."""
    from research.model_zoo.deep import architectures as arch

    config = arch.ArchitectureConfig(**config_kw)
    builders = {
        "tab_transformer": lambda n, v: arch.TabTransformer(v, config),
        "deep_cross": lambda n, v: arch.DeepCrossNetwork(n, config),
        "wide_and_deep": lambda n, v: arch.WideAndDeep(n, config),
        "gated_residual_mlp": lambda n, v: arch.GatedResidualMLP(n, config),
        "feature_token_mixer": lambda n, v: arch.FeatureTokenMixer(n, config),
        "neural_additive": lambda n, v: arch.NeuralAdditiveModel(n, config),
        "feature_cnn": lambda n, v: arch.FeatureCNN(n, config),
    }
    if architecture not in builders:
        raise ValueError(f"unknown architecture: {architecture!r}")
    return builders[architecture]


def build_tab_transformer(*, spec: ModelSpec, d_token: int = 16, n_blocks: int = 2, **kw: Any) -> Any:
    return _torch_adapter(spec, _new("tab_transformer", d_token=d_token, n_blocks=n_blocks), **kw)


def build_deep_cross(*, spec: ModelSpec, n_blocks: int = 2, d_hidden: int = 64, **kw: Any) -> Any:
    return _torch_adapter(spec, _new("deep_cross", n_blocks=n_blocks, d_hidden=d_hidden), **kw)


def build_wide_and_deep(*, spec: ModelSpec, d_hidden: int = 64, **kw: Any) -> Any:
    return _torch_adapter(spec, _new("wide_and_deep", d_hidden=d_hidden), **kw)


def build_gated_residual_mlp(*, spec: ModelSpec, n_blocks: int = 3, d_hidden: int = 64, **kw: Any) -> Any:
    return _torch_adapter(
        spec, _new("gated_residual_mlp", n_blocks=n_blocks, d_hidden=d_hidden), **kw
    )


def build_feature_token_mixer(*, spec: ModelSpec, d_token: int = 16, n_blocks: int = 2, **kw: Any) -> Any:
    return _torch_adapter(
        spec, _new("feature_token_mixer", d_token=d_token, n_blocks=n_blocks), **kw
    )


def build_neural_additive(*, spec: ModelSpec, d_hidden: int = 64, **kw: Any) -> Any:
    return _torch_adapter(spec, _new("neural_additive", d_hidden=d_hidden), **kw)


def build_feature_cnn(*, spec: ModelSpec, d_token: int = 16, **kw: Any) -> Any:
    return _torch_adapter(spec, _new("feature_cnn", d_token=d_token), **kw)


# =========================================================== registrations

_DEEP_CAPABILITIES = Capabilities(
    supports_predict_proba=True,
    supports_calibration=True,
    supports_feature_importance=False,
    supports_serialization=True,
    requires_scaling=True,
)


def _deep_spec(
    model_id: str,
    display_name: str,
    build: Any,
    rationale: str,
    *,
    resource_class: ResourceClass = ResourceClass.MODERATE,
    supports_feature_importance: bool = False,
    default_config: dict[str, Any] | None = None,
) -> ModelSpec:
    capabilities = (
        _DEEP_CAPABILITIES
        if not supports_feature_importance
        else Capabilities(
            supports_predict_proba=True,
            supports_calibration=True,
            supports_feature_importance=True,
            supports_serialization=True,
            requires_scaling=True,
        )
    )
    return ModelSpec(
        model_id=model_id,
        display_name=display_name,
        family=Family.DEEP,
        framework=Framework.TORCH,
        build=build,
        preprocessing=Preprocessing.MODEL_NATIVE,
        probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
        capabilities=capabilities,
        resource_class=resource_class,
        default_config=default_config or {},
        rationale=rationale,
    )


register(_deep_spec(
    "mlp", "MLP (128-64)", build_mlp,
    "Track K's baseline network, imported rather than copied so the two tracks "
    "cannot drift apart.",
    default_config={"hidden_dims": (128, 64)},
))

register(_deep_spec(
    "tabular_resnet", "Tabular ResNet (3 blocks)", build_tabular_resnet,
    "Track K's residual tower. It finished below the smaller MLP there; the zoo "
    "re-tests that at a fifth of the training budget.",
    default_config={"n_blocks": 3},
))

register(_deep_spec(
    "ft_transformer", "FT-Transformer (2 blocks)", build_ft_transformer,
    "Track K's attention model, which tokenises every feature including the "
    "continuous ones. The reference point for TabTransformer below.",
    resource_class=ResourceClass.HEAVY,
    default_config={"d_token": 16, "n_blocks": 2},
))

register(_deep_spec(
    "tab_transformer", "TabTransformer", build_tab_transformer,
    "Attention over the categorical features only, with BMI and PhysHlth "
    "bypassing it. Against FT-Transformer this isolates whether tokenising the "
    "continuous features was worth anything.",
    resource_class=ResourceClass.HEAVY,
    default_config={"d_token": 16, "n_blocks": 2},
))

register(_deep_spec(
    "deep_cross", "Deep & Cross Network", build_deep_cross,
    "Computes bounded-degree feature interactions explicitly rather than "
    "approximating them. If the plateau is an MLP failing to find "
    "interactions, this should beat it.",
    default_config={"n_blocks": 2, "d_hidden": 64},
))

register(_deep_spec(
    "wide_and_deep", "Wide & Deep", build_wide_and_deep,
    "Keeps an explicit linear term so the deep branch only learns the residual "
    "over a logistic-regression-shaped model - the one Track K found hard to beat.",
    default_config={"d_hidden": 64},
))

register(_deep_spec(
    "gated_residual_mlp", "Gated Residual MLP", build_gated_residual_mlp,
    "Gated linear units let a block learn to close. If near-linear is the right "
    "answer, this architecture can say so explicitly.",
    default_config={"n_blocks": 3, "d_hidden": 64},
))

register(_deep_spec(
    "feature_token_mixer", "Feature-Token MLP-Mixer", build_feature_token_mixer,
    "All-to-all feature mixing without attention. Against the two transformers "
    "it separates content-dependent weighting from mere feature interaction.",
    default_config={"d_token": 16, "n_blocks": 2},
))

register(_deep_spec(
    "neural_additive", "Neural Additive Model", build_neural_additive,
    "One shape function per feature, summed - structurally incapable of "
    "representing any interaction. If it matches the unconstrained networks, "
    "interactions are not where the remaining signal lives.",
    supports_feature_importance=True,
    default_config={"d_hidden": 64},
))

register(_deep_spec(
    "feature_cnn", "Feature CNN (experimental control)", build_feature_cnn,
    "EXPERIMENTAL INDUCTIVE-BIAS CONTROL. A 1D convolution assumes adjacent "
    "positions are related, but the feature order here is arbitrary - permuting "
    "the columns would change its predictions and nothing about the problem. "
    "Included as a case where the architecture demonstrably does not match the "
    "data, and explicitly not a production candidate.",
    default_config={"d_token": 16},
))
