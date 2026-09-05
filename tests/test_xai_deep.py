"""Gradient explanations, and the claim that a neural network has them.

The central test in this file is `test_the_gradient_declaration_matches_what
_the_architectures_actually_carry`. Track M declares that eight of the zoo's ten
torch models support gradient methods and two do not, and that declaration is
not read off the source: every model is built, fitted and differentiated, and
the feature slots that actually carry a derivative are counted.

That check exists because the failure it guards against is invisible. Asking
`ft_transformer` for an input gradient does not raise - it returns a perfectly
well-formed vector with 0.0 in nine of ten positions, because those features
reach the network through an embedding table rather than through the
differentiable input. Dropped into the cross-family comparison, that vector
would read as a transformer that ignores nine features, and every downstream
agreement, stability and faithfulness number computed from it would be a
measurement of the harness rather than of the model.

The rest of the file holds the gradient methods to the axioms they claim:
integrated gradients must very nearly sum to the score difference it promises,
and the neural additive model's per-feature terms must reconstruct its logit
exactly, because for that architecture the terms *are* the model.
"""
import warnings

import numpy as np
import pytest

from ml_core import feature_contract
from research.model_zoo.contracts import Framework, ResearchStatus
from research.model_zoo.registry import REGISTRY
from research.xai import capabilities as caps
from research.xai import worlds
from research.xai.capabilities import XaiCapability, profile_for
from research.xai.contracts import CapabilityError
from research.xai.explainers import deep
from research.xai.worlds import XaiWorld

pytest.importorskip("torch", reason="the zoo's deep models need PyTorch")

FEATURES = feature_contract.FEATURE_NAMES
INDEX = {name: i for i, name in enumerate(FEATURES)}

#: Two epochs on 240 rows. These tests are about differentiability and the
#: arithmetic of an attribution, neither of which needs a well-fitted model;
#: the one test that does need the model to have learned something says so and
#: trains longer.
SMOKE_EPOCHS = 2


def _torch_ids():
    return [
        spec.model_id
        for spec in REGISTRY
        if spec.framework is Framework.TORCH
        and spec.effective_status() is ResearchStatus.ACTIVE
    ]


def _fit(model_id, X, y, *, max_epochs=SMOKE_EPOCHS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REGISTRY.build(model_id, max_epochs=max_epochs).fit(X, y)


def _encoder_origin(model):
    """The row that encodes to all zeros: training means, but 0 for binaries.

    Built from the fitted standardiser rather than from the data, because the
    two are not the same row - which is the whole point of the tests that use
    this.
    """
    import pandas as pd

    state = model.standardiser
    scaled = set(state.scaled_indices)
    values = {
        name: (state.means[position] if position in scaled else 0.0)
        for position, name in enumerate(state.feature_names)
    }
    return pd.DataFrame([values], columns=list(state.feature_names))


@pytest.fixture(scope="module")
def rows():
    dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=240, seed=1)
    return dataset.X, dataset.y


@pytest.fixture(scope="module")
def deep_models(rows):
    """Every active torch model, fitted once - the roster sweep is the point."""
    X, y = rows
    return {model_id: _fit(model_id, X, y) for model_id in _torch_ids()}


@pytest.fixture(scope="module")
def gradient_models(deep_models):
    return {
        model_id: model
        for model_id, model in deep_models.items()
        if profile_for(REGISTRY.get(model_id)).supports(XaiCapability.GRADIENT_COMPATIBLE)
    }


@pytest.fixture(scope="module")
def learned_mlp():
    """A model that actually fits, for the ground-truth attribution tests."""
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=600, seed=3)
    X_train, y_train, _, _ = worlds.split(dataset, seed=3)
    return _fit("mlp", X_train, y_train, max_epochs=60), X_train


# =================================================== the declaration is measured

def test_the_gradient_declaration_matches_what_the_architectures_carry(deep_models, rows):
    """Build every torch model and count the slots that carry a derivative.

    A model declared gradient-compatible must move on every feature. A model
    declared incompatible must be incompatible for the stated reason - the
    discrete slots dead, the continuous one alive - rather than merely absent
    from the table.
    """
    X, _ = rows
    sample = X.iloc[:16]

    for model_id, model in deep_models.items():
        claims = profile_for(REGISTRY.get(model_id)).supports(
            XaiCapability.GRADIENT_COMPATIBLE
        )
        reachable = deep.gradient_reachable_features(model, sample)
        dead = [FEATURES[i] for i, alive in enumerate(reachable) if not alive]

        if claims:
            assert not dead, (
                f"{model_id} claims gradient support but {dead} have an exactly "
                "zero input gradient; either the declaration or the adapter is wrong"
            )
        else:
            assert len(dead) >= 9, (
                f"{model_id} is excluded from gradient methods, but only {dead} "
                "are unreachable; if the architecture changed, the exclusion "
                "should be removed rather than left in place"
            )


def test_the_excluded_models_are_exactly_the_two_that_tokenise(deep_models):
    excluded = {
        model_id
        for model_id in deep_models
        if not profile_for(REGISTRY.get(model_id)).supports(XaiCapability.GRADIENT_COMPATIBLE)
    }
    assert excluded == caps._EMBEDDING_INPUT_MODELS


def test_asking_a_scikit_learn_model_for_a_gradient_is_refused(rows):
    X, y = rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REGISTRY.build("random_forest").fit(X, y)

    with pytest.raises(CapabilityError, match="deep families"):
        deep.input_gradient(model, X.iloc[:2])


# =========================================================== the input gradient

def test_every_gradient_model_returns_one_finite_value_per_feature(gradient_models, rows):
    X, _ = rows
    for model_id, model in gradient_models.items():
        gradients = deep.input_gradient(model, X.iloc[:5])

        assert gradients.shape == (5, len(FEATURES)), f"{model_id} returned the wrong shape"
        assert np.isfinite(gradients).all(), f"{model_id} produced non-finite gradients"


def test_a_gradient_is_the_same_every_time_it_is_asked_for(gradient_models, rows):
    """The registry declares these deterministic, and the seed study reads that.

    Dropout and batch normalisation would both break this if the module were
    left in training mode, and the failure would look like genuine instability
    rather than like a missing ``eval()``.
    """
    X, _ = rows
    for model_id, model in gradient_models.items():
        first = deep.input_gradient(model, X.iloc[:4])
        again = deep.input_gradient(model, X.iloc[:4])

        assert np.array_equal(first, again), f"{model_id} gave two different gradients"


def test_a_rows_gradient_does_not_depend_on_its_neighbours(gradient_models, rows):
    """Which is what makes differentiating a summed batch legitimate.

    In training mode, batch normalisation would make each row's output depend on
    the rest of the batch, and a per-row attribution would silently become a
    property of how the rows were grouped.
    """
    X, _ = rows
    for model_id, model in gradient_models.items():
        alone = deep.input_gradient(model, X.iloc[[7]])
        in_company = deep.input_gradient(model, X.iloc[:16])[7]

        assert np.allclose(alone[0], in_company, atol=1e-6), (
            f"{model_id}: row 7's gradient changed when other rows were present"
        )


# ======================================================== gradient times input

def test_gradient_times_input_is_exactly_that(gradient_models, rows):
    X, _ = rows
    for model_id, model in gradient_models.items():
        sample = X.iloc[:6]
        numeric, _ = model.encode(sample)

        product = deep.gradient_x_input(model, sample)
        expected = deep.input_gradient(model, sample) * np.asarray(numeric, dtype=float)

        assert np.allclose(product, expected), f"{model_id} disagreed with its own parts"


def test_a_row_at_the_encoders_origin_receives_no_attribution(gradient_models):
    """The reference point, made visible - and it is not the training mean.

    The encoder standardises five features and leaves the five binary flags on
    their raw 0/1 scale, so its origin is the training mean for the first group
    and the literal value 0 for the second. A row built at that mixed origin
    gets nothing from this method anywhere, which is the property the record's
    declared baseline is describing.
    """
    for model_id, model in gradient_models.items():
        origin = _encoder_origin(model)

        numeric, _ = model.encode(origin)
        assert np.allclose(numeric, 0.0, atol=1e-6), f"{model_id} encodes its origin oddly"
        assert np.allclose(deep.gradient_x_input(model, origin), 0.0, atol=1e-9)


def test_the_data_mean_row_is_not_the_encoders_origin(gradient_models, rows):
    """Guards the correction above against quietly reverting.

    If binary indicators were ever standardised too, the mean row would become
    the origin, every "structural zero on a binary feature" caveat in this
    package would become false, and nothing else would fail.
    """
    X, _ = rows
    model = gradient_models["mlp"]

    numeric, _ = model.encode(X.mean().to_frame().T)

    assert np.abs(numeric).max() > 0.1, (
        "the data mean now encodes to the origin; the binary features appear to "
        "be standardised, so the mixed-reference caveats need revisiting"
    )


# ========================================================= integrated gradients

def test_integrated_gradients_very_nearly_satisfies_completeness(gradient_models, rows):
    """The axiom the method is built on, measured rather than assumed.

    Attributions should sum to the logit difference between the row and the
    baseline. With a finite Riemann sum they do not exactly, and the residual is
    the honest error bar on an individual explanation.
    """
    X, _ = rows
    baseline = worlds.baseline_row(X)

    for model_id, model in gradient_models.items():
        gaps = deep.completeness_gap(model, X.iloc[:8], baseline)

        assert np.abs(gaps).max() < 0.01, (
            f"{model_id} left {np.abs(gaps).max():.4f} logits unattributed; the "
            "step count is no longer sufficient for this architecture"
        )


def test_one_architecture_needs_far_more_integration_than_the_others(rows):
    """The measurement behind the step budget, kept as a live check.

    `feature_token_mixer` leaves 1.34 logits unattributed at 8 steps where the
    plain MLP leaves 0.018 - a seventy-fold difference driven by how curved the
    path from baseline to input is through its token-mixing blocks. The budget
    is set by this model rather than by the typical one, so if the gap between
    them ever closed, 128 steps would be paying for an architecture that no
    longer needs it.
    """
    X, y = rows
    baseline = worlds.baseline_row(X)
    sample = X.iloc[:8]

    mixer = _fit("feature_token_mixer", X, y)
    plain = _fit("mlp", X, y)

    mixer_gap = np.abs(deep.completeness_gap(mixer, sample, baseline, steps=8)).max()
    plain_gap = np.abs(deep.completeness_gap(plain, sample, baseline, steps=8)).max()

    assert mixer_gap > 10 * plain_gap, (
        f"the architectures now integrate comparably ({mixer_gap:.4f} against "
        f"{plain_gap:.4f}); the step budget should be revisited"
    )


def test_more_steps_narrow_the_completeness_gap(gradient_models, rows):
    """Confirms the residual is integration error and not a defect.

    An error that did not shrink with resolution would mean the path, the
    baseline or the gradient was wrong - a different problem with a different
    fix.
    """
    X, _ = rows
    baseline = worlds.baseline_row(X)
    model = gradient_models["mlp"]

    coarse = np.abs(deep.completeness_gap(model, X.iloc[:8], baseline, steps=2)).max()
    fine = np.abs(deep.completeness_gap(model, X.iloc[:8], baseline, steps=64)).max()

    assert fine < coarse


def test_a_row_that_is_the_baseline_receives_exactly_no_attribution(gradient_models, rows):
    """Nothing moved, so nothing is attributed - at every model, exactly."""
    X, _ = rows
    baseline = worlds.baseline_row(X)
    baseline_frame = baseline.to_frame().T[list(X.columns)]

    for model_id, model in gradient_models.items():
        attributions = deep.integrated_gradients(model, baseline_frame, baseline)
        assert np.allclose(attributions, 0.0, atol=1e-12), f"{model_id} attributed a null move"


def test_integrated_gradients_returns_one_value_per_feature_per_row(gradient_models, rows):
    X, _ = rows
    baseline = worlds.baseline_row(X)
    model = gradient_models["mlp"]

    assert deep.integrated_gradients(model, X.iloc[:5], baseline).shape == (5, len(FEATURES))


# ================================================ the exact additive attribution

def test_the_additive_models_terms_reconstruct_its_logit(deep_models, rows):
    """Exact by construction, so it must hold to floating-point precision.

    This is the deep counterpart of the linear model's coefficient decomposition
    and the only deep attribution in the zoo that needs no faithfulness proxy.
    If the terms stopped summing to the logit, the architecture would no longer
    be additive and its capability claim would be false.
    """
    import torch

    X, _ = rows
    model = deep_models["neural_additive"]
    sample = X.iloc[:12]

    terms = deep.additive_contributions(model, sample)
    numeric, levels = model.encode(sample)
    with torch.no_grad():
        logits = model.model(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(levels, dtype=torch.int64),
        ).numpy()
    bias = float(model.model.bias.detach().numpy().reshape(-1)[0])

    assert terms.shape == (12, len(FEATURES))
    assert np.allclose(terms.sum(axis=1) + bias, logits, atol=1e-5)


def test_a_non_additive_architecture_refuses_to_produce_additive_terms(deep_models):
    with pytest.raises(CapabilityError, match="not additive"):
        deep.additive_contributions(deep_models["mlp"], None)


def test_the_additive_model_is_the_only_deep_model_claiming_native_importance():
    claiming = {
        spec.model_id
        for spec in REGISTRY
        if spec.framework is Framework.TORCH
        and spec.effective_status() is ResearchStatus.ACTIVE
        and profile_for(spec).supports(XaiCapability.NATIVE_FEATURE_IMPORTANCE)
    }
    assert claiming == {"neural_additive"}


# ============================================================== ground truth

def test_gradient_methods_find_the_feature_that_generated_the_label(learned_mlp):
    """The only test here that needs the model to have actually learned.

    A gradient describes whatever the network computes, including a network that
    computes nothing useful - so this one trains to convergence first, and the
    attribution is checked against a driver known before the model existed.
    """
    model, X_train = learned_mlp
    baseline = worlds.baseline_row(X_train)
    sample = X_train.iloc[:40]

    for name, values in (
        ("gradient_x_input", deep.gradient_x_input(model, sample)),
        ("integrated_gradients", deep.integrated_gradients(model, sample, baseline)),
    ):
        totals = np.abs(values).mean(axis=0)
        winner = FEATURES[int(np.argmax(totals))]
        assert winner == "GenHlth", f"{name} ranked {winner} above the true driver"
