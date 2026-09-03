"""Every registered model must be what its spec says it is.

A registry of thirty algorithms is only useful if the declarations can be
trusted, because everything downstream reads them rather than the models: the
benchmark decides whether to ask for probabilities, the capability matrix
publishes what each model can do, and the cards describe it to a reader. A spec
that overclaims would put a column of meaningless numbers in the results table
under a header that says they mean something.

So each declaration is asserted against the built model. If a spec claims
probabilities, the model must produce them in [0, 1]; if it claims none, asking
must raise rather than return something plausible. If it claims determinism, two
builds must agree exactly.

These run on tiny synthetic data with tiny budgets - the point is the contract,
not the score.
"""
import warnings

import numpy as np
import pytest

from research.model_zoo import synthetic
from research.model_zoo.contracts import (
    CapabilityError,
    Family,
    Framework,
    ProbabilityBehavior,
    ResearchStatus,
)
from research.model_zoo.registry import REGISTRY, ModelRegistry

pytest.importorskip("torch", reason="the zoo's deep models need PyTorch")

#: Small enough that thirty models fit in the suite's budget, large enough that
#: a five-fold internal split and a 25-neighbour classifier remain well defined.
ROWS = 240
DEEP_EPOCHS = 2

ACTIVE_IDS = [spec.model_id for spec in REGISTRY.active()]


@pytest.fixture(scope="module")
def tiny():
    dataset = synthetic.make(synthetic.SyntheticProblem.LINEARLY_SEPARABLE, rows=ROWS, seed=3)
    return synthetic.split(dataset, seed=3)


def _build(model_id: str):
    spec = REGISTRY.get(model_id)
    overrides = {"max_epochs": DEEP_EPOCHS} if spec.framework is Framework.TORCH else {}
    return REGISTRY.build(model_id, **overrides)


@pytest.fixture(scope="module")
def fitted(tiny):
    """Every active model, fitted once and shared across the contract tests."""
    X_train, y_train, _X, _y = tiny
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model_id in ACTIVE_IDS:
            models[model_id] = _build(model_id).fit(X_train, y_train)
    return models


# ==================================================== the registry itself

def test_the_registry_is_populated():
    assert len(REGISTRY) >= 25, "the zoo should hold a broad set of algorithms"


def test_every_family_is_represented():
    """A zoo missing a family cannot support the diversity analysis."""
    for family in Family:
        assert REGISTRY.by_family(family), f"no models registered for {family.value}"


def test_model_ids_are_unique():
    assert len(REGISTRY.ids()) == len(set(REGISTRY.ids()))


def test_a_duplicate_registration_is_refused():
    registry = ModelRegistry()
    spec = REGISTRY.get("logistic_l2")
    registry.register(spec)

    with pytest.raises(ValueError, match="duplicate model_id"):
        registry.register(spec)


def test_an_unknown_model_id_names_what_is_available():
    with pytest.raises(KeyError, match="unknown model_id"):
        REGISTRY.get("perceptron_of_theseus")


def test_every_spec_explains_why_it_is_in_the_zoo():
    """A model with no stated rationale is a model nobody chose deliberately."""
    for spec in REGISTRY:
        assert len(spec.rationale) > 30, f"{spec.model_id} has no rationale"


def test_every_spec_declares_a_seed():
    for spec in REGISTRY:
        assert isinstance(spec.seed, int)


def test_optional_models_are_downgraded_when_absent():
    """The core install must not need LightGBM or CatBoost to import the zoo."""
    for spec in REGISTRY:
        if spec.optional_dependency is not None and not spec.is_available():
            assert spec.effective_status() is ResearchStatus.OPTIONAL


def test_a_model_with_no_optional_dependency_is_always_available():
    for spec in REGISTRY:
        if spec.optional_dependency is None:
            assert spec.is_available()


def test_the_registry_serialises_to_plain_data():
    """The capability matrix and the cards are generated from this."""
    for spec in REGISTRY:
        payload = spec.as_dict()
        assert payload["model_id"] == spec.model_id
        assert payload["family"] in {f.value for f in Family}
        assert set(payload["capabilities"]) >= {"supports_predict_proba", "requires_scaling"}


# ============================================ declarations match behaviour

@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_every_active_model_builds_and_fits(model_id, fitted):
    assert fitted[model_id] is not None


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_every_active_model_predicts_one_label_per_row(model_id, fitted, tiny):
    _Xtr, _ytr, X_test, _y = tiny

    predictions = fitted[model_id].predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert set(np.unique(predictions)) <= {0, 1}


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_probability_support_matches_the_declaration(model_id, fitted, tiny):
    """Claiming probabilities means producing them; claiming none means raising."""
    _Xtr, _ytr, X_test, _y = tiny
    spec = REGISTRY.get(model_id)
    model = fitted[model_id]

    if spec.capabilities.supports_predict_proba:
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert ((proba >= 0) & (proba <= 1)).all(), "probabilities outside [0, 1]"
    else:
        with pytest.raises(CapabilityError):
            model.predict_proba(X_test)


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_ranking_scores_exist_unless_the_model_declares_hard_labels(model_id, fitted, tiny):
    """Threshold-free metrics need a ranking; a hard-label model has none."""
    _Xtr, _ytr, X_test, _y = tiny
    spec = REGISTRY.get(model_id)
    model = fitted[model_id]

    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        with pytest.raises(CapabilityError, match="hard labels only"):
            model.decision_scores(X_test)
    else:
        scores = model.decision_scores(X_test)
        assert scores.shape == (len(X_test),)
        assert np.isfinite(scores).all()


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_a_model_declaring_no_probability_is_not_asked_for_one(model_id):
    """The declaration and the behaviour class must agree with each other."""
    spec = REGISTRY.get(model_id)

    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        assert not spec.capabilities.supports_predict_proba
    if spec.probability_behavior is ProbabilityBehavior.REQUIRES_EXTERNAL_CALIBRATION:
        assert not spec.capabilities.supports_predict_proba, (
            f"{spec.model_id} needs external calibration but claims native probabilities"
        )


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_scaling_declaration_matches_the_preprocessing_contract(model_id):
    from research.model_zoo.preprocessing import requires_scaling

    spec = REGISTRY.get(model_id)
    from research.model_zoo.contracts import Preprocessing

    if spec.preprocessing is Preprocessing.MODEL_NATIVE:
        # The deep models own their standardiser; the declaration describes the
        # model's need, not the pipeline's contents.
        return
    assert spec.capabilities.requires_scaling == requires_scaling(spec.preprocessing)


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_every_model_records_what_its_fit_cost(model_id, fitted):
    record = fitted[model_id].training_record

    assert record is not None, f"{model_id} recorded no training telemetry"
    assert record.fit_seconds >= 0
    assert record.training_rows > 0


@pytest.mark.parametrize("model_id", [s.model_id for s in REGISTRY.by_family(Family.DEEP)])
def test_every_deep_model_reports_a_parameter_count(model_id, fitted):
    """Track K's finding that bigger did not help needs a size to plot against."""
    spec = REGISTRY.get(model_id)
    if spec.effective_status() is not ResearchStatus.ACTIVE:
        pytest.skip(f"{model_id} is not active here")
    record = fitted[model_id].training_record

    assert record.parameter_count is not None
    assert record.parameter_count > 0
    assert record.epochs_run is not None and record.epochs_run >= 1


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_declared_determinism_holds(model_id, tiny):
    """Two builds with the same seed on the same rows must agree exactly."""
    spec = REGISTRY.get(model_id)
    if not spec.capabilities.deterministic:
        pytest.skip(f"{model_id} does not claim determinism")

    X_train, y_train, X_test, _y = tiny
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = _build(model_id).fit(X_train, y_train).predict(X_test)
        second = _build(model_id).fit(X_train, y_train).predict(X_test)

    assert np.array_equal(first, second), f"{model_id} is not reproducible"


@pytest.mark.parametrize("model_id", ACTIVE_IDS)
def test_feature_importance_matches_the_declaration(model_id, fitted):
    spec = REGISTRY.get(model_id)
    model = fitted[model_id]

    importance = model.feature_importance()
    if spec.capabilities.supports_feature_importance:
        assert importance is not None, f"{model_id} claims importance but returns none"
    else:
        assert importance is None


# ================================================= construction overrides

def test_build_applies_the_frozen_default_config():
    spec = REGISTRY.get("random_forest")

    assert spec.default_config["n_estimators"] == 300


def test_build_accepts_overrides_for_tests_only(tiny):
    """Tests may shrink a model; the benchmark always uses frozen defaults."""
    X_train, y_train, _X, _y = tiny

    model = REGISTRY.build("random_forest", n_estimators=5).fit(X_train, y_train)

    assert model.predict(X_train).shape == (len(X_train),)


def test_a_spec_is_immutable():
    spec = REGISTRY.get("logistic_l2")

    with pytest.raises((AttributeError, TypeError)):
        spec.model_id = "something_else"  # type: ignore[misc]


def test_specs_can_be_filtered_by_family_and_status():
    deep = REGISTRY.by_family(Family.DEEP)

    assert {s.model_id for s in deep} >= {"mlp", "ft_transformer", "neural_additive"}
    assert all(s.family is Family.DEEP for s in deep)


def test_track_k_models_are_reused_rather_than_redefined():
    """The zoo must not fork Track K's architectures and drift from its evidence."""
    import inspect

    from research.model_zoo.families import deep as zoo_deep

    source = inspect.getsource(zoo_deep)
    for imported in ("TabularMLP", "TabularResNet", "FTTransformer"):
        assert "from research.track_k.deep.models import" in source or imported in source
    assert "class TabularMLP" not in source, "Track K's MLP was copied, not imported"


def test_the_experimental_cnn_declares_its_weak_justification():
    """A model with an unjustified inductive bias must say so where it is read."""
    spec = REGISTRY.get("feature_cnn")

    assert "EXPERIMENTAL" in spec.rationale
    assert "arbitrary" in spec.rationale or "not a production candidate" in spec.rationale
