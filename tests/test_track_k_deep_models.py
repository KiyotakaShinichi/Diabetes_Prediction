"""The deep challengers: shapes, gradients, determinism and leakage.

These are behavioural, not decorative. Each one can fail: the leakage tests
compare against a standardiser deliberately fitted on the wrong partition, the
determinism tests train twice and compare weights, and the checkpoint tests
round-trip through a real file.

Everything here runs on CPU in seconds on tiny synthetic data. The real training
runs live in the benchmark, not the test suite.
"""
import numpy as np
import pandas as pd
import pytest

from ml_core import feature_contract
from research.track_k import protocol
from research.track_k.deep import models, preprocessing, training

torch = pytest.importorskip("torch", reason="PyTorch is a Track K research dependency")

FEATURES = feature_contract.FEATURE_NAMES


def synthetic_frame(rows: int = 256, seed: int = 0) -> pd.DataFrame:
    """Contract-valid rows, so every test exercises the real feature semantics."""
    rng = np.random.default_rng(seed)
    columns = {}
    for spec in feature_contract.FEATURE_SPECS:
        if spec.kind == "continuous":
            columns[spec.name] = rng.uniform(spec.minimum, spec.maximum, rows)
        else:
            columns[spec.name] = rng.integers(int(spec.minimum), int(spec.maximum) + 1, rows)
    return pd.DataFrame(columns)[list(FEATURES)]


def synthetic_target(frame: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """A learnable signal, so an overfit test has something to overfit."""
    rng = np.random.default_rng(seed)
    logit = 0.35 * frame["GenHlth"] + 0.06 * frame["BMI"] + 0.9 * frame["HighBP"] - 3.4
    noise = rng.normal(0, 0.25, len(frame))
    return (1 / (1 + np.exp(-(logit + noise))) > 0.5).astype(np.float32).to_numpy()


@pytest.fixture(scope="module")
def vocabulary():
    return preprocessing.build_ordinal_vocabulary(FEATURES)


@pytest.fixture(scope="module")
def prepared(vocabulary):
    frame = synthetic_frame()
    state = preprocessing.fit_standardiser(frame)
    return {
        "frame": frame,
        "numeric": preprocessing.apply_standardiser(state, frame),
        "levels": preprocessing.encode_ordinal_levels(vocabulary, frame),
        "target": synthetic_target(frame),
        "state": state,
    }


# ======================================================= preprocessing contract

def test_binary_features_are_not_standardised(prepared):
    """Scaling a 0/1 indicator buys nothing and obscures the learned weight."""
    scaled = {FEATURES[index] for index in prepared["state"].scaled_indices}

    for spec in feature_contract.FEATURE_SPECS:
        if spec.kind == "binary":
            assert spec.name not in scaled
        else:
            assert spec.name in scaled


def test_standardised_columns_have_train_statistics(prepared):
    numeric = prepared["numeric"]

    for index in prepared["state"].scaled_indices:
        assert numeric[:, index].mean() == pytest.approx(0.0, abs=1e-5)
        assert numeric[:, index].std() == pytest.approx(1.0, abs=1e-3)


def test_binary_columns_pass_through_unchanged(prepared):
    frame, numeric = prepared["frame"], prepared["numeric"]
    scaled = set(prepared["state"].scaled_indices)

    for index, name in enumerate(FEATURES):
        if index not in scaled:
            assert np.allclose(numeric[:, index], frame[name].to_numpy())


def test_a_transform_fitted_on_train_is_not_refitted_on_other_partitions():
    """The leakage guard: validation statistics must not move the transform."""
    train = synthetic_frame(rows=300, seed=1)
    validation = synthetic_frame(rows=300, seed=2) * 0 + 99.0  # wildly different

    state = preprocessing.fit_standardiser(train)
    before = state.means

    preprocessing.apply_standardiser(state, validation.astype(float))

    assert state.means == before, "applying a transform must never refit it"


def test_fitting_on_the_full_dataset_is_detectably_different():
    """Proves the previous test is not vacuous: a wrong fit changes the numbers."""
    train = synthetic_frame(rows=300, seed=1)
    everything = pd.concat([train, synthetic_frame(rows=300, seed=9) + 3.0], ignore_index=True)

    honest = preprocessing.fit_standardiser(train)
    leaked = preprocessing.fit_standardiser(everything)

    assert honest.means != leaked.means
    assert honest.fitted_rows == 300
    assert leaked.fitted_rows == 600


def test_a_column_order_mismatch_is_refused(prepared, vocabulary):
    shuffled = prepared["frame"][list(reversed(FEATURES))]

    with pytest.raises(preprocessing.LeakageError, match="column mismatch"):
        preprocessing.apply_standardiser(prepared["state"], shuffled)
    with pytest.raises(preprocessing.LeakageError, match="column mismatch"):
        preprocessing.encode_ordinal_levels(vocabulary, shuffled)


def test_fitting_on_nothing_is_refused():
    with pytest.raises(preprocessing.LeakageError):
        preprocessing.fit_standardiser(pd.DataFrame(columns=list(FEATURES)))


def test_a_zero_variance_column_does_not_produce_infinities():
    frame = synthetic_frame(rows=64)
    frame["BMI"] = 25.0

    state = preprocessing.fit_standardiser(frame)
    numeric = preprocessing.apply_standardiser(state, frame)

    assert np.isfinite(numeric).all()


# ============================================================ the vocabulary

def test_the_vocabulary_comes_from_the_contract_not_the_data(vocabulary):
    """A legal-but-unseen level must still have an embedding row."""
    for name, size in zip(FEATURES, vocabulary.cardinalities, strict=True):
        spec = feature_contract.spec_for(name)
        allowed = spec.allowed_values
        assert size == (0 if allowed is None else len(allowed))


def test_continuous_features_have_no_embedding_rows(vocabulary):
    position = FEATURES.index("BMI")

    assert vocabulary.cardinalities[position] == 0


def test_encoded_levels_stay_inside_their_table(prepared, vocabulary):
    levels = prepared["levels"]

    for index, size in enumerate(vocabulary.cardinalities):
        if size:
            assert levels[:, index].min() >= 0
            assert levels[:, index].max() < size


def test_out_of_contract_values_are_clipped_rather_than_indexing_out_of_bounds(vocabulary):
    frame = synthetic_frame(rows=16)
    frame["GenHlth"] = 99

    levels = preprocessing.encode_ordinal_levels(vocabulary, frame)

    position = FEATURES.index("GenHlth")
    assert levels[:, position].max() < vocabulary.cardinalities[position]


# ================================================================ the models

@pytest.fixture(params=["mlp", "ft_transformer"])
def model(request, vocabulary):
    if request.param == "mlp":
        return models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=(16, 8)))
    return models.FTTransformer(
        vocabulary, models.FTTransformerConfig(d_token=8, n_blocks=1, n_heads=2)
    )


def test_forward_returns_one_logit_per_row(model, prepared):
    numeric = torch.from_numpy(prepared["numeric"][:32])
    levels = torch.from_numpy(prepared["levels"][:32])

    model.eval()
    with torch.no_grad():
        logits = model(numeric, levels)

    assert logits.shape == (32,)
    assert torch.isfinite(logits).all()


def test_probabilities_stay_in_range(model, prepared):
    proba = training.predict_proba(model, prepared["numeric"][:64], prepared["levels"][:64])

    assert proba.shape == (64,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_a_backward_pass_updates_every_trainable_parameter(model, prepared):
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    logits = model(
        torch.from_numpy(prepared["numeric"][:64]), torch.from_numpy(prepared["levels"][:64])
    )
    loss = criterion(logits, torch.from_numpy(prepared["target"][:64]))
    loss.backward()
    optimiser.step()

    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    changed = sum(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert changed > 0, "no parameter moved"
    assert torch.isfinite(loss)


def test_a_checkpoint_round_trips_exactly(model, prepared, tmp_path):
    baseline = training.predict_proba(model, prepared["numeric"][:32], prepared["levels"][:32])
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)

    reloaded = type(model)(
        *( (len(FEATURES), model.config) if isinstance(model, models.TabularMLP)
           else (model.tokenizer.vocabulary, model.config) )
    )
    reloaded.load_state_dict(torch.load(path, weights_only=True))

    restored = training.predict_proba(
        reloaded, prepared["numeric"][:32], prepared["levels"][:32]
    )
    assert np.allclose(baseline, restored, atol=1e-7)


def test_the_transformer_rejects_an_indivisible_head_count(vocabulary):
    with pytest.raises(ValueError, match="divide evenly"):
        models.FTTransformer(
            vocabulary, models.FTTransformerConfig(d_token=10, n_heads=4)
        )


def test_the_transformer_tokenizes_one_token_per_feature_plus_cls(vocabulary, prepared):
    tokenizer = models.FeatureTokenizer(vocabulary, d_token=8)

    tokens = tokenizer(
        torch.from_numpy(prepared["numeric"][:5]), torch.from_numpy(prepared["levels"][:5])
    )

    assert tokens.shape == (5, len(FEATURES) + 1, 8)
    assert torch.isfinite(tokens).all()


def test_the_mlp_ignores_the_level_tensor(prepared):
    """Its modelling assumption is that ordinal codes are usable as numbers."""
    mlp = models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=(8,)))
    mlp.eval()
    numeric = torch.from_numpy(prepared["numeric"][:16])

    with torch.no_grad():
        first = mlp(numeric, torch.from_numpy(prepared["levels"][:16]))
        second = mlp(numeric, torch.zeros_like(torch.from_numpy(prepared["levels"][:16])))

    assert torch.equal(first, second)


def test_the_transformer_does_not_ignore_the_level_tensor(vocabulary, prepared):
    """The substantive difference from the MLP: levels are embedded, not scaled."""
    model = models.FTTransformer(
        vocabulary, models.FTTransformerConfig(d_token=8, n_blocks=1, n_heads=2)
    )
    model.eval()
    numeric = torch.from_numpy(prepared["numeric"][:16])

    with torch.no_grad():
        first = model(numeric, torch.from_numpy(prepared["levels"][:16]))
        second = model(numeric, torch.zeros_like(torch.from_numpy(prepared["levels"][:16])))

    assert not torch.equal(first, second)


def build_family(family: str, vocabulary, *, tiny: bool = False):
    """Construct the model a family name actually refers to.

    Written as an exhaustive mapping rather than an if/else chain because the
    chain it replaces silently built an FT-Transformer for every family that was
    not the MLP - so adding a third challenger produced a test that passed
    without ever instantiating the model it was parametrised for.
    """
    builders = {
        "mlp": lambda: models.TabularMLP(
            len(FEATURES), models.MLPConfig(hidden_dims=(8,) if tiny else (128, 64))
        ),
        "ft_transformer": lambda: models.FTTransformer(
            vocabulary,
            models.FTTransformerConfig(d_token=8, n_blocks=1, n_heads=2)
            if tiny
            else models.FTTransformerConfig(),
        ),
        "tabular_resnet": lambda: models.TabularResNet(
            len(FEATURES),
            models.TabularResNetConfig(d_hidden=8, n_blocks=1)
            if tiny
            else models.TabularResNetConfig(),
        ),
    }
    assert set(builders) == set(protocol.DEEP_FAMILIES), (
        "a deep family has no builder here, so it would go untested"
    )
    return builders[family]()


@pytest.mark.parametrize("family", protocol.DEEP_FAMILIES)
def test_parameter_counts_stay_proportionate(family, vocabulary):
    """Ten features and 40k rows do not support a large network."""
    count = models.count_parameters(build_family(family, vocabulary))

    assert 1_000 < count < 200_000, f"{family} has {count:,} parameters"


# ============================================================== the training loop

def train_tiny(model, prepared, *, seed: int = 3, epochs: int = 3):
    return training.train_model(
        model,
        train_numeric=prepared["numeric"][:200],
        train_levels=prepared["levels"][:200],
        train_target=prepared["target"][:200],
        val_numeric=prepared["numeric"][200:],
        val_levels=prepared["levels"][200:],
        val_target=prepared["target"][200:],
        config=training.TrainingConfig(
            max_epochs=epochs, batch_size=64, seed=seed, patience=epochs
        ),
    )


def build_mlp(seed: int = 3, hidden=(8,)):
    """Weights are randomised in __init__, so construction must be seeded too."""
    return training.build_seeded(
        lambda: models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=hidden)), seed
    )


def test_training_is_deterministic_under_a_fixed_seed(prepared):
    """Two runs, same seed, identical weights - the property a benchmark needs."""
    first, _ = train_tiny(build_mlp(), prepared)
    second, _ = train_tiny(build_mlp(), prepared)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_a_different_seed_produces_a_different_model(prepared):
    """Proves the determinism test is measuring seeding, not a frozen model."""
    first, _ = train_tiny(build_mlp(seed=1), prepared, seed=1)
    second, _ = train_tiny(build_mlp(seed=2), prepared, seed=2)

    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )


def test_training_records_a_learning_curve(prepared):
    _model, result = train_tiny(
        models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=(8,))), prepared, epochs=4
    )

    assert len(result.history) == 4
    assert all(np.isfinite(record.train_loss) for record in result.history)
    assert all(0.0 <= record.val_roc_auc <= 1.0 for record in result.history)
    assert result.duration_seconds > 0


def test_training_reduces_loss_on_a_learnable_signal(prepared):
    _model, result = train_tiny(
        models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=(32, 16))),
        prepared, epochs=25,
    )

    assert result.history[-1].train_loss < result.history[0].train_loss


def test_early_stopping_restores_the_best_checkpoint(prepared):
    """A late overfit must not be what the benchmark evaluates."""
    _model, result = train_tiny(
        models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=(64, 32))),
        prepared, epochs=40,
    )

    assert result.best_epoch >= 1
    assert result.best_val_loss <= min(record.val_loss for record in result.history) + 1e-9


def test_the_loop_runs_on_cpu_without_cuda(prepared):
    """CI has no GPU; the benchmark must never require one."""
    model, _result = train_tiny(
        models.TabularMLP(len(FEATURES), models.MLPConfig(hidden_dims=(8,))), prepared
    )

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.parametrize("family", protocol.DEEP_FAMILIES)
def test_each_deep_family_trains_end_to_end(family, prepared, vocabulary):
    model = build_family(family, vocabulary, tiny=True)

    trained, result = train_tiny(model, prepared)
    proba = training.predict_proba(trained, prepared["numeric"], prepared["levels"])

    assert result.epochs_run >= 1
    assert proba.shape == (len(prepared["numeric"]),)
    assert np.isfinite(proba).all()


# ================================================================= the metric

def test_the_internal_roc_auc_matches_sklearn(prepared):
    """The loop computes its own AUC; it must agree with the reference."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = (rng.random(500) > 0.5).astype(int)

    assert training.roc_auc(labels, scores) == pytest.approx(
        roc_auc_score(labels, scores), abs=1e-9
    )


def test_a_single_class_partition_yields_the_uninformative_value():
    """Degenerate batches happen early in training; they must not raise."""
    assert training.roc_auc(np.ones(10, dtype=int), np.random.random(10)) == 0.5
    assert training.roc_auc(np.zeros(10, dtype=int), np.random.random(10)) == 0.5


def test_tied_scores_do_not_inflate_the_metric():
    labels = np.array([0, 0, 1, 1])
    tied = np.array([0.5, 0.5, 0.5, 0.5])

    assert training.roc_auc(labels, tied) == pytest.approx(0.5)
