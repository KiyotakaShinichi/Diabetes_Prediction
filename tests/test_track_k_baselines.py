"""Baselines, challenger wiring, calibration and threshold selection.

The property under test throughout is that nothing sees the test partition
before the final evaluation, and that every choice made on validation is made
honestly - including the choice of calibrator, which is the one place a
non-parametric fit can flatter itself if it is judged on the rows it was fitted
on.

Model training here uses tiny budgets on small slices. The real runs belong to
the benchmark.
"""
import numpy as np
import pandas as pd
import pytest

from ml_core import feature_contract
from ml_core import training as core_training
from research.track_k import baselines, calibration, challengers, evaluation, protocol

torch = pytest.importorskip("torch", reason="PyTorch is a Track K research dependency")

FEATURES = feature_contract.FEATURE_NAMES


def synthetic_splits(rows: int = 900, seed: int = 0) -> core_training.TrainingSplits:
    """A contract-valid dataset with real signal, split the protocol's way."""
    rng = np.random.default_rng(seed)
    columns = {}
    for spec in feature_contract.FEATURE_SPECS:
        if spec.kind == "continuous":
            columns[spec.name] = rng.uniform(spec.minimum, spec.maximum, rows)
        else:
            columns[spec.name] = rng.integers(int(spec.minimum), int(spec.maximum) + 1, rows)
    frame = pd.DataFrame(columns)[list(FEATURES)]
    logit = 0.4 * frame["GenHlth"] + 0.05 * frame["BMI"] + 1.1 * frame["HighBP"] - 3.6
    target = pd.Series(
        (1 / (1 + np.exp(-(logit + rng.normal(0, 0.4, rows)))) > 0.5).astype(int),
        name=protocol.TARGET_COLUMN,
    )
    return core_training.split_training_data(
        frame, target,
        test_size=protocol.TEST_SIZE,
        val_size=protocol.VALIDATION_SIZE_OF_REMAINDER,
        random_state=protocol.SPLIT_SEED,
    )


@pytest.fixture(scope="module")
def splits():
    return synthetic_splits()


# ================================================= frozen search budgets

@pytest.mark.parametrize("family", protocol.MODEL_FAMILIES)
def test_every_family_has_a_frozen_search_budget(family):
    """Budgets are committed before the final evaluation, not chosen after."""
    budget = challengers.search_budget(family)

    assert isinstance(budget, int)
    assert 0 < budget <= 50, f"{family} budget of {budget} is not modest"


def test_the_deep_budgets_are_not_larger_than_the_classical_ones():
    """A fair comparison gives each family comparable development effort."""
    classical = max(challengers.search_budget(f) for f in protocol.CLASSICAL_FAMILIES)
    deep = max(challengers.search_budget(f) for f in protocol.DEEP_FAMILIES)

    assert deep <= classical


# ================================================= classical baselines

def test_the_logistic_baseline_trains_and_predicts(splits):
    model = baselines.fit_final("logistic_regression", {"C": 1.0, "solver": "lbfgs"}, splits, 1042)

    proba = baselines.predict_proba(model, splits.X_val)

    assert proba.shape == (len(splits.X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_the_xgboost_baseline_trains_and_predicts(splits):
    model = baselines.fit_final(
        "xgboost", {"n_estimators": 20, "max_depth": 3}, splits, 2042
    )

    proba = baselines.predict_proba(model, splits.X_val)

    assert proba.shape == (len(splits.X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_the_logistic_baseline_standardises_inside_the_pipeline(splits):
    """So the scaler is fitted on train when the pipeline is fitted on train."""
    model = baselines.build_logistic({"C": 1.0}, seed=1)

    assert list(dict(model.steps)) == ["scaler", "model"]


def test_baselines_are_deterministic(splits):
    first = baselines.fit_final("xgboost", {"n_estimators": 15}, splits, 2042)
    second = baselines.fit_final("xgboost", {"n_estimators": 15}, splits, 2042)

    assert np.array_equal(
        baselines.predict_proba(first, splits.X_val),
        baselines.predict_proba(second, splits.X_val),
    )


def test_a_search_reports_what_it_optimised(splits):
    outcome = baselines.search_logistic(splits, trials=3, seed=1042)

    assert outcome.family == "logistic_regression"
    assert outcome.metric == protocol.PRIMARY_METRIC
    assert outcome.as_dict()["searched_on"] == "train+validation"
    assert 0.0 <= outcome.best_validation_score <= 1.0


def test_a_search_never_touches_the_test_partition(splits, monkeypatch):
    """Fail loudly if any search path reads test rows."""
    forbidden = splits.X_test

    class Tripwire:
        def __getattr__(self, name):
            raise AssertionError("the search touched the test partition")

    guarded = core_training.TrainingSplits(
        X_train=splits.X_train, X_val=splits.X_val, X_test=Tripwire(),
        y_train=splits.y_train, y_val=splits.y_val, y_test=Tripwire(),
        feature_names=splits.feature_names,
    )

    outcome = baselines.search_logistic(guarded, trials=2, seed=1042)

    assert outcome.trials == 2
    assert len(forbidden) > 0


# ================================================== challenger wiring

def test_preparation_fits_transforms_on_train_only(splits):
    data = challengers.prepare(splits)

    assert data.standardiser.fitted_rows == len(splits.X_train)
    assert data.train_numeric.shape == (len(splits.X_train), len(FEATURES))
    assert data.val_numeric.shape == (len(splits.X_val), len(FEATURES))


def test_the_test_partition_is_transformed_by_frozen_statistics(splits):
    data = challengers.prepare(splits)
    before = data.standardiser.means

    numeric, levels = data.transform(splits.X_test)

    assert data.standardiser.means == before, "transforming test must not refit"
    assert numeric.shape == (len(splits.X_test), len(FEATURES))
    assert levels.shape == (len(splits.X_test), len(FEATURES))


@pytest.mark.parametrize("family", protocol.DEEP_FAMILIES)
def test_each_challenger_trains_and_predicts(family, splits):
    data = challengers.prepare(splits)
    params = (
        {"hidden_dims": [16], "batch_size": 256}
        if family == "mlp"
        else {"d_token": 8, "n_blocks": 1, "n_heads": 2, "batch_size": 256}
    )

    model, result = challengers.train_challenger(
        family, params, data, seed=protocol.model_seed(family), max_epochs=2
    )
    proba = challengers.predict_proba(model, data, splits.X_test)

    assert result.epochs_run >= 1
    assert proba.shape == (len(splits.X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert challengers.parameter_count(model) > 0


def test_challenger_training_is_reproducible(splits):
    data = challengers.prepare(splits)
    params = {"hidden_dims": [16], "batch_size": 256}

    first, _ = challengers.train_challenger("mlp", params, data, seed=3042, max_epochs=2)
    second, _ = challengers.train_challenger("mlp", params, data, seed=3042, max_epochs=2)

    assert np.array_equal(
        challengers.predict_proba(first, data, splits.X_val),
        challengers.predict_proba(second, data, splits.X_val),
    )


# ===================================================== threshold selection

def test_the_threshold_comes_from_validation(splits):
    model = baselines.fit_final("logistic_regression", {"C": 1.0}, splits, 1042)
    val_proba = baselines.predict_proba(model, splits.X_val)

    threshold = calibration.select_threshold(splits.y_val.to_numpy(), val_proba)

    assert 0.0 <= threshold <= 1.0
    assert np.isfinite(threshold)


# ========================================================== calibration

@pytest.fixture
def miscalibrated():
    """Predictions with a known, fixable distortion."""
    rng = np.random.default_rng(5)
    truth_proba = rng.uniform(0.05, 0.95, 4000)
    outcomes = (rng.random(4000) < truth_proba).astype(int)
    logit = np.log(truth_proba / (1 - truth_proba))
    distorted = 1 / (1 + np.exp(-(logit * 2.0)))  # over-confident
    return outcomes, distorted


def test_calibration_candidates_are_scored_out_of_fold(miscalibrated):
    """The defect this guards: in-sample isotonic memorises and always wins.

    Fitting and judging on the same rows gave isotonic an ECE near 1e-17. Out of
    fold it reports a real number, so the two candidates can be compared.
    """
    outcomes, distorted = miscalibrated

    _cal, outcome = calibration.select_calibrator(outcomes, distorted)

    for name, score in outcome.candidates.items():
        assert score > 1e-6, f"{name} scored {score:g}, which is an in-sample artefact"


def test_calibration_is_applied_when_it_helps(miscalibrated):
    outcomes, distorted = miscalibrated

    calibrator, outcome = calibration.select_calibrator(outcomes, distorted)

    assert outcome.applied
    assert outcome.method in {"sigmoid", "isotonic"}
    after = evaluation.expected_calibration_error(outcomes, calibrator(distorted))
    assert after < outcome.validation_ece_before


def test_calibration_is_declined_when_it_does_not_help(monkeypatch):
    """A transform that makes things worse must not be applied on principle."""
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(0.05, 0.95, 3000)
    outcomes = (rng.random(3000) < probabilities).astype(int)

    monkeypatch.setattr(
        calibration, "_out_of_fold_ece", lambda *args, **kwargs: 0.9
    )
    calibrator, outcome = calibration.select_calibrator(outcomes, probabilities)

    assert not outcome.applied
    assert outcome.method == "none"
    assert np.array_equal(calibrator(probabilities), probabilities)
    assert "did not improve" in outcome.reason


def test_isotonic_is_skipped_when_validation_is_small():
    """It needs enough rows to estimate a step function rather than memorise it."""
    rng = np.random.default_rng(9)
    probabilities = rng.uniform(0.1, 0.9, 200)
    outcomes = (rng.random(200) < probabilities).astype(int)

    _cal, outcome = calibration.select_calibrator(outcomes, probabilities)

    assert "isotonic" not in outcome.candidates


def test_the_calibration_record_states_where_it_was_fitted(miscalibrated):
    outcomes, distorted = miscalibrated

    _cal, outcome = calibration.select_calibrator(outcomes, distorted)

    assert outcome.as_dict()["fitted_on"] == "validation"


def test_a_calibrator_returns_probabilities(miscalibrated):
    outcomes, distorted = miscalibrated

    calibrator, _outcome = calibration.select_calibrator(outcomes, distorted)
    calibrated = calibrator(distorted)

    assert calibrated.shape == distorted.shape
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


# ==================================================== challenger search

@pytest.fixture
def quick_search(monkeypatch):
    """Search with a two-epoch ceiling, so one trial costs seconds."""
    monkeypatch.setattr(challengers, "SEARCH_MAX_EPOCHS", 2)


def test_the_mlp_search_returns_a_configuration_the_trainer_accepts(splits, quick_search):
    """The search must emit params, not Optuna's internal choice indices."""
    data = challengers.prepare(splits)

    outcome = challengers.search_mlp(data, trials=1, seed=3042)

    assert outcome.family == "mlp"
    assert "width_choice" not in outcome.best_params, "an index leaked into the config"
    assert isinstance(outcome.best_params["hidden_dims"], list)
    model, _result = challengers.train_challenger(
        "mlp", outcome.best_params, data, seed=3042, max_epochs=1
    )
    assert challengers.parameter_count(model) > 0


def test_the_ft_transformer_search_returns_a_trainable_configuration(splits, quick_search):
    data = challengers.prepare(splits)

    outcome = challengers.search_ft_transformer(data, trials=1, seed=4042)

    assert outcome.family == "ft_transformer"
    assert "token_choice" not in outcome.best_params
    assert outcome.best_params["d_token"] in challengers._FT_TOKENS
    assert outcome.best_params["d_token"] % outcome.best_params["n_heads"] == 0, (
        "attention heads must divide the token width"
    )
    model, _result = challengers.train_challenger(
        "ft_transformer", outcome.best_params, data, seed=4042, max_epochs=1
    )
    assert challengers.parameter_count(model) > 0


def test_a_search_reports_the_budget_it_was_given(splits, quick_search):
    data = challengers.prepare(splits)

    outcome = challengers.search_mlp(data, trials=2, seed=3042)

    assert outcome.trials == 2
    assert outcome.as_dict()["searched_on"] == "train+validation"


def test_the_residual_tower_search_returns_a_trainable_configuration(splits, quick_search):
    data = challengers.prepare(splits)

    outcome = challengers.search_tabular_resnet(data, trials=1, seed=5042)

    assert outcome.family == "tabular_resnet"
    assert "width_choice" not in outcome.best_params
    assert outcome.best_params["d_hidden"] in challengers._RESNET_WIDTHS
    model, _result = challengers.train_challenger(
        "tabular_resnet", outcome.best_params, data, seed=5042, max_epochs=1
    )
    assert challengers.parameter_count(model) > 0


@pytest.mark.parametrize("family", ["mlp", "ft_transformer", "tabular_resnet"])
def test_the_search_dispatcher_reaches_every_deep_family(family, splits, quick_search):
    """One dispatch point, so a new challenger cannot be half-wired."""
    data = challengers.prepare(splits)

    outcome = challengers.search(family, data, trials=1, seed=protocol.model_seed(family))

    assert outcome.family == family


def test_the_search_dispatcher_refuses_a_classical_family(splits):
    data = challengers.prepare(splits)

    with pytest.raises(ValueError, match="not a deep challenger"):
        challengers.search("xgboost", data, trials=1, seed=1)


def test_building_a_classical_family_as_a_challenger_is_refused(splits):
    data = challengers.prepare(splits)

    with pytest.raises(ValueError, match="not a deep challenger"):
        challengers.train_challenger("logistic_regression", {}, data, seed=1, max_epochs=1)
