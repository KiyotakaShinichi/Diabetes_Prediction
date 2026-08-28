"""Training the deep challengers under the Track K protocol.

Wraps the DL core in the same shape the classical baselines expose - search on
validation, fit on train, predict probabilities - so the benchmark runner treats
all four families identically and no family gets a privileged code path.

Search budgets are frozen here and committed before any test evaluation. They
are small by design. A ten-feature tabular problem does not need a large search,
and a big one would mostly buy variance: the honest comparison is between
families given comparable, modest development effort, not between one family
tuned exhaustively and another tuned casually.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from ml_core import training as core_training
from research.track_k import protocol
from research.track_k.baselines import SearchOutcome
from research.track_k.deep import models, preprocessing
from research.track_k.deep import training as deep_training

#: Frozen search budgets, matched in spirit to the classical ones.
MLP_TRIALS: int = 20
FT_TRANSFORMER_TRIALS: int = 15
TABULAR_RESNET_TRIALS: int = 15

#: Epoch ceiling during search. Early stopping usually ends a trial sooner; the
#: cap keeps one pathological configuration from dominating the budget.
SEARCH_MAX_EPOCHS: int = 30
FINAL_MAX_EPOCHS: int = 80

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True, slots=True)
class PreparedTensors:
    """The frozen split, transformed once and reused by every trial."""

    train_numeric: np.ndarray
    train_levels: np.ndarray
    train_target: np.ndarray
    val_numeric: np.ndarray
    val_levels: np.ndarray
    val_target: np.ndarray
    standardiser: preprocessing.StandardiserState
    vocabulary: preprocessing.OrdinalVocabulary

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply the FROZEN training transforms to any partition."""
        return (
            preprocessing.apply_standardiser(self.standardiser, frame),
            preprocessing.encode_ordinal_levels(self.vocabulary, frame),
        )


def prepare(splits: core_training.TrainingSplits) -> PreparedTensors:
    """Fit transforms on train, then encode train and validation.

    The test partition is deliberately absent: it is transformed later, by the
    same frozen state, at evaluation time.
    """
    standardiser = preprocessing.fit_standardiser(splits.X_train)
    vocabulary = preprocessing.build_ordinal_vocabulary(splits.feature_names)
    return PreparedTensors(
        train_numeric=preprocessing.apply_standardiser(standardiser, splits.X_train),
        train_levels=preprocessing.encode_ordinal_levels(vocabulary, splits.X_train),
        train_target=splits.y_train.to_numpy().astype(np.float32),
        val_numeric=preprocessing.apply_standardiser(standardiser, splits.X_val),
        val_levels=preprocessing.encode_ordinal_levels(vocabulary, splits.X_val),
        val_target=splits.y_val.to_numpy().astype(np.float32),
        standardiser=standardiser,
        vocabulary=vocabulary,
    )


def _build(family: str, params: dict[str, Any], data: PreparedTensors, seed: int) -> Any:
    """Construct a challenger with reproducible initial weights."""
    n_features = len(data.standardiser.feature_names)
    if family == "mlp":
        config = models.MLPConfig(
            hidden_dims=tuple(params.get("hidden_dims", (128, 64))),
            dropout=params.get("dropout", 0.2),
            activation=params.get("activation", "relu"),
        )
        return deep_training.build_seeded(
            lambda: models.TabularMLP(n_features, config), seed
        )
    if family == "tabular_resnet":
        config_res = models.TabularResNetConfig(
            d_hidden=params.get("d_hidden", 64),
            d_expansion=params.get("d_expansion", 2.0),
            n_blocks=params.get("n_blocks", 3),
            dropout=params.get("dropout", 0.1),
            residual_dropout=params.get("residual_dropout", 0.0),
        )
        return deep_training.build_seeded(
            lambda: models.TabularResNet(n_features, config_res), seed
        )
    if family != "ft_transformer":
        raise ValueError(f"not a deep challenger: {family!r}")
    config_ft = models.FTTransformerConfig(
        d_token=params.get("d_token", 32),
        n_blocks=params.get("n_blocks", 3),
        n_heads=params.get("n_heads", 4),
        attention_dropout=params.get("attention_dropout", 0.1),
        ffn_dropout=params.get("ffn_dropout", 0.1),
    )
    return deep_training.build_seeded(
        lambda: models.FTTransformer(data.vocabulary, config_ft), seed
    )


def train_challenger(
    family: str,
    params: dict[str, Any],
    data: PreparedTensors,
    *,
    seed: int,
    max_epochs: int = FINAL_MAX_EPOCHS,
) -> tuple[Any, deep_training.TrainingResult]:
    """Train one challenger on train, selecting on validation."""
    model = _build(family, params, data, seed)
    config = deep_training.TrainingConfig(
        max_epochs=max_epochs,
        batch_size=params.get("batch_size", 256),
        learning_rate=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
        seed=seed,
    )
    return deep_training.train_model(
        model,
        train_numeric=data.train_numeric,
        train_levels=data.train_levels,
        train_target=data.train_target,
        val_numeric=data.val_numeric,
        val_levels=data.val_levels,
        val_target=data.val_target,
        config=config,
    )


#: Search spaces. Small and defensible for ten features: widths that a 40k-row
#: dataset can support, dropout in a normal range, and a token width and depth
#: that keep the transformer at a few tens of thousands of parameters.
_MLP_WIDTHS: tuple[tuple[int, ...], ...] = ((64,), (128, 64), (256, 128), (128, 64, 32))
_FT_TOKENS: tuple[int, ...] = (16, 32, 64)
_RESNET_WIDTHS: tuple[int, ...] = (32, 64, 128)


def search_mlp(
    data: PreparedTensors,
    *,
    trials: int = MLP_TRIALS,
    seed: int,
    max_epochs: int | None = None,
) -> SearchOutcome:
    epochs = SEARCH_MAX_EPOCHS if max_epochs is None else max_epochs

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_dims": _MLP_WIDTHS[
                trial.suggest_int("width_choice", 0, len(_MLP_WIDTHS) - 1)
            ],
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        }
        _model, result = train_challenger(
            "mlp", params, data, seed=seed, max_epochs=epochs
        )
        return result.best_val_roc_auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["hidden_dims"] = list(_MLP_WIDTHS[best.pop("width_choice")])
    return SearchOutcome(
        family="mlp",
        trials=trials,
        best_params=best,
        best_validation_score=float(study.best_value),
    )


def search_ft_transformer(
    data: PreparedTensors,
    *,
    trials: int = FT_TRANSFORMER_TRIALS,
    seed: int,
    max_epochs: int | None = None,
) -> SearchOutcome:
    epochs = SEARCH_MAX_EPOCHS if max_epochs is None else max_epochs

    def objective(trial: optuna.Trial) -> float:
        d_token = _FT_TOKENS[trial.suggest_int("token_choice", 0, len(_FT_TOKENS) - 1)]
        params = {
            "d_token": d_token,
            "n_blocks": trial.suggest_int("n_blocks", 1, 3),
            # Heads must divide the token width; 4 divides every option above.
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
            "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": 512,
        }
        _model, result = train_challenger(
            "ft_transformer", params, data, seed=seed, max_epochs=epochs
        )
        return result.best_val_roc_auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["d_token"] = _FT_TOKENS[best.pop("token_choice")]
    best["batch_size"] = 512
    return SearchOutcome(
        family="ft_transformer",
        trials=trials,
        best_params=best,
        best_validation_score=float(study.best_value),
    )


def search_tabular_resnet(
    data: PreparedTensors,
    *,
    trials: int = TABULAR_RESNET_TRIALS,
    seed: int,
    max_epochs: int | None = None,
) -> SearchOutcome:
    """Depth, width and regularisation for the residual tower."""
    epochs = SEARCH_MAX_EPOCHS if max_epochs is None else max_epochs

    def objective(trial: optuna.Trial) -> float:
        params = {
            "d_hidden": _RESNET_WIDTHS[
                trial.suggest_int("width_choice", 0, len(_RESNET_WIDTHS) - 1)
            ],
            "n_blocks": trial.suggest_int("n_blocks", 2, 4),
            "d_expansion": trial.suggest_categorical("d_expansion", [1.0, 2.0]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": 512,
        }
        _model, result = train_challenger(
            "tabular_resnet", params, data, seed=seed, max_epochs=epochs
        )
        return result.best_val_roc_auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["d_hidden"] = _RESNET_WIDTHS[best.pop("width_choice")]
    best["batch_size"] = 512
    return SearchOutcome(
        family="tabular_resnet",
        trials=trials,
        best_params=best,
        best_validation_score=float(study.best_value),
    )


def search(
    family: str,
    data: PreparedTensors,
    *,
    trials: int,
    seed: int,
    max_epochs: int | None = None,
) -> SearchOutcome:
    """Search any deep family. One dispatch point, so none can be half-wired."""
    searchers = {
        "mlp": search_mlp,
        "ft_transformer": search_ft_transformer,
        "tabular_resnet": search_tabular_resnet,
    }
    if family not in searchers:
        raise ValueError(f"not a deep challenger: {family!r}")
    return searchers[family](data, trials=trials, seed=seed, max_epochs=max_epochs)


def search_budget(family: str) -> int:
    """The frozen trial budget for one family."""
    return {
        "logistic_regression": 20,
        "xgboost": 30,
        "mlp": MLP_TRIALS,
        "ft_transformer": FT_TRANSFORMER_TRIALS,
        "tabular_resnet": TABULAR_RESNET_TRIALS,
    }[family]


def predict_proba(model: Any, data: PreparedTensors, frame: pd.DataFrame) -> np.ndarray:
    """Probabilities for any partition, using the frozen training transforms."""
    numeric, levels = data.transform(frame)
    return deep_training.predict_proba(model, numeric, levels)


def parameter_count(model: Any) -> int:
    return models.count_parameters(model)


def deep_families() -> tuple[str, ...]:
    return protocol.DEEP_FAMILIES
