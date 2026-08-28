"""Fresh classical baselines, trained under the Track K protocol.

These are NOT the deployed artifacts. The models in model_artifacts/ predate
this project's provenance system and their attestation records their lineage as
unknown, so comparing a new network against them would compare things that
differ in more ways than architecture. These instances are trained here, on the
frozen split, with recorded seeds and known lineage.

The search budgets below are fixed and committed before any test-set evaluation.
They are deliberately modest: the goal is a fair development process for every
family, not a leaderboard. Both searches optimise validation ROC-AUC and never
see the test partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml_core import training as core_training
from research.track_k import protocol

#: Frozen search budgets. Small on purpose - see the module docstring.
LOGISTIC_TRIALS: int = 20
XGBOOST_TRIALS: int = 30

#: Silence Optuna's per-trial chatter; the study result is what gets recorded.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """What a validation-only search decided, recorded into provenance."""

    family: str
    trials: int
    best_params: dict[str, Any]
    best_validation_score: float
    metric: str = protocol.PRIMARY_METRIC

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "trials": self.trials,
            "best_params": self.best_params,
            "best_validation_score": self.best_validation_score,
            "metric": self.metric,
            "searched_on": "train+validation",
        }


def _validation_roc_auc(model: Any, splits: core_training.TrainingSplits) -> float:
    """Fit on train, score on validation. The test partition is never touched."""
    from sklearn.metrics import roc_auc_score

    model.fit(splits.X_train, splits.y_train)
    proba = core_training.positive_class_proba(model, splits.X_val)
    return float(roc_auc_score(splits.y_val.to_numpy(), proba))


def build_logistic(params: dict[str, Any], seed: int) -> Pipeline:
    """Standardised logistic regression.

    A scaler is included because a penalised linear model is scale-sensitive;
    it is fitted inside the pipeline, so it sees only the training partition
    when the pipeline is fitted on train.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=params.get("C", 1.0),
                    solver=params.get("solver", "lbfgs"),
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_xgboost(params: dict[str, Any], seed: int) -> XGBClassifier:
    """Gradient-boosted trees. No scaling: trees are invariant to it."""
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def search_logistic(
    splits: core_training.TrainingSplits, *, trials: int = LOGISTIC_TRIALS, seed: int
) -> SearchOutcome:
    """Tune regularisation strength and solver on validation ROC-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        }
        return _validation_roc_auc(build_logistic(params, seed), splits)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return SearchOutcome(
        family="logistic_regression",
        trials=trials,
        best_params=dict(study.best_params),
        best_validation_score=float(study.best_value),
    )


def search_xgboost(
    splits: core_training.TrainingSplits, *, trials: int = XGBOOST_TRIALS, seed: int
) -> SearchOutcome:
    """Tune depth, rate and regularisation on validation ROC-AUC.

    The space is the one the repository's existing boosted pipeline already
    searches, so the Track K baseline is a fair representative of what this
    project actually knows how to build rather than a deliberately weak foil.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        }
        return _validation_roc_auc(build_xgboost(params, seed), splits)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return SearchOutcome(
        family="xgboost",
        trials=trials,
        best_params=dict(study.best_params),
        best_validation_score=float(study.best_value),
    )


def fit_final(family: str, params: dict[str, Any], splits: core_training.TrainingSplits, seed: int) -> Any:
    """Fit the chosen configuration on the TRAINING partition only.

    Deliberately not train+validation. Validation selected the threshold and any
    calibrator, so refitting on it would make those choices self-referential and
    quietly optimistic.
    """
    builder = {"logistic_regression": build_logistic, "xgboost": build_xgboost}[family]
    model = builder(params, seed)
    model.fit(splits.X_train, splits.y_train)
    return model


def predict_proba(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities, via the shared helper both pipelines use."""
    return core_training.positive_class_proba(model, frame)
