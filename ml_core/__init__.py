"""Model-agnostic evaluation utilities shared by the maintained training pipelines.

This package holds only reusable, pure logic: metric computation, bootstrap
confidence intervals and threshold selection. Training orchestration - data
loading, Optuna studies, model fitting, calibration, SHAP and artifact writing -
stays in the pipeline scripts.

Nothing here reads a dataset, trains a model, writes a file, prints, plots or
mutates global state (including warnings filters and the global NumPy RNG), so
it can be imported and tested in isolation from any working directory.
"""
from ml_core.bootstrap import BOOTSTRAP_METRICS, bootstrap_confidence_interval
from ml_core.evaluation import EVALUATION_KEYS, evaluate_predictions
from ml_core.thresholds import compute_youden_threshold

__all__ = [
    "BOOTSTRAP_METRICS",
    "EVALUATION_KEYS",
    "bootstrap_confidence_interval",
    "compute_youden_threshold",
    "evaluate_predictions",
]
