"""Shared orchestration helpers for the maintained training pipelines.

The G4 decomposition split both pipelines into callable stages, but seven of
those stages came out 87-100% identical between the logistic and boosted
variants. This module is their single owner.

Everything here differs between the variants only in DATA - a variant letter, a
model name, a set of filenames, a preprocessor description - or in a parameter
name. Nothing with a real behavioural difference lives here: model construction,
Optuna objectives, cross-validation folds, SHAP explainers, drift baselines,
per-variant reporting and ``write_training_outputs`` stay with their pipelines,
because those genuinely differ.

``PipelineSpec`` carries the per-variant data. It is a frozen record of
configuration, not a framework: it has no behaviour and nothing subclasses it.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, classification_report

from ml_core import provenance, training
from ml_core.thresholds import compute_youden_threshold

#: Repository root. Identical to each pipeline's PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: ml_core modules whose contents affect a training run, hashed into manifests.
SOURCE_MODULES = (
    "evaluation.py", "bootstrap.py", "thresholds.py",
    "training.py", "pipeline.py", "feature_contract.py", "provenance.py",
)


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Per-variant configuration. Data only - no behaviour, no subclasses."""

    variant: str
    model_name: str
    #: Training script filename, relative to PROJECT_ROOT. Fingerprinted into
    #: the manifest: it owns the estimator, the search space and the writers,
    #: so a manifest without it would attest only half the producing code.
    entrypoint: str
    scaler: str | None
    filenames: Mapping[str, str]
    #: Artifact roles that serving requires; anything else is written but optional.
    serving_roles: tuple[str, ...]
    random_state: int
    test_size: float
    validation_size: float
    cv_splits: int
    calibration_method: str
    calibration_cv: int
    n_bootstrap: int
    bootstrap_alpha: float = 0.05
    optional_roles: tuple[str, ...] = field(default_factory=tuple)


# ------------------------------------------------------------------ CLI

def build_arg_parser(
    description: str | None,
    *,
    default_data_path: Path,
    default_artifacts_dir: Path,
    default_optuna_trials: int,
) -> argparse.ArgumentParser:
    """The CLI both pipelines expose. Flags and defaults are unchanged."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, default=default_data_path, metavar="CSV",
                        help="Training dataset.")
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts_dir, metavar="DIR",
                        help="Directory the model bundle and metrics are written to.")
    parser.add_argument("--optuna-trials", type=int, default=default_optuna_trials, metavar="N",
                        help="Hyperparameter search budget.")
    return parser


def parse_pipeline_args(
    argv: list[str] | None = None,
    *,
    description: str | None = None,
    default_data_path: Path,
    default_artifacts_dir: Path,
    default_optuna_trials: int,
) -> argparse.Namespace:
    """Parse a training run's arguments. --help exits before any data is read."""
    return build_arg_parser(
        description,
        default_data_path=default_data_path,
        default_artifacts_dir=default_artifacts_dir,
        default_optuna_trials=default_optuna_trials,
    ).parse_args(argv)


# ------------------------------------------------------- artifact paths

def resolve_artifact_paths(artifacts_dir: Path, *, spec: PipelineSpec) -> dict[str, Path]:
    """Resolve logical artifact roles to output paths under `artifacts_dir`.

    Each pipeline still owns its own filenames via its spec; this owns only the
    resolution. Paths are checked for escapes so a stray ``..`` or absolute
    filename cannot write outside the requested directory.
    """
    artifacts_dir = Path(artifacts_dir)
    resolved: dict[str, Path] = {}
    for role, filename in spec.filenames.items():
        candidate = artifacts_dir / filename
        if candidate.parent.resolve() != artifacts_dir.resolve():
            raise ValueError(f"artifact {role!r} would escape {artifacts_dir}: {filename!r}")
        resolved[role] = candidate
    return resolved


# ------------------------------------------------------ data preparation

def prepare_training_data(
    data_path: Path,
    *,
    spec: PipelineSpec,
    feature_names: list[str],
    target_column: str,
    feature_labels: Mapping[str, str] | None = None,
    report_class_distribution: bool = False,
    verbose: bool = True,
) -> training.TrainingSplits:
    """Load, validate against the feature contract, and split.

    ``report_class_distribution`` exists because the logistic pipeline printed
    two extra lines here and the boosted one did not. The lines sit mid-sequence
    so a caller cannot emit them afterwards; the flag keeps both pipelines'
    output byte-identical to before. It affects reporting only.
    """
    if verbose:
        print(f"\n📥 Loading dataset from {data_path}...")
    frame = training.load_training_dataset(data_path, feature_names, target_column)
    if verbose:
        print(f"✅ Data loaded: {frame.shape[0]:,} rows, {frame.shape[1]} columns")

    X, y = training.select_features(frame, feature_names, target_column)

    if verbose:
        labels = feature_labels or {}
        print(f"\n📊 Selected Features ({len(feature_names)}):")
        for feat in feature_names:
            print(f"   - {feat}: {labels.get(feat, feat)}")
        if report_class_distribution:
            print(f"\n🎯 Target: {target_column}")
            print(f"   Class distribution: {dict(y.value_counts())}")
        print("\n✂️ Splitting data: 60% train / 20% validation / 20% test (stratified)...")

    splits = training.split_training_data(
        X, y,
        test_size=spec.test_size,
        val_size=spec.validation_size,
        random_state=spec.random_state,
    )
    if verbose:
        for part, size in splits.sizes.items():
            print(f"   {part.capitalize()}: {size:,} samples")
    return splits


# -------------------------------------------------- threshold + evaluation

def select_threshold(model, splits: training.TrainingSplits) -> tuple[float, dict]:
    """Youden's J on the VALIDATION set, so the test set stays untouched.

    Model-agnostic: it needs only predict_proba, which is why the logistic and
    boosted versions were identical apart from a parameter name.
    """
    print("\n🎯 Computing optimal threshold using Youden's J on validation set...")
    val_proba = training.positive_class_proba(model, splits.X_val)
    best_threshold = compute_youden_threshold(splits.y_val.values, val_proba)
    print(f"   Best threshold (Youden's J): {best_threshold:.4f}")

    _, val_metrics = training.evaluate_at_threshold(splits.y_val.values, val_proba, best_threshold)
    print(f"   Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"   Validation F1: {val_metrics['f1']:.4f}")
    return best_threshold, val_metrics


def evaluate_on_test(
    model, splits: training.TrainingSplits, threshold: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Score the held-out test set and print the standard report."""
    print("\n🔍 Evaluating on held-out TEST set...")
    test_proba = training.positive_class_proba(model, splits.X_test)
    test_pred, test_metrics = training.evaluate_at_threshold(
        splits.y_test.values, test_proba, threshold
    )

    print("\n📊 TEST SET METRICS:")
    for label, key in (("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc"),
                       ("Cohen's Kappa", "cohen_kappa"), ("MCC", "mcc")):
        print(f"   {label + ':':15s}{test_metrics[key]:.4f}")
    cm = test_metrics["confusion_matrix"]
    print("\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:5d}  FP={cm[0][1]:5d}]")
    print(f"    [FN={cm[1][0]:5d}  TP={cm[1][1]:5d}]]")
    print("\n   Classification Report:")
    print(classification_report(
        splits.y_test, test_pred, digits=4, target_names=["No Diabetes", "Diabetes"]
    ))
    return test_proba, test_pred, test_metrics


# --------------------------------------------------------- calibration

def calibrate_estimator(
    estimator,
    splits: training.TrainingSplits,
    threshold: float,
    uncalibrated_test_proba: np.ndarray,
    *,
    method: str,
    cv: int,
) -> tuple[CalibratedClassifierCV, np.ndarray, np.ndarray, dict, float, float]:
    """Platt-scale over train+validation, then re-evaluate on the test set.

    Works for a bare estimator or a Pipeline - CalibratedClassifierCV treats
    both the same, which is why the two variants' versions were identical apart
    from a local variable name.
    """
    print("\n🎯 Calibrating probabilities (Platt scaling on validation set)...")
    calibrated = CalibratedClassifierCV(estimator, cv=cv, method=method)
    calibrated.fit(
        pd.concat([splits.X_train, splits.X_val]),
        pd.concat([splits.y_train, splits.y_val]),
    )

    cal_proba = training.positive_class_proba(calibrated, splits.X_test)
    brier_before = brier_score_loss(splits.y_test, uncalibrated_test_proba)
    brier_after = brier_score_loss(splits.y_test, cal_proba)
    print(f"   Brier score (before calibration): {brier_before:.4f}")
    print(f"   Brier score (after calibration):  {brier_after:.4f}")

    test_pred_final, test_metrics = training.evaluate_at_threshold(
        splits.y_test.values, cal_proba, threshold
    )
    print(f"   Calibrated ROC-AUC: {test_metrics['roc_auc']:.4f}")
    return calibrated, cal_proba, test_pred_final, test_metrics, brier_before, brier_after


# ---------------------------------------------------------- provenance

def attested_source_files(spec: PipelineSpec, project_root: Path) -> list[Path]:
    """Source files a manifest fingerprints: the entrypoint plus the ml_core modules.

    The entrypoint is listed explicitly because this module cannot infer its
    caller. Dropping it would quietly shrink what the manifest attests, leaving
    the estimator, search space and writers unhashed.

    A run rooted outside the repository attests no source at all - unchanged
    behaviour, and the reason a manifest written into a scratch tree stays
    honest about what it can observe.
    """
    if Path(project_root) != PROJECT_ROOT:
        return []
    return [PROJECT_ROOT / spec.entrypoint] + [
        PROJECT_ROOT / "ml_core" / name for name in SOURCE_MODULES
    ]


def emit_pipeline_provenance(
    spec: PipelineSpec,
    paths: Mapping[str, Path],
    *,
    data_path: Path,
    artifacts_dir: Path,
    feature_names: list[str],
    target_column: str,
    best_params: dict,
    best_cv_auc: float,
    threshold: float,
    n_trials: int,
    val_metrics: dict,
    test_metrics: dict,
    ci_results: dict,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Emit the training manifest LAST, once every artifact is on disk.

    A thin, parameterised call into ml_core.provenance - the provenance
    internals stay there. Nothing about containment, dirty-tree reporting,
    relative paths, environment capture or write-last ordering changes here.
    """
    artifact_specs = [
        (role, paths[role], role in spec.serving_roles)
        for role in spec.filenames
        if role != "provenance" and role in paths
    ]
    source_files = attested_source_files(spec, project_root)
    lockfile = PROJECT_ROOT / "requirements.lock" if Path(project_root) == PROJECT_ROOT else None

    written = provenance.emit_training_manifest(
        project_root=project_root,
        output_path=paths["provenance"],
        variant=spec.variant,
        model_name=spec.model_name,
        dataset_path=data_path,
        target_column=target_column,
        # Straight from the canonical contract, so the schema hash can never
        # describe a feature list the models were not trained on.
        feature_names=list(feature_names),
        training={
            "random_state": spec.random_state,
            "test_size": spec.test_size,
            "validation_size_of_train": spec.validation_size,
            "stratified": True,
            "scaler": spec.scaler,
            "optuna_sampler": "TPESampler",
            "optuna_sampler_seed": spec.random_state,
            "optuna_n_trials": n_trials,
            "optuna_direction": "maximize",
            "optuna_best_params": best_params,
            "optuna_best_cv_auc": best_cv_auc,
            "cv_splits": spec.cv_splits,
            "calibration_method": spec.calibration_method,
            "calibration_cv": spec.calibration_cv,
            "threshold_method": "youden_j",
            "selected_threshold": threshold,
            "n_bootstrap": spec.n_bootstrap,
            "bootstrap_alpha": spec.bootstrap_alpha,
            "artifacts_dir": provenance.relative_path(artifacts_dir, project_root),
        },
        evaluation={
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "confidence_intervals": ci_results,
        },
        artifact_specs=artifact_specs,
        source_files=source_files,
        lockfile=lockfile,
    )
    print(f"💾 Provenance manifest saved: {written}")
    return written
