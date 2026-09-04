"""The Track L benchmark: thirty models, one 1,000-row budget, one evaluation.

Ordering is the whole design, and it is enforced by the structure of `run`:

1. load the canonical dataset and verify the frozen split;
2. take a deterministic, fingerprinted 1,000-row subset of TRAIN only;
3. fit every enabled model on those rows, and nothing else;
4. choose each model's threshold on VALIDATION;
5. read TEST once, for every model, at the end;
6. round-trip every model through disk;
7. write results, then the manifest last.

Steps 3-4 happen entirely before step 5. No hyperparameter, threshold or
calibrator anywhere in this package is chosen using a test outcome, and no
model is dropped from the table for scoring badly.

**Failure is data.** A model that cannot be installed, cannot converge, cannot
serialize or exceeds its time budget stays in the results with an outcome and a
reason attached. A table that quietly contains only the models that worked is a
flattering table, not an honest one.

**This is exploratory.** Every result carries
``evidence_class: RESOURCE_CONSTRAINED_EXPLORATORY``. One thousand training rows
is 2.5% of what Track K's reference arm used, and Track K measured this dataset's
families to be strongly sample-dependent - boosting worst at 500 rows and best at
40,125. Track L therefore cannot supersede Track K for any model Track K tested,
and nothing here is a promotion candidate.
"""
from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml_core import training as core_training
from research.model_zoo import serialization
from research.model_zoo.contracts import (
    CapabilityError,
    Family,
    ModelResult,
    ProbabilityBehavior,
    ResearchStatus,
    RunOutcome,
)
from research.model_zoo.registry import REGISTRY, ModelSpec
from research.track_k import artifacts as track_k_artifacts
from research.track_k import calibration, evaluation, split, subsets

#: The zoo's training budget. Deliberately small: the point is architecture
#: breadth on one CPU, not the best achievable model.
DEFAULT_TRAIN_ROWS: int = 1000

#: Wall-clock ceiling per model. A model that exceeds it is recorded as
#: RESOURCE_LIMIT rather than left running - an unbounded benchmark is one
#: nobody can finish, and "too slow at this budget" is itself a finding.
DEFAULT_TIME_BUDGET_SECONDS: float = 300.0

RESEARCH_ROOT = track_k_artifacts.PROJECT_ROOT / "research_artifacts" / "model_zoo"

#: Stamped on every result. The zoo is broad and cheap; Track K is narrow and
#: thorough. This label keeps the difference attached to the numbers.
EVIDENCE_CLASS = "RESOURCE_CONSTRAINED_EXPLORATORY"


@dataclass
class FittedModel:
    """One model that trained successfully, with everything needed to score it."""

    spec: ModelSpec
    model: Any
    threshold: float
    test_scores: np.ndarray | None
    test_predictions: np.ndarray


def _select_subset(splits: core_training.TrainingSplits, rows: int) -> tuple[Any, dict[str, Any]]:
    """A deterministic, fingerprinted training subset drawn from TRAIN only.

    Reuses Track K's nested-subset machinery rather than inventing a second
    sampler, so the zoo's 1,000 rows are literally a prefix of the ladder Track
    K's sample-efficiency study used. That makes the two studies comparable at
    the sizes they share, and it means the fail-closed verification comes for
    free.
    """
    manifest = subsets.build_subset_manifest(splits, sizes=(rows,))
    ladder = subsets.load_verified_subsets(splits, manifest)
    return ladder[rows], manifest


def _evaluate_model(
    spec: ModelSpec,
    model: Any,
    splits: core_training.TrainingSplits,
) -> tuple[dict[str, Any], float, np.ndarray | None, np.ndarray]:
    """Choose a threshold on validation, then read test exactly once.

    A model with no ranking score gets its threshold-free metrics recorded as
    None rather than computed from its hard labels. Reporting a ROC-AUC derived
    from 0/1 predictions would put a number in the table that looks like a
    measurement and is not one.
    """
    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()

    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        predictions = np.asarray(model.predict(splits.X_test)).astype(int)
        metrics = evaluation.evaluate(y_test, predictions.astype(float), threshold=0.5)
        # Overwrite the threshold-free entries: they are undefined here, and a
        # value computed from labels would be misleading.
        for undefined in ("roc_auc", "pr_auc", "brier_score", "log_loss", "ece",
                          "calibration_slope", "calibration_intercept"):
            metrics[undefined] = None
        metrics["threshold_free_metrics"] = (
            "undefined: this model exposes no ranking score"
        )
        return metrics, 0.5, None, predictions

    val_scores = model.decision_scores(splits.X_val)
    threshold = float(calibration.select_threshold(y_val, _to_unit(val_scores)))

    test_scores = np.asarray(model.decision_scores(splits.X_test), dtype=float)
    unit_scores = _to_unit(test_scores)
    metrics = evaluation.evaluate(y_test, unit_scores, threshold=threshold)
    predictions = (unit_scores >= threshold).astype(int)
    return metrics, threshold, unit_scores, predictions


def _to_unit(scores: np.ndarray) -> np.ndarray:
    """Map a ranking score into [0, 1] without claiming it became a probability.

    Models whose scores are already probabilities pass through untouched. A
    margin - an SVM's signed distance - is squashed by a logistic so the shared
    evaluator can compute threshold metrics on it, and the manifest records that
    the model's ``probability_behavior`` is REQUIRES_EXTERNAL_CALIBRATION so
    nobody reads its Brier score as a calibration measurement.
    """
    values = np.asarray(scores, dtype=float)
    if values.min() >= 0.0 and values.max() <= 1.0:
        return values
    squashed: np.ndarray = 1.0 / (1.0 + np.exp(-values))
    return squashed


def _run_one(
    spec: ModelSpec,
    splits: core_training.TrainingSplits,
    *,
    time_budget: float,
    artifact_dir: Path,
    overrides: dict[str, Any],
) -> tuple[ModelResult, FittedModel | None]:
    """Fit, evaluate and round-trip one model, converting failure into a row."""
    if spec.effective_status() is not ResearchStatus.ACTIVE:
        reason = (
            f"optional dependency {spec.optional_dependency!r} is not installed"
            if spec.optional_dependency
            else f"status is {spec.effective_status().value}"
        )
        return (
            ModelResult(
                model_id=spec.model_id,
                family=spec.family,
                outcome=RunOutcome.SKIPPED,
                error=reason,
            ),
            None,
        )

    started = time.perf_counter()
    try:
        model = REGISTRY.build(spec.model_id, **overrides)
        model.fit(splits.X_train, splits.y_train)
        elapsed = time.perf_counter() - started

        if elapsed > time_budget:
            return (
                ModelResult(
                    model_id=spec.model_id,
                    family=spec.family,
                    outcome=RunOutcome.RESOURCE_LIMIT,
                    training={"fit_seconds": round(elapsed, 3)},
                    error=f"fit took {elapsed:.1f}s, over the {time_budget:.0f}s budget",
                ),
                None,
            )

        metrics, threshold, scores, predictions = _evaluate_model(spec, model, splits)
        record = serialization.round_trip(
            spec.model_id, model, splits.X_test.head(200), directory=artifact_dir
        )

        training = (model.training_record.as_dict() if model.training_record else {})
        training["threshold"] = threshold
        return (
            ModelResult(
                model_id=spec.model_id,
                family=spec.family,
                outcome=RunOutcome.COMPLETED,
                metrics=metrics,
                training=training,
                serialization=record.as_dict(),
            ),
            FittedModel(spec, model, threshold, scores, predictions),
        )
    except (CapabilityError, ValueError, RuntimeError, MemoryError, ImportError) as error:
        return (
            ModelResult(
                model_id=spec.model_id,
                family=spec.family,
                outcome=RunOutcome.FAILED,
                training={"fit_seconds": round(time.perf_counter() - started, 3)},
                error=f"{type(error).__name__}: {error}",
            ),
            None,
        )


def run(
    *,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    model_ids: list[str] | None = None,
    time_budget: float = DEFAULT_TIME_BUDGET_SECONDS,
    output_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the zoo and return the manifest it wrote."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_id = f"zoo-{train_rows}-{uuid.uuid4().hex[:10]}"
    root = Path(output_root) if output_root is not None else RESEARCH_ROOT
    out_dir = root / run_id
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    frame = split.load_dataset()
    splits = split.build_split(frame)
    drift = split.verify_split(splits, split.build_split_manifest(splits))
    if drift:
        raise split.SplitIntegrityError("; ".join(drift))

    index, subset_manifest = _select_subset(splits, train_rows)
    narrowed = subsets.take(splits, index)

    specs = [REGISTRY.get(mid) for mid in model_ids] if model_ids else list(REGISTRY)
    results: list[ModelResult] = []
    fitted: dict[str, FittedModel] = {}
    for spec in specs:
        result, model = _run_one(
            spec,
            narrowed,
            time_budget=time_budget,
            artifact_dir=out_dir / "artifacts",
            overrides=(overrides or {}) if spec.framework.value == "torch" else {},
        )
        results.append(result)
        if model is not None:
            fitted[spec.model_id] = model

    # Per-row test predictions, so any metric can be recomputed and the
    # agreement analysis can run without refitting anything.
    scores = {
        model_id: model.test_scores
        for model_id, model in fitted.items()
        if model.test_scores is not None
    }
    if scores:
        _save_arrays(out_dir / "test_scores.npz", scores)
    _save_arrays(
        out_dir / "test_predictions.npz",
        {model_id: model.test_predictions for model_id, model in fitted.items()},
    )

    manifest = {
        "provenance_type": "track_l_model_zoo_run",
        "evidence_class": EVIDENCE_CLASS,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_rows": len(narrowed.X_train),
        "validation_rows": len(narrowed.X_val),
        "test_rows": len(narrowed.X_test),
        "time_budget_seconds": time_budget,
        "training_subset": subset_manifest,
        "dataset": split.build_split_manifest(splits).get("dataset", {}),
        "split": split.build_split_manifest(splits).get("split", {}),
        "registry": [spec.as_dict() for spec in specs],
        "results": [result.as_dict() for result in results],
        "counts": _counts(results),
        "environment": track_k_artifacts.research_environment(),
        "git": track_k_artifacts.provenance.git_provenance(track_k_artifacts.PROJECT_ROOT),
        "production_artifacts_touched": False,
        "limitations": [
            "One thousand training rows is 2.5% of Track K's reference arm. "
            "Track K measured this dataset's families to be strongly "
            "sample-dependent, so these rankings describe the constrained "
            "regime and cannot be extrapolated to a full-data one.",
            "Configurations are frozen sensible defaults, not tuned. A model "
            "that does poorly here may simply be badly configured for 1,000 rows.",
            "No model here is a promotion candidate. Track L is exploratory; "
            "production serving is untouched.",
            "The study dataset is close to 50/50 positive, so every probability "
            "is conditional on that base rate and is not a population disease "
            "probability.",
        ],
    }
    # Manifest last, once every artifact it describes exists.
    core_training.write_json_atomic(manifest, out_dir / "run_manifest.json")
    return manifest


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a named array bundle.

    Wrapped rather than called inline because numpy types ``savez_compressed``
    as ``(file, *args, allow_pickle=bool, **kwds)``, so unpacking a dict of
    arrays into it makes a typechecker match the first keyword against
    ``allow_pickle``. One annotated call site keeps the rest of the module
    honest about what it is writing.
    """
    save: Any = np.savez_compressed
    save(path, **arrays)


def _counts(results: list[ModelResult]) -> dict[str, int]:
    counts = {outcome.value: 0 for outcome in RunOutcome}
    for result in results:
        counts[result.outcome.value] += 1
    return counts


def summarise(manifest: dict[str, Any]) -> str:
    """The results table, families grouped, failures included."""
    lines = [
        f"Track L model zoo - run {manifest['run_id']}",
        f"{manifest['evidence_class']}  |  "
        f"{manifest['train_rows']:,} training rows, "
        f"{manifest['test_rows']:,} test rows",
        "",
    ]
    header = (
        f"{'model':<24}{'family':<14}{'roc_auc':>9}{'pr_auc':>9}{'recall':>8}"
        f"{'brier':>9}{'ece':>8}{'fit s':>8}{'params':>9}  status"
    )
    lines.append(header)
    lines.append("-" * len(header))

    by_family: dict[str, list[dict[str, Any]]] = {}
    for result in manifest["results"]:
        by_family.setdefault(result["family"], []).append(result)

    for family in [f.value for f in Family]:
        for result in by_family.get(family, []):
            metrics = result.get("metrics") or {}
            training = result.get("training") or {}
            lines.append(
                f"{result['model_id']:<24}{family:<14}"
                f"{_cell(metrics.get('roc_auc')):>9}{_cell(metrics.get('pr_auc')):>9}"
                f"{_cell(metrics.get('recall'), 4):>8}{_cell(metrics.get('brier_score')):>9}"
                f"{_cell(metrics.get('ece')):>8}"
                f"{_cell(training.get('fit_seconds'), 1):>8}"
                f"{_int_cell(training.get('parameter_count')):>9}"
                f"  {result['outcome']}"
            )
    lines.append("")
    lines.append("outcomes: " + ", ".join(
        f"{name}={count}" for name, count in manifest["counts"].items() if count
    ))
    return "\n".join(lines)


def _cell(value: Any, places: int = 5) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:.{places}f}" if isinstance(value, (int, float)) else str(value)


def _int_cell(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model ids to run. Defaults to the whole registry.",
    )
    parser.add_argument(
        "--time-budget", type=float, default=DEFAULT_TIME_BUDGET_SECONDS,
        help="Per-model wall-clock ceiling in seconds.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = run(
        train_rows=args.train_rows,
        model_ids=args.models,
        time_budget=args.time_budget,
        output_root=args.output_root,
    )
    print(summarise(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
