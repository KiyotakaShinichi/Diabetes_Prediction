"""The Track K benchmark: one command, four families, one held-out evaluation.

Order matters and is enforced by the structure of ``run``:

1. validate the dataset and the feature contract;
2. derive the frozen split and fail closed if it has drifted;
3. for each family: search on validation, fit on train, choose a threshold and a
   calibrator on validation;
4. only then touch the test partition, once, for every family;
5. bootstrap, compare, apply the frozen promotion policy;
6. write artifacts, then the manifest last.

Steps 3 and 4 are separated deliberately. Every model-selection decision is made
before any test row is read, so the final numbers are a measurement rather than
the best of several looks.

Two modes. ``--smoke`` runs a tiny deterministic configuration to prove the
plumbing, in seconds, and is what CI executes; its numbers are never research
results and the manifest it writes says so. The full run is explicit and takes
minutes.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_core import training as core_training
from research.track_k import (
    artifacts,
    baselines,
    calibration,
    challengers,
    comparison,
    evaluation,
    protocol,
    split,
)

#: Rows timed for the single-row latency figure. Enough to make a median stable
#: without turning the benchmark into a microbenchmark suite.
LATENCY_SAMPLES = 200
LATENCY_BATCH = 1000


@dataclass
class FamilyResult:
    """One family's complete result, before manifests are written."""

    family: str
    record: artifacts.ModelRecord
    test_proba: np.ndarray
    test_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    reliability: list[dict[str, Any]]


def _timed_latency(predict: Any, frame: pd.DataFrame) -> dict[str, Any]:
    """Median single-row and batch latency on CPU.

    Comparative, not absolute: one machine, one process, no isolation from
    other load. Reported so a metric gain can be weighed against serving cost,
    and documented as such rather than presented as a benchmark of the model.
    """
    single = frame.iloc[:1]
    timings = []
    for _ in range(LATENCY_SAMPLES):
        start = time.perf_counter()
        predict(single)
        timings.append(time.perf_counter() - start)

    batch = frame.iloc[:LATENCY_BATCH]
    batch_start = time.perf_counter()
    predict(batch)
    batch_elapsed = time.perf_counter() - batch_start

    return {
        "single_row_median_ms": float(np.median(timings) * 1000),
        "single_row_p95_ms": float(np.percentile(timings, 95) * 1000),
        "batch_rows": len(batch),
        "batch_total_ms": float(batch_elapsed * 1000),
        "batch_per_row_ms": float(batch_elapsed * 1000 / max(len(batch), 1)),
        "method": (
            f"median of {LATENCY_SAMPLES} single-row calls and one "
            f"{LATENCY_BATCH}-row batch, CPU, one machine, comparative only"
        ),
    }


def _run_classical(
    family: str, splits: core_training.TrainingSplits, *, smoke: bool, out_dir: Path
) -> FamilyResult:
    seed = protocol.model_seed(family)
    trials = 2 if smoke else challengers.search_budget(family)

    search = (
        baselines.search_logistic(splits, trials=trials, seed=seed)
        if family == "logistic_regression"
        else baselines.search_xgboost(splits, trials=trials, seed=seed)
    )
    model = baselines.fit_final(family, search.best_params, splits, seed)

    val_proba = baselines.predict_proba(model, splits.X_val)
    threshold = calibration.select_threshold(splits.y_val.to_numpy(), val_proba)
    calibrator, cal_outcome = calibration.select_calibrator(
        splits.y_val.to_numpy(), val_proba
    )

    # ---- the test partition is read here, and only here, for this family ----
    test_proba = calibrator(baselines.predict_proba(model, splits.X_test))
    test_metrics = evaluation.evaluate(
        splits.y_test.to_numpy(), test_proba, threshold=threshold
    )
    val_metrics = evaluation.evaluate(
        splits.y_val.to_numpy(), calibrator(val_proba), threshold=threshold
    )

    import joblib

    model_path = out_dir / f"{family}_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, model_path)

    resources = _timed_latency(
        lambda frame: calibrator(baselines.predict_proba(model, frame)), splits.X_test
    )
    resources["artifact_bytes"] = int(model_path.stat().st_size)

    relative, hashes = artifacts.inventory({"model": model_path}, out_dir)
    return FamilyResult(
        family=family,
        record=artifacts.ModelRecord(
            family=family,
            seed=seed,
            config=dict(search.best_params),
            search=search.as_dict(),
            calibration=cal_outcome.as_dict(),
            threshold=threshold,
            parameter_count=None,
            training={"fitted_on": "train", "smoke": smoke},
            artifact_paths=relative,
            artifact_hashes=hashes,
            resources=resources,
        ),
        test_proba=test_proba,
        test_metrics=test_metrics,
        validation_metrics=val_metrics,
        reliability=[
            item.as_dict()
            for item in evaluation.reliability_bins(splits.y_test.to_numpy(), test_proba)
        ],
    )


def _run_deep(
    family: str, splits: core_training.TrainingSplits, *, smoke: bool, out_dir: Path
) -> FamilyResult:
    import torch

    seed = protocol.model_seed(family)
    data = challengers.prepare(splits)

    if smoke:
        params = (
            {"hidden_dims": [16], "batch_size": 512}
            if family == "mlp"
            else {"d_token": 8, "n_blocks": 1, "n_heads": 2, "batch_size": 512}
        )
        search = baselines.SearchOutcome(
            family=family, trials=0, best_params=params, best_validation_score=float("nan")
        )
        epochs = 2
    else:
        search = (
            challengers.search_mlp(data, trials=challengers.MLP_TRIALS, seed=seed)
            if family == "mlp"
            else challengers.search_ft_transformer(
                data, trials=challengers.FT_TRANSFORMER_TRIALS, seed=seed
            )
        )
        params = search.best_params
        epochs = challengers.FINAL_MAX_EPOCHS

    model, training_result = challengers.train_challenger(
        family, params, data, seed=seed, max_epochs=epochs
    )

    val_proba = challengers.predict_proba(model, data, splits.X_val)
    threshold = calibration.select_threshold(splits.y_val.to_numpy(), val_proba)
    calibrator, cal_outcome = calibration.select_calibrator(
        splits.y_val.to_numpy(), val_proba
    )

    # ---- the test partition is read here, and only here, for this family ----
    test_proba = calibrator(challengers.predict_proba(model, data, splits.X_test))
    test_metrics = evaluation.evaluate(
        splits.y_test.to_numpy(), test_proba, threshold=threshold
    )
    val_metrics = evaluation.evaluate(
        splits.y_val.to_numpy(), calibrator(val_proba), threshold=threshold
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{family}_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    curve_path = out_dir / f"{family}_learning_curve.json"
    core_training.write_json_atomic(
        {"family": family, "history": [r.as_dict() for r in training_result.history]},
        curve_path,
    )

    resources = _timed_latency(
        lambda frame: calibrator(challengers.predict_proba(model, data, frame)),
        splits.X_test,
    )
    resources["artifact_bytes"] = int(checkpoint_path.stat().st_size)

    relative, hashes = artifacts.inventory(
        {"checkpoint": checkpoint_path, "learning_curve": curve_path}, out_dir
    )
    return FamilyResult(
        family=family,
        record=artifacts.ModelRecord(
            family=family,
            seed=seed,
            config=dict(params),
            search=search.as_dict(),
            calibration=cal_outcome.as_dict(),
            threshold=threshold,
            parameter_count=challengers.parameter_count(model),
            training={
                "fitted_on": "train",
                "smoke": smoke,
                **{k: v for k, v in training_result.as_dict().items() if k != "history"},
                "preprocessing": data.standardiser.as_dict(),
            },
            artifact_paths=relative,
            artifact_hashes=hashes,
            resources=resources,
        ),
        test_proba=test_proba,
        test_metrics=test_metrics,
        validation_metrics=val_metrics,
        reliability=[
            item.as_dict()
            for item in evaluation.reliability_bins(splits.y_test.to_numpy(), test_proba)
        ],
    )


def error_analysis(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, frame: pd.DataFrame
) -> dict[str, Any]:
    """Where a model is wrong, and how confidently.

    Restricted to what the dataset actually contains. The served contract holds
    no demographic attribute beyond age band and education, so no fairness
    analysis is invented for attributes that are not there.
    """
    y_true = np.asarray(y_true).astype(int)
    predicted = (proba >= threshold).astype(int)
    false_positive = (predicted == 1) & (y_true == 0)
    false_negative = (predicted == 0) & (y_true == 1)

    def summarise(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"count": 0}
        return {
            "count": int(mask.sum()),
            "mean_probability": float(proba[mask].mean()),
            "mean_general_health": float(frame.loc[mask, "GenHlth"].mean()),
            "mean_bmi": float(frame.loc[mask, "BMI"].mean()),
            "mean_age_band": float(frame.loc[mask, "Age"].mean()),
            "high_blood_pressure_rate": float(frame.loc[mask, "HighBP"].mean()),
        }

    confident_wrong = ((proba >= 0.9) & (y_true == 0)) | ((proba <= 0.1) & (y_true == 1))
    uncertain = (proba > 0.4) & (proba < 0.6)

    return {
        "false_positives": summarise(false_positive),
        "false_negatives": summarise(false_negative),
        "confidently_wrong": {
            "count": int(confident_wrong.sum()),
            "definition": "predicted >=0.9 while negative, or <=0.1 while positive",
        },
        "uncertain_band": {
            "count": int(uncertain.sum()),
            "accuracy": float((predicted[uncertain] == y_true[uncertain]).mean())
            if uncertain.any()
            else float("nan"),
            "definition": "predicted probability strictly between 0.4 and 0.6",
        },
        "subgroups_note": (
            "No demographic fairness analysis is reported: the served feature "
            "contract contains no protected attribute, and inventing one would "
            "not be supported by this dataset."
        ),
    }


def run(*, smoke: bool = False, output_root: Path | None = None) -> dict[str, Any]:
    """Execute the benchmark and return the manifest it wrote."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_id = f"{'smoke' if smoke else 'full'}-{uuid.uuid4().hex[:12]}"
    root = Path(output_root) if output_root is not None else artifacts.RESEARCH_ROOT
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = split.load_dataset()
    splits = split.build_split(frame)
    split_manifest = split.build_split_manifest(splits)
    # Fail closed: refuse to benchmark rows that are not the frozen ones.
    drift = split.verify_split(splits, split_manifest)
    if drift:
        raise split.SplitIntegrityError("; ".join(drift))
    split.write_split_manifest(splits, out_dir / "split_manifest.json")

    results: list[FamilyResult] = []
    for family in protocol.MODEL_FAMILIES:
        runner = _run_deep if protocol.is_deep(family) else _run_classical
        results.append(runner(family, splits, smoke=smoke, out_dir=out_dir))

    predictions = {result.family: result.test_proba for result in results}
    y_test = splits.y_test.to_numpy()

    resamples = 50 if smoke else protocol.BOOTSTRAP_RESAMPLES
    intervals, replicates = comparison.bootstrap_metrics(
        y_test, predictions, resamples=resamples
    )
    comparisons = [
        item.as_dict()
        for item in comparison.compare_all(replicates, comparison.default_pairs())
    ]

    by_family = {result.family: result for result in results}
    best_classical = max(
        protocol.CLASSICAL_FAMILIES,
        key=lambda name: by_family[name].test_metrics[protocol.PRIMARY_METRIC],
    )
    baseline_result = by_family[best_classical]

    promotion: dict[str, Any] = {"baseline": best_classical, "decisions": {}}
    for challenger in protocol.DEEP_FAMILIES:
        challenger_result = by_family[challenger]
        delta = comparison.paired_delta(
            replicates, challenger, best_classical, protocol.PRIMARY_METRIC
        )
        verdict, reasons = comparison.promotion_verdict(
            primary_delta=delta,
            ece_delta=challenger_result.test_metrics["ece"]
            - baseline_result.test_metrics["ece"],
            recall_delta=challenger_result.test_metrics["recall"]
            - baseline_result.test_metrics["recall"],
            latency_multiple=(
                challenger_result.record.resources["single_row_median_ms"]
                / max(baseline_result.record.resources["single_row_median_ms"], 1e-9)
            ),
        )
        promotion["decisions"][challenger] = {
            "verdict": verdict,
            "reasons": reasons,
            "primary_delta": delta.as_dict(),
        }

    for result in results:
        core_training.write_json_atomic(
            {
                "family": result.family,
                "smoke": smoke,
                "test_metrics": result.test_metrics,
                "validation_metrics": result.validation_metrics,
                "reliability_bins": result.reliability,
                "bootstrap": {
                    metric: interval.as_dict()
                    for metric, interval in intervals[result.family].items()
                },
                "error_analysis": error_analysis(
                    y_test, result.test_proba, result.record.threshold, splits.X_test
                ),
            },
            out_dir / f"{result.family}_metrics.json",
        )
        np.save(out_dir / f"{result.family}_test_proba.npy", result.test_proba)
        result.record.artifact_paths["metrics"] = f"{result.family}_metrics.json"

    manifest = artifacts.build_run_manifest(
        run_id=run_id,
        split_manifest=split_manifest,
        models=[result.record for result in results],
        comparisons=comparisons,
        promotion=promotion,
        bootstrap={
            "resamples": resamples,
            "alpha": protocol.BOOTSTRAP_ALPHA,
            "seed": protocol.BOOTSTRAP_SEED,
            "paired": True,
            "intervals": {
                family: {m: i.as_dict() for m, i in metrics.items()}
                for family, metrics in intervals.items()
            },
        },
        started_at=started_at,
        root=out_dir,
    )
    manifest["smoke"] = smoke
    if smoke:
        manifest["warning"] = (
            "SMOKE RUN - tiny configurations and 50 bootstrap resamples. "
            "These numbers prove the pipeline executes and are NOT research results."
        )
    artifacts.write_manifest(manifest, out_dir / "run_manifest.json")
    return manifest


def summarise(manifest: dict[str, Any], run_dir: Path) -> str:
    """A human-readable table of the run."""
    lines = [
        f"Track K benchmark - run {manifest['track_k_run_id']}",
        f"protocol {manifest['protocol_version']}  primary metric "
        f"{manifest['evaluation']['primary_metric']}",
    ]
    if manifest.get("smoke"):
        lines.append("*** SMOKE RUN - not research results ***")
    lines.append("")
    header = f"{'family':<20}{'roc_auc':>10}{'pr_auc':>10}{'recall':>9}{'brier':>9}{'ece':>9}{'ms/row':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for family, record in manifest["models"].items():
        metrics_path = run_dir / record["artifacts"]["metrics"]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test_metrics"]
        lines.append(
            f"{family:<20}{metrics['roc_auc']:>10.5f}{metrics['pr_auc']:>10.5f}"
            f"{metrics['recall']:>9.4f}{metrics['brier_score']:>9.5f}"
            f"{metrics['ece']:>9.5f}"
            f"{record['resources']['single_row_median_ms']:>9.3f}"
        )
    lines.append("")
    lines.append(f"baseline for promotion: {manifest['promotion']['baseline']}")
    for challenger, decision in manifest["promotion"]["decisions"].items():
        delta = decision["primary_delta"]
        lines.append(
            f"  {challenger:<18} {decision['verdict']:<14} "
            f"delta {delta['point']:+.5f} [{delta['ci_lower']:+.5f}, {delta['ci_upper']:+.5f}]"
        )
    lines.append("")
    for item in manifest["comparisons"]:
        delta = item["delta"]
        lines.append(
            f"  {item['challenger']:<16} vs {item['baseline']:<20} "
            f"{delta['point']:+.5f} [{delta['ci_lower']:+.5f}, {delta['ci_upper']:+.5f}]  "
            f"{item['outcome']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny deterministic run that proves the pipeline, in seconds. Not research results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where artifacts are written (default: research_artifacts/track_k).",
    )
    args = parser.parse_args(argv)

    manifest = run(smoke=args.smoke, output_root=args.output_root)
    root = Path(args.output_root) if args.output_root else artifacts.RESEARCH_ROOT
    print(summarise(manifest, root / manifest["track_k_run_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
