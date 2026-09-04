"""How each family's quality changes as the training budget grows.

The CPU constraint that shrank the main benchmark also created the opportunity
for this study. If every family must train on a small subset anyway, the
interesting question is not only "who wins at 5,000 rows" but "who gets there
fastest" - which family extracts the most from the least data. That is a real
research question about tabular architectures, and it is cheap to answer once
deterministic nested subsets exist.

Three design decisions make the curve readable:

**Frozen configurations, no search.** Every family trains the same architecture
at every subset size. Re-searching per size would confound two effects - more
data, and a differently-shaped model - and would cost hours. The configurations
below are conventional mid-sized choices, not the winners of any search, so no
size gets an advantage from search luck.

**Nested subsets.** 500 rows are contained in the 1,000, and so on up. A change
between two points is caused by having more data, not different data.

**The full test partition, every time.** Training is expensive; scoring is not.
A model fitted on 500 rows is still evaluated on all 13,376 test rows, so the
uncertainty in each point comes from the model rather than from a thin
evaluation set.

No calibration is fitted in this arm. Calibrators are themselves data-hungry,
and fitting one per size would mix "the model learned more" with "the
calibrator had more to work with". The curve is therefore reported on raw
probabilities, and the threshold-free metrics are the ones to read.
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
from research.track_k import (
    artifacts,
    baselines,
    calibration,
    challengers,
    evaluation,
    protocol,
    split,
    subsets,
)

#: Epoch ceiling for the deep families in this study. Early stopping usually
#: ends a run sooner. Kept low deliberately: the reference run showed both
#: networks reaching within 0.002 of their final validation ROC-AUC in a single
#: epoch, so a long budget would buy noise rather than signal.
MAX_EPOCHS: int = 15

#: One configuration per family, held fixed across every subset size.
#:
#: These are conventional middle-of-the-road choices rather than tuned winners.
#: That is the point: the study measures how much each ARCHITECTURE extracts
#: from a given amount of data, and a per-size search would answer a different
#: question at many times the cost.
REPRESENTATIVE_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "solver": "liblinear"},
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "mlp": {"hidden_dims": [128, 64], "dropout": 0.1, "batch_size": 256},
    "ft_transformer": {
        "d_token": 16,
        "n_blocks": 2,
        "n_heads": 4,
        "attention_dropout": 0.1,
        "ffn_dropout": 0.1,
        "batch_size": 256,
    },
    "tabular_resnet": {
        "d_hidden": 64,
        "n_blocks": 3,
        "d_expansion": 2.0,
        "dropout": 0.1,
        "batch_size": 256,
    },
}


@dataclass(frozen=True, slots=True)
class Point:
    """One family at one training-set size."""

    family: str
    training_rows: int
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    recall: float
    threshold: float
    train_seconds: float
    parameter_count: int | None
    epochs_run: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "training_rows": self.training_rows,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "recall": self.recall,
            "threshold": self.threshold,
            "train_seconds": self.train_seconds,
            "parameter_count": self.parameter_count,
            "epochs_run": self.epochs_run,
        }


def _fit_and_score(
    family: str, narrowed: core_training.TrainingSplits, *, max_epochs: int
) -> Point:
    """Train one family on one subset and score it on the full test partition."""
    seed = protocol.model_seed(family)
    params = REPRESENTATIVE_CONFIGS[family]
    started = time.perf_counter()

    if protocol.is_deep(family):
        data = challengers.prepare(narrowed)
        model, result = challengers.train_challenger(
            family, params, data, seed=seed, max_epochs=max_epochs
        )
        elapsed = time.perf_counter() - started
        val_proba = challengers.predict_proba(model, data, narrowed.X_val)
        test_proba = challengers.predict_proba(model, data, narrowed.X_test)
        parameters: int | None = challengers.parameter_count(model)
        epochs: int | None = result.epochs_run
    else:
        model = baselines.fit_final(family, params, narrowed, seed)
        elapsed = time.perf_counter() - started
        val_proba = baselines.predict_proba(model, narrowed.X_val)
        test_proba = baselines.predict_proba(model, narrowed.X_test)
        parameters = None
        epochs = None

    threshold = calibration.select_threshold(narrowed.y_val.to_numpy(), val_proba)
    metrics = evaluation.evaluate(
        narrowed.y_test.to_numpy(), test_proba, threshold=threshold
    )
    return Point(
        family=family,
        training_rows=len(narrowed.X_train),
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        brier_score=metrics["brier_score"],
        log_loss=metrics["log_loss"],
        recall=metrics["recall"],
        threshold=threshold,
        train_seconds=round(elapsed, 3),
        parameter_count=parameters,
        epochs_run=epochs,
    )


def run(
    *,
    sizes: tuple[int, ...] = protocol.SAMPLE_EFFICIENCY_SIZES,
    families: tuple[str, ...] = protocol.MODEL_FAMILIES,
    max_epochs: int = MAX_EPOCHS,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Train every family at every subset size and record the curve."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    run_id = f"sample-efficiency-{uuid.uuid4().hex[:12]}"
    root = Path(output_root) if output_root is not None else artifacts.RESEARCH_ROOT
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = split.build_split(split.load_dataset())
    split_manifest = split.build_split_manifest(splits)
    drift = split.verify_split(splits, split_manifest)
    if drift:
        raise split.SplitIntegrityError("; ".join(drift))

    subset_manifest = subsets.build_subset_manifest(splits, sizes=sizes)
    ladder = subsets.load_verified_subsets(splits, subset_manifest)

    points: list[Point] = []
    for size in sorted(ladder):
        narrowed = subsets.take(splits, ladder[size])
        for family in families:
            points.append(_fit_and_score(family, narrowed, max_epochs=max_epochs))

    manifest = {
        "provenance_type": "track_k_sample_efficiency",
        "schema_version": artifacts.provenance.SCHEMA_VERSION,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "track_k_run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": split_manifest.get("dataset", {}),
        "split": split_manifest.get("split", {}),
        "split_sizes": split_manifest.get("sizes", {}),
        "training_subset": subset_manifest,
        "evaluated_on": "test",
        "evaluation_rows": len(splits.X_test),
        "max_epochs": max_epochs,
        "calibration": "none - see the module docstring",
        "configurations": REPRESENTATIVE_CONFIGS,
        "points": [point.as_dict() for point in points],
        "environment": artifacts.research_environment(),
        "git": artifacts.provenance.git_provenance(artifacts.PROJECT_ROOT),
        "source": artifacts.source_fingerprint(),
        "production_artifacts_touched": False,
    }
    core_training.write_json_atomic(manifest, out_dir / "sample_efficiency.json")
    return manifest


def summarise(manifest: dict[str, Any]) -> str:
    """A table per metric, families down, training sizes across."""
    points = manifest["points"]
    sizes = sorted({point["training_rows"] for point in points})
    families = list(dict.fromkeys(point["family"] for point in points))
    indexed = {(p["family"], p["training_rows"]): p for p in points}

    lines = [
        f"Track K sample efficiency - run {manifest['track_k_run_id']}",
        f"frozen configurations, no search, evaluated on {manifest['evaluation_rows']:,} test rows",
        "",
    ]
    for metric, fmt in (("roc_auc", "{:.5f}"), ("pr_auc", "{:.5f}"), ("train_seconds", "{:.1f}")):
        lines.append(metric)
        header = f"{'family':<20}" + "".join(f"{size:>12,}" for size in sizes)
        lines.append(header)
        lines.append("-" * len(header))
        for family in families:
            row = f"{family:<20}"
            for size in sizes:
                point = indexed.get((family, size))
                row += f"{fmt.format(point[metric]):>12}" if point else f"{'-':>12}"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def gain_per_doubling(manifest: dict[str, Any], metric: str = "roc_auc") -> dict[str, float]:
    """Mean change in ``metric`` per doubling of training rows, per family.

    A single number for "how much does this family benefit from more data".
    Computed across the observed range only; it is a description of this curve,
    not an extrapolation beyond it.
    """
    points = manifest["points"]
    families = dict.fromkeys(point["family"] for point in points)
    gains: dict[str, float] = {}
    for family in families:
        series = sorted(
            (p for p in points if p["family"] == family), key=lambda p: p["training_rows"]
        )
        if len(series) < 2:
            continue
        doublings = np.log2(series[-1]["training_rows"] / series[0]["training_rows"])
        gains[family] = float((series[-1][metric] - series[0][metric]) / doublings)
    return gains


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(protocol.SAMPLE_EFFICIENCY_SIZES),
        help="Training-set sizes to evaluate. Must be drawable from train.",
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = run(
        sizes=tuple(args.sizes),
        max_epochs=args.max_epochs,
        output_root=args.output_root,
    )
    print(summarise(manifest))
    print("mean ROC-AUC gain per doubling of training rows:")
    for family, gain in sorted(
        gain_per_doubling(manifest).items(), key=lambda item: -item[1]
    ):
        print(f"  {family:<20} {gain:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
