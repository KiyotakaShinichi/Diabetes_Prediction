"""Re-derive a completed run's statistics from its saved predictions.

Training is expensive; scoring is not. Every run persists per-row test
probabilities exactly so that a metric can be recomputed later without touching
a GPU-less machine for another ninety minutes.

The immediate reason this exists: a full benchmark completed, and afterwards the
bootstrap's average-precision implementation was found to mishandle tied scores.
The models were correct, the predictions were correct, and only a secondary
metric's arithmetic was wrong. Retraining four families to fix arithmetic would
have burned an hour and a half of CPU to reproduce predictions that were already
on disk, byte for byte.

What this module will NOT do:

* it never retrains, so it cannot change a model, a hyperparameter, a threshold
  or a calibrator;
* it never rewrites the original manifest - the recomputation is written beside
  it, naming the run it came from and the reason it was made, so both the
  original numbers and the corrected ones remain inspectable;
* it fails closed if the saved predictions no longer match their recorded
  hashes, or if the frozen split has moved underneath them.

That last point is what makes this legitimate rather than convenient. A
recomputation is only trustworthy if the inputs are provably the same inputs.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml_core import training as core_training
from research.track_k import artifacts, comparison, evaluation, protocol, split

#: Written beside the original manifest, never over it.
RECOMPUTED_FILENAME = "recomputed_statistics.json"


class RecomputeError(RuntimeError):
    """The recorded run cannot be trusted to recompute from."""


def load_run(run_dir: Path) -> dict[str, Any]:
    """The manifest of a completed run, or a clear failure."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise RecomputeError(f"no run manifest at {manifest_path}")
    loaded: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return loaded


def load_predictions(run_dir: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-family test probabilities as they were written."""
    predictions = {}
    for family in manifest["models"]:
        path = Path(run_dir) / f"{family}_test_proba.npy"
        if not path.is_file():
            raise RecomputeError(f"{family}: no saved test predictions at {path}")
        predictions[family] = np.load(path)
    return predictions


def verify_inputs(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Everything that must still hold for a recomputation to mean anything.

    Checks the run's own artifacts against their recorded hashes, and the frozen
    split against the dataset as it exists now. A recomputation over different
    rows would be a new experiment wearing an old run's name.
    """
    problems = list(artifacts.verify_run_manifest(manifest, run_dir))

    recorded_split = manifest.get("split", {})
    current = split.fingerprint_split(split.build_split(split.load_dataset())).as_dict()
    for field in ("dataset_sha256", "combined_sha256", "seed"):
        if recorded_split.get(field) != current.get(field):
            problems.append(
                f"split {field} changed since the run: recorded "
                f"{recorded_split.get(field)!r}, now {current.get(field)!r}"
            )
    return problems


def recompute(
    run_dir: Path,
    *,
    reason: str,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Recompute metrics, intervals and comparisons under the current code.

    Returns the recomputed record. The caller decides whether to write it.
    """
    run_dir = Path(run_dir)
    manifest = load_run(run_dir)
    problems = verify_inputs(run_dir, manifest)
    if problems:
        raise RecomputeError("; ".join(problems))

    predictions = load_predictions(run_dir, manifest)
    splits = split.build_split(split.load_dataset())
    y_test = splits.y_test.to_numpy()

    intervals, replicates = comparison.bootstrap_metrics(
        y_test, predictions, resamples=resamples
    )
    comparisons = [
        item.as_dict()
        for item in comparison.compare_all(replicates, comparison.default_pairs())
    ]

    metrics: dict[str, Any] = {}
    changed: dict[str, Any] = {}
    for family, proba in predictions.items():
        threshold = manifest["models"][family]["threshold"]
        recomputed = evaluation.evaluate(y_test, proba, threshold=threshold)
        metrics[family] = recomputed
        original = json.loads(
            (run_dir / f"{family}_metrics.json").read_text(encoding="utf-8")
        )
        differences = {
            name: {"original": original["test_metrics"][name], "recomputed": value}
            for name, value in recomputed.items()
            if isinstance(value, float)
            and not _close(original["test_metrics"].get(name), value)
        }
        original_intervals = original.get("bootstrap", {})
        for name, interval in intervals[family].items():
            was = original_intervals.get(name, {})
            if not _close(was.get("point"), interval.point):
                differences[f"bootstrap.{name}.point"] = {
                    "original": was.get("point"),
                    "recomputed": interval.point,
                }
        if differences:
            changed[family] = differences

    return {
        "provenance_type": "track_k_recomputation",
        "recomputed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run_id": manifest["track_k_run_id"],
        "source_protocol_version": manifest["protocol_version"],
        "protocol_version": protocol.PROTOCOL_VERSION,
        "reason": reason,
        "retrained": False,
        "note": (
            "Recomputed from the run's saved per-row test predictions. No model, "
            "hyperparameter, threshold or calibrator was changed or re-fitted."
        ),
        "bootstrap": {
            "resamples": resamples,
            "alpha": protocol.BOOTSTRAP_ALPHA,
            "seed": protocol.BOOTSTRAP_SEED,
            "paired": True,
            "intervals": {
                family: {name: i.as_dict() for name, i in per_metric.items()}
                for family, per_metric in intervals.items()
            },
        },
        "test_metrics": metrics,
        "comparisons": comparisons,
        "differences_from_original": changed,
        "source": artifacts.source_fingerprint(),
        "environment": artifacts.research_environment(),
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return bool(left == right)


def write(record: dict[str, Any], run_dir: Path) -> Path:
    return core_training.write_json_atomic(record, Path(run_dir) / RECOMPUTED_FILENAME)


def summarise(record: dict[str, Any]) -> str:
    lines = [
        f"Recomputed run {record['source_run_id']} without retraining",
        f"reason: {record['reason']}",
        "",
    ]
    changed = record["differences_from_original"]
    if not changed:
        lines.append("No statistic changed: the original run reproduces exactly.")
        return "\n".join(lines)

    lines.append("Statistics that changed:")
    for family, differences in changed.items():
        for name, pair in differences.items():
            original, recomputed = pair["original"], pair["recomputed"]
            delta = (
                f"{recomputed - original:+.5f}"
                if isinstance(original, float) and isinstance(recomputed, float)
                else "n/a"
            )
            lines.append(
                f"  {family:<20} {name:<24} {original!s:>12} -> {recomputed!s:>12}  {delta}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A completed run directory.")
    parser.add_argument(
        "--reason",
        required=True,
        help="Why this recomputation was made. Recorded in the output.",
    )
    args = parser.parse_args(argv)

    record = recompute(args.run_dir, reason=args.reason)
    path = write(record, args.run_dir)
    print(summarise(record))
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
