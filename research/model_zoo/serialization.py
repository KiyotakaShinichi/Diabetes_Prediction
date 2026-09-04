"""Fit, save, load, predict again - and check the predictions still match.

Most benchmarks stop at "the file was written". That proves almost nothing: a
model can serialize to disk perfectly and still come back predicting something
different, because what round-trips is the estimator's parameters and not
always its surrounding state. A fitted scaler, a feature ordering, a torch
module's buffers - lose any of them and the reloaded model is a different model
that fails silently, on production traffic, weeks later.

So the lab reloads and re-scores. `round_trip` fits nothing; it takes an already
fitted model, writes it, reads it back into a fresh object and compares
predictions row by row on rows the model has already seen. The interesting
number is `max_abs_difference`, and the interesting result is when it is not
zero.

Tolerance is deliberately tight. These are deterministic models being reloaded
on the same machine in the same process; anything beyond floating-point noise
means state was lost, not that the format is approximate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.model_zoo.contracts import (
    CapabilityError,
    ProbabilityBehavior,
    SerializationRecord,
)
from research.model_zoo.registry import REGISTRY

#: Predictions must match to within this after a save/load cycle. Set at
#: float32 epsilon rather than something forgiving: a real state loss moves
#: predictions by orders of magnitude more than this, and a tolerance loose
#: enough to hide one would make the check decorative.
TOLERANCE: float = 1e-6


def _scores(model: Any, X: pd.DataFrame, behavior: ProbabilityBehavior) -> np.ndarray:
    """Whatever this model can be compared on: scores if it ranks, else labels.

    A hard-label model still deserves a round-trip check; it just has to be
    compared on the only output it has.
    """
    if behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.decision_scores(X), dtype=float)


def round_trip(
    model_id: str, model: Any, X: pd.DataFrame, *, directory: Path
) -> SerializationRecord:
    """Save a fitted model, load it back, and prove it still predicts the same.

    Returns a record rather than raising, because a serialization failure is a
    result the zoo reports about that model rather than an error that stops the
    benchmark. Every exception becomes ``round_trip_ok=False`` with the reason
    attached.
    """
    spec = REGISTRY.get(model_id)
    if not spec.capabilities.supports_serialization:
        return SerializationRecord(
            format="none",
            bytes_written=0,
            round_trip_ok=False,
            error="the model declares no serialization support",
        )

    path = Path(directory) / f"{model_id}{_suffix(spec.framework.value)}"
    try:
        before = _scores(model, X, spec.probability_behavior)
        record = model.serialize(path)
        restored = _restore(model, path)
        after = _scores(restored, X, spec.probability_behavior)
    except (OSError, ValueError, TypeError, RuntimeError, CapabilityError) as error:
        return SerializationRecord(
            format="unknown",
            bytes_written=path.stat().st_size if path.exists() else 0,
            round_trip_ok=False,
            error=f"{type(error).__name__}: {error}",
        )

    if before.shape != after.shape:
        return SerializationRecord(
            format=record.format,
            bytes_written=record.bytes_written,
            round_trip_ok=False,
            error=f"shape changed on reload: {before.shape} -> {after.shape}",
        )

    difference = float(np.max(np.abs(before - after))) if len(before) else 0.0
    return SerializationRecord(
        format=record.format,
        bytes_written=record.bytes_written,
        round_trip_ok=difference <= TOLERANCE,
        max_abs_difference=difference,
        error=(
            None
            if difference <= TOLERANCE
            else f"predictions moved by {difference:.3e} after reload"
        ),
    )


def _suffix(framework: str) -> str:
    return ".pt" if framework == "torch" else ".joblib"


def _restore(model: Any, path: Path) -> Any:
    """Rebuild a model from disk, by whichever route its adapter provides.

    The torch adapter reloads into a freshly constructed module rather than
    unpickling one, which is why it needs the original adapter to supply the
    architecture. That is a security property as much as a convenience: a state
    dict cannot execute code on load, and a pickled module can.
    """
    if hasattr(model, "load_state"):
        import copy

        clone = copy.copy(model)
        return clone.load_state(path)

    from research.model_zoo.adapters.sklearn_adapter import SklearnAdapter

    restored = SklearnAdapter(
        model.model_id,
        SklearnAdapter.deserialize(path),
        capabilities=model.capabilities,
        probability_behavior=model.probability_behavior,
    )
    return restored


def summarise(records: dict[str, SerializationRecord]) -> str:
    """A short table of what survived a round trip and what did not."""
    lines = [f"{'model':<24}{'format':<18}{'bytes':>10}{'delta':>12}  ok"]
    lines.append("-" * len(lines[0]))
    for model_id, record in records.items():
        delta = (
            f"{record.max_abs_difference:.2e}"
            if record.max_abs_difference is not None
            else "-"
        )
        lines.append(
            f"{model_id:<24}{record.format:<18}{record.bytes_written:>10,}{delta:>12}  "
            f"{'yes' if record.round_trip_ok else 'NO'}"
        )
    failures = [m for m, r in records.items() if not r.round_trip_ok]
    if failures:
        lines.append("")
        lines.append(f"{len(failures)} model(s) failed the round trip:")
        for model_id in failures:
            lines.append(f"  {model_id}: {records[model_id].error}")
    return "\n".join(lines)
