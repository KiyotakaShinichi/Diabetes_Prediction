"""What every model in the zoo must be able to say about itself.

The zoo holds thirty-odd algorithms from six families, written against four
different libraries, with genuinely different capabilities. A nearest-centroid
classifier has no probabilities. A support vector machine has a decision
function but no calibrated ones. A neural network has a parameter count; a
random forest does not. A gradient-boosted tree needs no scaling; a k-nearest
neighbour classifier is meaningless without it.

The temptation is a benchmark runner full of ``if model_id == ...``. That
collapses under its own weight by the tenth model and lies by the twentieth,
because the special cases stop matching the models.

So capability is **declared** rather than inferred, and a test asserts every
declaration against the built model. If a spec claims probabilities, the model
must produce them; if it claims none, the runner must not ask. A model that
misdescribes itself fails its own test rather than silently producing a column
of fabricated numbers in the results table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class Family(StrEnum):
    """Algorithmic family. Used to group diversity analysis, not to rank."""

    LINEAR = "linear"
    PROBABILISTIC = "probabilistic"
    KERNEL = "kernel"
    DISTANCE = "distance"
    TREE = "tree"
    BOOSTING = "boosting"
    DEEP = "deep"


class Framework(StrEnum):
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    TORCH = "torch"


class Preprocessing(StrEnum):
    """How a model wants its features, declared per model.

    Every one of these is fitted on training rows only. The distinction is not
    cosmetic: k-nearest neighbours on unscaled features is a distance metric
    dominated by whichever column has the widest range, and a tree does not
    care either way.
    """

    #: Zero mean, unit variance. Required by distance and gradient methods.
    STANDARDIZED = "standardized"
    #: Median and IQR. For models sensitive to the outliers in BMI/PhysHlth.
    ROBUST_SCALED = "robust_scaled"
    #: Features as they come. Trees and boosting are scale-invariant.
    RAW_NUMERIC = "raw_numeric"
    #: The model does its own preparation - the Track K deep models own their
    #: standardiser and ordinal vocabulary.
    MODEL_NATIVE = "model_native"


class ProbabilityBehavior(StrEnum):
    """How honest a model's ``predict_proba`` actually is.

    This is the field the zoo most needs, because three very different things
    get called "probability" and averaging them into one table without saying
    so would be the single most misleading thing this module could do.
    """

    #: Fitted to maximise likelihood; the number means something as a probability.
    NATIVE_PROBABILISTIC = "native_probabilistic"
    #: A real distribution, but from a model not fitted to calibrate it - a
    #: forest's vote share, a boosted margin through a sigmoid.
    NATIVE_UNCALIBRATED = "native_uncalibrated"
    #: Only a decision function exists; any probability comes from a calibrator
    #: fitted afterwards on held-out validation rows.
    REQUIRES_EXTERNAL_CALIBRATION = "requires_external_calibration"
    #: No meaningful score at all. Hard labels only, and the threshold-free
    #: metrics are genuinely undefined rather than zero.
    HARD_LABELS_ONLY = "hard_labels_only"


class ResourceClass(StrEnum):
    """Expected cost at the zoo's 1,000-row budget, set from measurement."""

    LIGHT = "light"        # under a second
    MODERATE = "moderate"  # seconds
    HEAVY = "heavy"        # tens of seconds


class ResearchStatus(StrEnum):
    ACTIVE = "active"
    OPTIONAL = "optional"
    UNSUPPORTED = "unsupported"
    SKIPPED_RESOURCE_LIMIT = "skipped_resource_limit"


class RunOutcome(StrEnum):
    """What happened when the benchmark actually tried to run a model.

    FAILED and SKIPPED are results, not omissions. A model that cannot be
    installed, cannot converge or cannot serialize stays in the table with the
    reason attached; deleting the row would turn an honest table into a
    flattering one.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RESOURCE_LIMIT = "resource_limit"
    UNSUITABLE = "unsuitable"


@runtime_checkable
class TabularModelAdapter(Protocol):
    """The uniform surface the benchmark drives.

    Deliberately small. Anything a particular family needs beyond this belongs
    in that family's adapter, not in the protocol every model must satisfy.
    """

    model_id: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TabularModelAdapter: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability, one value per row.

        Raises ``CapabilityError`` when the spec declares HARD_LABELS_ONLY.
        """
        ...

    def decision_scores(self, X: pd.DataFrame) -> np.ndarray:
        """A monotone ranking score, whether or not it is a probability.

        This is what makes threshold-free comparison possible across the whole
        zoo: an SVM's signed distance ranks as well as a logistic regression's
        probability, and ROC-AUC only needs the ranking.
        """
        ...

    def serialize(self, path: Any) -> Any: ...

    def metadata(self) -> dict[str, Any]: ...


class CapabilityError(RuntimeError):
    """A model was asked for something it never claimed to provide."""


class ModelZooError(RuntimeError):
    """A failure in the zoo harness itself, distinct from a model failing."""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a model can do, declared once and asserted by tests."""

    supports_predict_proba: bool
    supports_calibration: bool
    supports_feature_importance: bool
    supports_serialization: bool
    requires_scaling: bool
    deterministic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "supports_predict_proba": self.supports_predict_proba,
            "supports_calibration": self.supports_calibration,
            "supports_feature_importance": self.supports_feature_importance,
            "supports_serialization": self.supports_serialization,
            "requires_scaling": self.requires_scaling,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """What one fit actually cost, recorded rather than estimated."""

    fit_seconds: float
    training_rows: int
    epochs_run: int | None = None
    best_epoch: int | None = None
    parameter_count: int | None = None
    peak_memory_mib: float | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_seconds": round(self.fit_seconds, 4),
            "training_rows": self.training_rows,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "parameter_count": self.parameter_count,
            "peak_memory_mib": (
                round(self.peak_memory_mib, 3) if self.peak_memory_mib is not None else None
            ),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class SerializationRecord:
    """Proof that a fitted model survives a round trip.

    The interesting field is ``max_abs_difference``. Saving and loading a model
    is only useful if the loaded one predicts what the original did, so the lab
    reloads and re-scores rather than merely checking the file exists.
    """

    format: str
    bytes_written: int
    round_trip_ok: bool
    max_abs_difference: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "bytes_written": self.bytes_written,
            "round_trip_ok": self.round_trip_ok,
            "max_abs_difference": self.max_abs_difference,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ModelResult:
    """One model's complete row in the results table, success or failure."""

    model_id: str
    family: Family
    outcome: RunOutcome
    metrics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    serialization: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family.value,
            "outcome": self.outcome.value,
            "metrics": self.metrics,
            "training": self.training,
            "serialization": self.serialization,
            "error": self.error,
        }
