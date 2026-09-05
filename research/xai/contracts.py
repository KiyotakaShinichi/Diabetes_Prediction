"""What an explanation is, in this repository, and what it is not.

Track L established that twenty-nine algorithms make substantially the same
predictions. Track M asks the harder question: do they do it for the same
reasons? That only means something if "the reason" is a recorded, comparable,
reproducible object rather than a plot someone liked.

Three decisions shape everything here.

**Attributions are not comparable in their raw units.** A logistic coefficient,
a permutation importance in ROC-AUC points, an integrated gradient in logit
space and an impurity decrease are four different quantities. Putting them in
one table without normalising would produce a number that looks like a
comparison and is not one. Every record therefore carries raw values *and* a
normalised share *and* a rank, and the cross-model analysis reads only the last
two.

**Evidence is JSON, not pickles.** A pickled explainer is a snapshot of a
library version that cannot be diffed, cannot be read in five years, and can
execute code on load. Records serialise to plain data with the hashes needed to
tie them back to a model, a subset and a commit.

**Attribution is not causation.** Nothing in this package licenses a claim that
a feature *causes* an outcome or that changing it would change a patient's risk.
The vocabulary is association and model dependence, and
`tests/test_xai_language.py` enforces that in the generated reports.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

#: Schema version for the record format. Bumped when a field's meaning changes,
#: so a stored run can never be silently reinterpreted under new rules.
RECORD_SCHEMA_VERSION: str = "1.0.0"


class Scope(StrEnum):
    """Whether an explanation describes one row or the whole model."""

    LOCAL = "local"
    GLOBAL = "global"


class Determinism(StrEnum):
    """Whether repeating a method with the same inputs gives the same answer.

    Load-bearing for the seed study: running a seed sweep over a deterministic
    method would manufacture a variance measurement out of nothing, so the
    stability code reads this rather than guessing.
    """

    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class BaselineStrategy(StrEnum):
    """The reference point a method perturbs away from.

    Integrated gradients, occlusion and ablation all answer "compared to what?".
    Leaving that implicit is how two runs become incomparable, so the strategy
    is declared per method and recorded in every record it produces.

    Every one of these is computed from TRAINING rows only. A baseline built
    from test statistics would leak the evaluation distribution into the
    explanation of a model being judged on it.
    """

    TRAINING_MEDIAN = "training_median"
    TRAINING_MEAN = "training_mean"
    #: The origin of the model's own encoded space, which is NOT uniformly the
    #: training mean. Track K's encoder standardises the continuous and ordinal
    #: features but leaves binary indicators on their raw 0/1 scale, so zero
    #: means "the training mean" for five features and "No" for the other five.
    #: Named separately from TRAINING_MEAN for exactly that reason.
    ZERO_STANDARDIZED = "zero_standardized"
    #: The method needs no reference at all - a coefficient, an impurity count.
    NOT_APPLICABLE = "not_applicable"


class RunStatus(StrEnum):
    """What happened when a method was asked to explain a model.

    Failures stay in the tables. A results file containing only the
    combinations that worked is a flattering file, and the whole point of the
    capability contract is that a method being unavailable is itself evidence
    about the model.
    """

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    RESOURCE_LIMIT = "resource_limit"
    NUMERICAL_FAILURE = "numerical_failure"
    INVALID_CAPABILITY = "invalid_capability"


class AgreementLabel(StrEnum):
    """Descriptive bands for rank agreement. Not a probability.

    Thresholds are frozen in `agreement.py` before any aggregate result is
    inspected. They describe a correlation, and calling one "confidence" would
    imply a calibration nobody measured.
    """

    HIGH = "high_agreement"
    MODERATE = "moderate_agreement"
    LOW = "low_agreement"


class DisagreementLabel(StrEnum):
    """The inverse banding, used by the composite disagreement summary."""

    LOW = "low_disagreement"
    MODERATE = "moderate_disagreement"
    HIGH = "high_disagreement"


class XaiError(RuntimeError):
    """A failure inside the XAI harness, distinct from a method failing."""


class CapabilityError(XaiError):
    """A model was asked for an explanation it never claimed to support."""


@dataclass(frozen=True, slots=True)
class ExplanationRecord:
    """One explanation, with everything needed to reproduce and audit it.

    Frozen because an explanation is evidence: mutating one after the fact
    would leave the hashes describing something that no longer exists.
    """

    explanation_id: str
    model_id: str
    model_family: str
    method: str
    method_version: str
    scope: Scope
    feature_names: tuple[str, ...]
    raw_attributions: tuple[float, ...]
    normalized_attributions: tuple[float, ...]
    ranking: tuple[str, ...]
    baseline_reference: str
    seed: int | None = None
    sample_id: int | None = None
    prediction: float | None = None
    prediction_probability: float | None = None
    target_class: int = 1
    resource_status: RunStatus = RunStatus.SUCCESS
    source_sha: str = ""
    model_config_hash: str = ""
    training_subset_hash: str = ""
    data_fingerprint: str = ""
    runtime_seconds: float | None = None
    peak_memory_mib: float | None = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema_version: str = RECORD_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "method": self.method,
            "method_version": self.method_version,
            "scope": self.scope.value,
            "sample_id": self.sample_id,
            "feature_names": list(self.feature_names),
            "raw_attributions": [float(v) for v in self.raw_attributions],
            "normalized_attributions": [float(v) for v in self.normalized_attributions],
            "ranking": list(self.ranking),
            "baseline_reference": self.baseline_reference,
            "seed": self.seed,
            "prediction": self.prediction,
            "prediction_probability": self.prediction_probability,
            "target_class": self.target_class,
            "resource_status": self.resource_status.value,
            "source_sha": self.source_sha,
            "model_config_hash": self.model_config_hash,
            "training_subset_hash": self.training_subset_hash,
            "data_fingerprint": self.data_fingerprint,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mib": self.peak_memory_mib,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @property
    def top_feature(self) -> str:
        return self.ranking[0]

    def rank_of(self, feature: str) -> int:
        """1-based position of a feature in this explanation's ranking."""
        return self.ranking.index(feature) + 1

    def attribution_for(self, feature: str) -> float:
        return float(self.normalized_attributions[self.feature_names.index(feature)])


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    """The result of asking one method to explain one model.

    Carries either a record or a reason there is none. The runner writes one of
    these per (model, method) pair whatever happens, so the results table has a
    row for every combination that was attempted.
    """

    model_id: str
    method: str
    status: RunStatus
    record: ExplanationRecord | None = None
    error: str | None = None
    runtime_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "method": self.method,
            "status": self.status.value,
            "record": self.record.as_dict() if self.record else None,
            "error": self.error,
            "runtime_seconds": self.runtime_seconds,
        }


def normalize_attributions(values: np.ndarray) -> np.ndarray:
    """Absolute attribution as a share of total absolute attribution.

    This is what makes a coefficient comparable to an impurity decrease: both
    become "what fraction of this model's total attributed magnitude sits on
    this feature". Sign is deliberately discarded here - direction is preserved
    in ``raw_attributions`` and read by the local analyses, but a global
    importance comparison across families cannot use it, because a permutation
    importance has no sign to compare against a coefficient's.

    An all-zero attribution vector - a model that ignored every feature, or a
    degenerate fit - returns a uniform share rather than dividing by zero, and
    the caller can detect that case because every value is identical.
    """
    magnitudes = np.abs(np.asarray(values, dtype=float))
    total = magnitudes.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(magnitudes), 1.0 / max(len(magnitudes), 1))
    shares: np.ndarray = magnitudes / total
    return shares


def rank_features(
    feature_names: tuple[str, ...], values: np.ndarray
) -> tuple[str, ...]:
    """Features ordered most- to least-attributed, ties broken by name.

    Deterministic tie-breaking matters more than it looks: a model that assigns
    identical importance to several features - an L1 fit that zeroed them, a
    tree that never split on them - would otherwise produce a different ranking
    on every run, and the stability study would measure the sort, not the model.
    """
    magnitudes = np.abs(np.asarray(values, dtype=float))
    order = sorted(
        range(len(feature_names)),
        key=lambda i: (-magnitudes[i], feature_names[i]),
    )
    return tuple(feature_names[i] for i in order)


def explanation_id(
    model_id: str, method: str, *, sample_id: int | None, seed: int | None
) -> str:
    """A stable identifier for one explanation.

    Content-addressed over the coordinates that define it, so re-running the
    same explanation produces the same id and a results directory can be
    deduplicated without comparing floats.
    """
    payload = f"{model_id}|{method}|{sample_id}|{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def hash_payload(payload: Any) -> str:
    """Stable SHA-256 over any JSON-serialisable structure."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
