"""Research artifact provenance, kept separate from production provenance.

Two rules govern this module.

The first is separation. Track K writes only under ``research_artifacts/``.
Nothing here reads, writes or references ``model_artifacts/`` or the production
attestation, so a research checkpoint can never be confused with - or overwrite -
a deployed model. A test parses every Track K module and asserts that no string
literal the interpreter evaluates names the production directory: prose like
this sentence is allowed to explain the rule, a path that could open it is not.

The second is reuse. The provenance philosophy is the repository's existing one,
not a second incompatible invention: dataset and feature fingerprints, git
state, environment capture, artifact inventory and atomic writes all come from
``ml_core.provenance``. What is added is the part production has no concept of -
the frozen split, the search budget, the calibration decision and the training
curve.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_core import provenance
from ml_core import training as core_training
from research.track_k import protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Everything Track K writes lives here. Gitignored; runs are reproducible from
#: the committed protocol rather than from committed weights.
RESEARCH_ROOT = PROJECT_ROOT / "research_artifacts" / "track_k"

#: Identifies a research manifest, distinct from production's TRAINING_RUN.
RESEARCH_RUN = "track_k_research_run"

#: Source files whose contents change what a Track K run does.
SOURCE_MODULES = (
    "protocol.py",
    "split.py",
    "evaluation.py",
    "comparison.py",
    "calibration.py",
    "baselines.py",
    "challengers.py",
    "benchmark.py",
    "artifacts.py",
    "deep/models.py",
    "deep/training.py",
    "deep/preprocessing.py",
)


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Everything needed to identify and reproduce one trained model."""

    family: str
    seed: int
    config: dict[str, Any]
    search: dict[str, Any]
    calibration: dict[str, Any]
    threshold: float
    parameter_count: int | None
    training: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "config": self.config,
            "search": self.search,
            "calibration": self.calibration,
            "threshold": self.threshold,
            "parameter_count": self.parameter_count,
            "training": self.training,
            "artifacts": self.artifact_paths,
            "artifact_sha256": self.artifact_hashes,
            "resources": self.resources,
        }


def research_environment() -> dict[str, Any]:
    """Interpreter and library versions, extended with the DL stack.

    ml_core.provenance captures the production environment; Track K adds torch,
    which production does not depend on and therefore does not record.
    """
    captured = provenance.fingerprint_environment(
        PROJECT_ROOT / "requirements.lock" if (PROJECT_ROOT / "requirements.lock").is_file() else None
    )
    try:
        import torch

        captured["torch_version"] = str(torch.__version__)
        captured["torch_cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a declared dev dependency
        captured["torch_version"] = None
        captured["torch_cuda_available"] = False
    captured["python_implementation"] = platform.python_implementation()
    captured["platform"] = platform.platform()
    captured["executable_version"] = sys.version.split()[0]
    return captured


def source_fingerprint() -> dict[str, Any]:
    """Hash the Track K source that determines a run's behaviour."""
    paths = [Path(__file__).parent / name for name in SOURCE_MODULES]
    return provenance.fingerprint_source(
        [path for path in paths if path.is_file()], PROJECT_ROOT
    )


def build_run_manifest(
    *,
    run_id: str,
    split_manifest: dict[str, Any],
    models: list[ModelRecord],
    comparisons: list[dict[str, Any]],
    promotion: dict[str, Any],
    bootstrap: dict[str, Any],
    started_at: str,
    root: Path = RESEARCH_ROOT,
) -> dict[str, Any]:
    """The manifest describing one complete benchmark run.

    Written LAST, once every artifact it inventories exists on disk, so a
    manifest can never describe files that were not produced.
    """
    return {
        "provenance_type": RESEARCH_RUN,
        "schema_version": provenance.SCHEMA_VERSION,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "track_k_run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": split_manifest.get("dataset", {}),
        "features": split_manifest.get("features", {}),
        "split": split_manifest.get("split", {}),
        "split_sizes": split_manifest.get("sizes", {}),
        "class_balance": split_manifest.get("class_balance", {}),
        "evaluation": {
            "primary_metric": protocol.PRIMARY_METRIC,
            "secondary_metrics": list(protocol.SECONDARY_METRICS),
            "calibration_metrics": list(protocol.CALIBRATION_METRICS),
            "ece_bins": protocol.ECE_BINS,
        },
        "bootstrap": bootstrap,
        "promotion_policy": {
            "baseline": protocol.PROMOTION_POLICY.baseline,
            "min_primary_delta": protocol.PROMOTION_POLICY.min_primary_delta,
            "max_ece_regression": protocol.PROMOTION_POLICY.max_ece_regression,
            "max_recall_regression": protocol.PROMOTION_POLICY.max_recall_regression,
            "max_latency_multiple": protocol.PROMOTION_POLICY.max_latency_multiple,
        },
        "models": {record.family: record.as_dict() for record in models},
        "comparisons": comparisons,
        "promotion": promotion,
        "environment": research_environment(),
        "git": provenance.git_provenance(PROJECT_ROOT),
        "source": source_fingerprint(),
        "artifact_root": describe_root(root),
        "production_artifacts_touched": False,
        "limitations": [
            "The study dataset is close to 50/50 positive, so every probability "
            "here is conditional on that base rate and must not be read as a "
            "population disease probability.",
            "Ten served features only; eleven further columns in the file are "
            "excluded to match the production contract.",
            "One split, one test set: intervals describe sampling variability "
            "within this test partition, not generalisation to another population.",
        ],
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Atomic write, so a partial manifest cannot look complete."""
    return core_training.write_json_atomic(manifest, path)


def verify_run_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    """Problems with a recorded run. Empty means the manifest still holds.

    ``root`` is the run directory the manifest describes, since artifact names
    are recorded relative to it.
    """
    problems: list[str] = []

    if manifest.get("provenance_type") != RESEARCH_RUN:
        problems.append(
            f"provenance_type is {manifest.get('provenance_type')!r}, expected {RESEARCH_RUN!r}"
        )
    if manifest.get("protocol_version") != protocol.PROTOCOL_VERSION:
        problems.append(
            f"protocol_version {manifest.get('protocol_version')!r} != "
            f"current {protocol.PROTOCOL_VERSION!r}"
        )
    if manifest.get("production_artifacts_touched"):
        problems.append("manifest claims production artifacts were touched")

    for family, record in manifest.get("models", {}).items():
        for role, relative in record.get("artifacts", {}).items():
            path = Path(root) / relative
            if not path.is_file():
                problems.append(f"{family}.{role}: missing artifact {relative}")
                continue
            expected = record.get("artifact_sha256", {}).get(role)
            if expected and provenance.sha256_file(path) != expected:
                problems.append(f"{family}.{role}: {relative} no longer matches its hash")

    return problems


def describe_root(root: Path) -> str:
    """The run directory, relative to the project when it lies inside it.

    A run written outside the repository - a scratch directory, a CI temp path -
    is recorded as its absolute location instead of failing. The manifest should
    describe where a run actually happened.
    """
    try:
        return provenance.relative_path(root, PROJECT_ROOT)
    except ValueError:
        return str(Path(root).resolve()).replace("\\", "/")


def inventory(paths: dict[str, Path], root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Artifact names and hashes, recorded relative to the RUN directory.

    Relative to the run rather than to the project root, so a run directory can
    be moved, archived or produced outside the repository and still verify. The
    manifest records where the root was; these are the names inside it.
    """
    root = Path(root)
    relative = {}
    for role, path in paths.items():
        resolved = Path(path).resolve()
        try:
            relative[role] = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative[role] = resolved.name
    hashes = {
        role: provenance.sha256_file(path)
        for role, path in paths.items()
        if Path(path).is_file()
    }
    return relative, hashes
