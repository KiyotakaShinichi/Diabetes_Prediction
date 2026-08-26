"""Build a truthful integrity attestation for the committed artifacts.

    python tools/build_artifact_attestation.py

The artifacts in ``model_artifacts/`` predate this provenance system. Their
actual training lineage - which commit, which dataset bytes, which
configuration - was never recorded and cannot be recovered. Reconstructing it
from file timestamps or from what the current code happens to do would be
fabrication.

So this writes ``provenance/legacy_artifact_attestation.json``, which states
only what is observable right now:

* the SHA256 and size of every committed artifact;
* library versions recovered from *inside* each artifact by byte scan, labelled
  as observed rather than assumed;
* which artifacts the current serving code actually loads, and which are dead
  leftovers.

Every historical field is explicitly ``null``. Artifact integrity is not proof
of training provenance.

Regenerate this file only when the artifacts themselves legitimately change,
and expect CI to fail until you do.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_core import provenance  # noqa: E402 - after sys.path setup

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
DEFAULT_OUTPUT = PROJECT_ROOT / "provenance" / "legacy_artifact_attestation.json"

#: Artifacts the serving code loads today, mapped to their logical role.
#: Verified against app.py / streamlit_app.py / admin_app.py by the test suite.
SERVING_ROLES: dict[str, str] = {
    "model_bundle.pkl": "model_bundle_variant_a",
    "boosted_model_bundle.pkl": "model_bundle_variant_b",
    "shap_explainer.pkl": "shap_explainer_variant_a",
    "boosted_shap_explainer.pkl": "shap_explainer_variant_b",
    "drift_baseline.pkl": "drift_baseline_variant_a",
    "boosted_drift_baseline.pkl": "drift_baseline_variant_b",
    "metrics.json": "metrics_variant_a",
    "boosted_metrics.json": "metrics_variant_b",
}

#: Present in model_artifacts/ but loaded by no serving module. Recorded so they
#: stay visible and cannot quietly become serving dependencies. diabetes_model
#: and scaler carry scikit-learn 1.7.1, older than the served bundles.
NON_SERVING_ROLES: dict[str, str] = {
    "diabetes_model.pkl": "unused_legacy_estimator",
    "scaler.pkl": "unused_legacy_scaler",
    "threshold.txt": "unused_legacy_threshold",
    "test_predictions.csv": "training_output_not_served",
}


def build_attestation() -> dict:
    """Inventory the committed artifacts as they are on disk right now."""
    artifacts = []
    for name, role in {**SERVING_ROLES, **NON_SERVING_ROLES}.items():
        path = ARTIFACTS_DIR / name
        if not path.is_file():
            continue
        artifacts.append(
            provenance.inventory_artifact(
                role, path, PROJECT_ROOT,
                required_for_serving=name in SERVING_ROLES,
            )
        )
    artifacts.sort(key=lambda entry: entry["path"])

    dataset_path = PROJECT_ROOT / "cleaned_data.csv"
    current_dataset = None
    if dataset_path.is_file():
        current_dataset = {
            "path": provenance.relative_path(dataset_path, PROJECT_ROOT),
            "sha256": provenance.sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "note": (
                "Hash of the dataset committed TODAY. It is NOT established that "
                "these bytes are what produced the artifacts above."
            ),
        }

    return {
        "schema_version": provenance.SCHEMA_VERSION,
        "provenance_type": provenance.LEGACY_ATTESTATION,
        "attestation": {
            "statement": (
                "Integrity inventory of artifacts that predate this provenance "
                "system. Artifact integrity is NOT proof of training provenance."
            ),
            "observed_by": "tools/build_artifact_attestation.py",
            "generated_from_commit": provenance.git_provenance(PROJECT_ROOT),
        },
        # Everything below was never recorded and is not recoverable. Leaving
        # these as explicit nulls is the honest answer; guessing is not.
        "unknown_history": {
            "producer_git_sha": None,
            "training_run_id": None,
            "training_dataset_sha256": None,
            "training_started_at": None,
            "training_configuration": None,
            "training_environment": None,
            "reason": (
                "These artifacts were committed before provenance manifests "
                "existed. No run record was kept, and reconstructing one from "
                "file timestamps or from current source would be fabrication."
            ),
        },
        "current_dataset": current_dataset,
        "environment_note": (
            "Versions under artifacts[].embedded_versions are read from bytes "
            "inside each artifact and are therefore OBSERVED, not assumed. They "
            "describe what wrote the artifact, not the current interpreter."
        ),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true",
        help="Do not write; exit non-zero if the file on disk is out of date.",
    )
    args = parser.parse_args(argv)

    attestation = build_attestation()

    if args.check:
        if not args.output.is_file():
            print(f"missing attestation: {args.output}", file=sys.stderr)
            return 1
        current = provenance.load_manifest(args.output)
        # Ignore the generating commit, which legitimately moves between runs.
        expected = dict(attestation)
        expected["attestation"] = {
            **attestation["attestation"],
            "generated_from_commit": current.get("attestation", {}).get("generated_from_commit"),
        }
        if provenance.canonical_json(current) != provenance.canonical_json(expected):
            print(f"attestation is out of date: {args.output}", file=sys.stderr)
            print("regenerate with: python tools/build_artifact_attestation.py", file=sys.stderr)
            return 1
        print(f"OK: {args.output} matches the committed artifacts")
        return 0

    written = provenance.write_manifest(attestation, args.output)
    print(f"wrote {written.relative_to(PROJECT_ROOT).as_posix()} "
          f"({len(attestation['artifacts'])} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
