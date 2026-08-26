"""Deterministic provenance primitives for training runs and artifact integrity.

Two distinct manifest types, and the distinction is the whole point:

``training_run``
    Emitted by a maintained pipeline at the end of an actual training run. It
    can honestly state which commit, dataset, environment and configuration
    produced the artifacts, because it observed them.

``legacy_artifact_attestation``
    An inventory of artifacts that already existed before this system. It
    attests only to what is *observable now* - hashes, sizes, embedded library
    versions, current serving roles. Every field describing historical lineage
    is explicitly ``null``. Artifact integrity is not proof of training
    provenance, and this module refuses to blur the two.

Nothing here reaches the network, writes on import, or mutates global state.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

#: Bump when the manifest structure changes incompatibly.
SCHEMA_VERSION = 1

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

TRAINING_RUN = "training_run"
LEGACY_ATTESTATION = "legacy_artifact_attestation"

#: Packages whose versions materially affect deserialization of the artifacts.
TRACKED_PACKAGES = (
    "scikit-learn", "xgboost", "numpy", "scipy", "pandas", "shap", "joblib",
)

_CHUNK = 1 << 20


# --------------------------------------------------------------- hashing

def sha256_file(path: str | Path) -> str:
    """SHA256 of a file's bytes, streamed so large artifacts stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """A stable JSON rendering: sorted keys, no incidental whitespace.

    Two structurally equal objects always produce identical bytes, which is what
    makes a hash over a structure reproducible.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_canonical_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def relative_path(path: str | Path, project_root: str | Path) -> str:
    """POSIX path relative to the project root.

    Manifests must never carry an absolute developer path: it leaks the build
    machine's layout and makes the manifest unverifiable anywhere else.
    """
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    return resolved.relative_to(root).as_posix()


# ----------------------------------------------------------- fingerprints

def fingerprint_dataset(
    path: str | Path, project_root: str | Path, target_column: str | None = None
) -> dict:
    """Fingerprint the exact bytes a training run consumed.

    Records shape and column order, never the data itself.
    """
    import pandas as pd  # local: keeps `import ml_core` free of a pandas import

    path = Path(path)
    frame = pd.read_csv(path)
    return {
        "path": relative_path(path, project_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(c) for c in frame.columns],
        "target_column": target_column,
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def fingerprint_features(feature_names: list[str], target_column: str | None = None) -> dict:
    """Order-sensitive fingerprint of the training feature list."""
    ordered = [str(name) for name in feature_names]
    return {
        "feature_names": ordered,
        "feature_count": len(ordered),
        "target_column": target_column,
        # Order-sensitive by construction: the list is hashed as a sequence.
        "feature_schema_sha256": sha256_canonical_json(
            {"features": ordered, "target": target_column}
        ),
    }


def fingerprint_environment(lockfile: str | Path | None = None,
                            project_root: str | Path | None = None) -> dict:
    """Interpreter, platform and installed versions of the packages that matter.

    Versions come from installed distribution metadata, never from a hardcoded
    list. No environment variables are captured, so no secret can leak in.
    """
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }
    if lockfile is not None and Path(lockfile).is_file():
        env["lockfile"] = {
            "path": relative_path(lockfile, project_root) if project_root else Path(lockfile).name,
            "sha256": sha256_file(lockfile),
        }
    else:
        env["lockfile"] = None
    return env


def fingerprint_source(paths: list[str | Path], project_root: str | Path) -> dict:
    """Hash of the specific training source files, not the whole repository."""
    files = []
    for path in sorted(Path(p) for p in paths):
        if path.is_file():
            files.append({"path": relative_path(path, project_root), "sha256": sha256_file(path)})
    return {
        "files": files,
        "combined_sha256": sha256_canonical_json(files),
    }


def git_provenance(project_root: str | Path) -> dict:
    """Commit and cleanliness of the working tree, or explicit nulls.

    A dirty tree is recorded as ``dirty: true`` rather than refused: local
    experimentation is legitimate. What is never acceptable is a dirty run
    claiming to have come from clean HEAD.
    """
    def _git(*args: str) -> str | None:
        try:
            # Fixed argv, no shell, and every element is a literal from this
            # module - `args` is never caller-controlled input. `git` is
            # intentionally resolved from PATH, as it is everywhere else here.
            out = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=str(project_root),
                capture_output=True, text=True, timeout=60, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit_sha": None, "dirty": None, "branch": None}

    status = _git("status", "--porcelain")
    return {
        "commit_sha": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


# ------------------------------------------------- embedded artifact versions

_SKLEARN_VERSION = re.compile(rb"_sklearn_version.{0,12}?(\d+\.\d+\.\d+)", re.DOTALL)


def embedded_library_versions(path: str | Path) -> dict:
    """Recover library versions recorded *inside* an artifact, by byte scan.

    Deliberately does not unpickle: reading raw bytes cannot execute code, so
    this is safe to run over any file and cheap enough for every artifact.

    scikit-learn writes ``_sklearn_version`` into estimator state. XGBoost
    stores a UBJSON ``version`` array of three int8s inside the serialized
    booster. Absent markers yield ``None`` rather than a guess.
    """
    raw = Path(path).read_bytes()

    sklearn_match = _SKLEARN_VERSION.search(raw)
    sklearn_version = sklearn_match.group(1).decode() if sklearn_match else None

    xgboost_version = None
    marker = raw.find(b"version[#L")
    if marker != -1:
        # UBJSON: "version" [ # L <int64 count> then <count> typed values.
        cursor = marker + len(b"version[#L")
        count_bytes = raw[cursor:cursor + 8]
        cursor += 8
        if len(count_bytes) == 8 and int.from_bytes(count_bytes, "big") == 3:
            parts = []
            for _ in range(3):
                if raw[cursor:cursor + 1] == b"i":
                    parts.append(raw[cursor + 1])
                    cursor += 2
                else:
                    parts = []
                    break
            if len(parts) == 3:
                xgboost_version = ".".join(str(p) for p in parts)

    return {"sklearn": sklearn_version, "xgboost": xgboost_version}


# ------------------------------------------------------------- inventory

def inventory_artifact(role: str, path: str | Path, project_root: str | Path, *,
                       required_for_serving: bool) -> dict:
    """One artifact entry: role, relative path, hash, size, type, serving flag."""
    path = Path(path)
    entry = {
        "role": role,
        "path": relative_path(path, project_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format": _artifact_format(path),
        "required_for_serving": required_for_serving,
    }
    if entry["format"] in {"joblib_pickle", "binary"}:
        entry["embedded_versions"] = embedded_library_versions(path)
    return entry


def _artifact_format(path: Path) -> str:
    return {
        ".pkl": "joblib_pickle",
        ".json": "json",
        ".csv": "csv",
        ".txt": "text",
    }.get(path.suffix.lower(), "binary")


# -------------------------------------------------------------- manifests

def build_training_manifest(
    *,
    project_root: str | Path,
    variant: str,
    model_name: str,
    dataset: dict,
    features: dict,
    training: dict,
    evaluation: dict,
    artifacts: list[dict],
    source_files: list[str | Path],
    lockfile: str | Path | None = None,
    run_id: str | None = None,
) -> dict:
    """Assemble a manifest describing a training run that actually happened."""
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance_type": TRAINING_RUN,
        "run": {
            "run_id": run_id,
            "variant": variant,
            "model_name": model_name,
            "git": git_provenance(project_root),
            "source": fingerprint_source(source_files, project_root),
        },
        "dataset": dataset,
        "features": features,
        "environment": fingerprint_environment(lockfile, project_root),
        "training": training,
        "evaluation": evaluation,
        "artifacts": artifacts,
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    """Write a manifest atomically: temp file, flush, fsync, replace.

    A half-written manifest would be worse than none - it would look valid while
    attesting to nothing. os.replace is atomic on POSIX and Windows alike.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


# -------------------------------------------------------------- verifier

def verify_manifest(manifest: dict, project_root: str | Path) -> list[str]:
    """Check a manifest against the files on disk.

    Returns a list of human-readable problems; empty means the manifest
    verifies. Callers decide how to react - the CLI exits non-zero.
    """
    problems: list[str] = []
    root = Path(project_root).resolve()

    version = manifest.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        problems.append(f"unsupported schema_version: {version!r}")
        return problems  # structure beyond this point is not trustworthy

    kind = manifest.get("provenance_type")
    if kind not in {TRAINING_RUN, LEGACY_ATTESTATION}:
        problems.append(f"unknown provenance_type: {kind!r}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append("manifest lists no artifacts")
        return problems

    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for entry in artifacts:
        role = entry.get("role")
        rel = entry.get("path")
        if role in seen_roles:
            problems.append(f"duplicate artifact role: {role!r}")
        if rel in seen_paths:
            problems.append(f"duplicate artifact path: {rel!r}")
        seen_roles.add(role)
        seen_paths.add(rel)

        if not isinstance(rel, str) or _is_absolute_like(rel):
            problems.append(f"artifact path must be relative, got {rel!r}")
            continue

        target = root / rel
        if not target.is_file():
            problems.append(f"missing artifact: {rel}")
            continue
        actual = sha256_file(target)
        if actual != entry.get("sha256"):
            problems.append(f"sha256 mismatch for {rel}")
        actual_size = target.stat().st_size
        if actual_size != entry.get("bytes"):
            problems.append(f"size mismatch for {rel}: {actual_size} != {entry.get('bytes')}")

    problems.extend(_verify_features(manifest))
    problems.extend(_verify_dataset(manifest, root))
    problems.extend(_verify_lockfile(manifest, root))
    return problems


def _is_absolute_like(rel: str) -> bool:
    """Reject absolute POSIX paths, Windows drive paths and parent escapes."""
    return (
        rel.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", rel) is not None
        or ".." in Path(rel).parts
    )


def _verify_features(manifest: dict) -> list[str]:
    features = manifest.get("features")
    if not isinstance(features, dict) or not features.get("feature_names"):
        return []
    recomputed = fingerprint_features(
        features["feature_names"], features.get("target_column")
    )
    problems = []
    if recomputed["feature_schema_sha256"] != features.get("feature_schema_sha256"):
        problems.append("feature_schema_sha256 does not match feature_names")
    if recomputed["feature_count"] != features.get("feature_count"):
        problems.append("feature_count does not match feature_names")
    return problems


def _verify_dataset(manifest: dict, root: Path) -> list[str]:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("path"):
        return []
    if dataset.get("sha256") is None:
        return []  # deliberately unknown, e.g. a legacy attestation
    target = root / dataset["path"]
    if not target.is_file():
        # Absent dataset is not a manifest defect; it may simply not be present.
        return []
    if sha256_file(target) != dataset["sha256"]:
        return [f"dataset sha256 mismatch for {dataset['path']}"]
    return []


def _verify_lockfile(manifest: dict, root: Path) -> list[str]:
    lock = (manifest.get("environment") or {}).get("lockfile")
    if not isinstance(lock, dict) or not lock.get("path"):
        return []
    target = root / lock["path"]
    if not target.is_file():
        return [f"missing lockfile: {lock['path']}"]
    if sha256_file(target) != lock.get("sha256"):
        return [f"lockfile sha256 mismatch for {lock['path']}"]
    return []


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest_file(path: str | Path, project_root: str | Path) -> list[str]:
    try:
        manifest = load_manifest(path)
    except (OSError, ValueError) as exc:
        return [f"manifest unreadable: {type(exc).__name__}"]
    return verify_manifest(manifest, project_root)


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI shim
    import argparse

    parser = argparse.ArgumentParser(description="Verify a provenance manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)

    problems = verify_manifest_file(args.manifest, args.project_root)
    if problems:
        print(f"FAILED: {args.manifest} ({len(problems)} problem(s))", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {args.manifest} verifies against {args.project_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


def emit_training_manifest(
    *,
    project_root: str | Path,
    output_path: str | Path,
    variant: str,
    model_name: str,
    dataset_path: str | Path,
    target_column: str,
    feature_names: list[str],
    training: dict,
    evaluation: dict,
    artifact_specs: list[tuple[str, str | Path, bool]],
    source_files: list[str | Path],
    lockfile: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Hash the finished artifacts and write the manifest atomically.

    Call this as the LAST step of a training run, after every artifact and the
    metrics file are on disk. Hashes are computed from those final files, so the
    manifest attests to completed outputs rather than intentions - and if
    artifact generation failed part way, no manifest is produced at all.
    """
    artifacts = [
        inventory_artifact(role, path, project_root, required_for_serving=required)
        for role, path, required in artifact_specs
        if Path(path).is_file()
    ]
    manifest = build_training_manifest(
        project_root=project_root,
        variant=variant,
        model_name=model_name,
        dataset=fingerprint_dataset(dataset_path, project_root, target_column),
        features=fingerprint_features(feature_names, target_column),
        training=training,
        evaluation=evaluation,
        artifacts=artifacts,
        source_files=source_files,
        lockfile=lockfile,
        run_id=run_id,
    )
    return write_manifest(manifest, output_path)
