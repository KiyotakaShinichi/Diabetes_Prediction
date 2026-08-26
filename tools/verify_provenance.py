"""Verify provenance manifests against the files they attest to.

    python tools/verify_provenance.py                       # all manifests found
    python tools/verify_provenance.py path/to/manifest.json # one manifest

Exits non-zero on any integrity failure. It does not print a warning and
succeed: a manifest that no longer matches its artifacts is a failure, and CI
must treat it as one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_core import provenance  # noqa: E402 - after sys.path setup


def discover_manifests(root: Path) -> list[Path]:
    """Every committed manifest: the legacy attestation and any training runs."""
    found = sorted((root / "provenance").glob("*.json")) if (root / "provenance").is_dir() else []
    found += sorted((root / "model_artifacts").glob("*training_manifest.json"))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)

    manifests = args.manifests or discover_manifests(args.project_root)
    if not manifests:
        print("no provenance manifests found", file=sys.stderr)
        return 1

    failures = 0
    for manifest in manifests:
        problems = provenance.verify_manifest_file(manifest, args.project_root)
        label = manifest.relative_to(args.project_root).as_posix() if manifest.is_absolute() else str(manifest)
        if problems:
            failures += 1
            print(f"FAILED {label} ({len(problems)} problem(s))", file=sys.stderr)
            for problem in problems:
                print(f"   - {problem}", file=sys.stderr)
        else:
            print(f"OK     {label}")

    if failures:
        print(f"\n{failures} manifest(s) failed integrity verification", file=sys.stderr)
        return 1
    print(f"\nall {len(manifests)} manifest(s) verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
