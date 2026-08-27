"""Portable path configuration for the training and experiment scripts.

Every path in this repository resolves from the project root - the directory
containing this file - rather than from the process working directory, so the
scripts and the serving application behave identically no matter where they are
launched from.

This module is deliberately import-safe: importing it never reads a dataset,
never writes a file and never trains anything, so the CLI and default-path
behaviour can be tested directly.

Two datasets are referenced across the repository:

* ``cleaned_data.csv`` is committed and is what the maintained pipelines
  (``logisticregression_only.py`` and ``boostedtrees_ab.py``) read. Its target
  column is ``Diabetes_binary`` and its features use the short BRFSS names
  (``GenHlth``, ``HighBP``, ``Age``, ...).

* ``cleaned_data_upd.csv`` is what the archived single-model experiments were
  written against. It is a renamed/re-encoded variant - target ``DiabetesStatus``
  with the string labels "No Diabetes"/"Diabetes", and long feature names such
  as ``GeneralHealth``, ``HasHighBP`` and ``AgeCategory``. That file is NOT
  committed, so the default below only names its expected location; pass
  ``--data-path`` to point at your own copy. The two schemas are not
  interchangeable, which is why the archived scripts do not silently fall back
  to ``cleaned_data.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

#: Directory containing this file, i.e. the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent

#: Dataset used by the maintained training pipelines (committed).
DEFAULT_DATA_FILENAME = "cleaned_data.csv"
DEFAULT_DATA_PATH = PROJECT_ROOT / DEFAULT_DATA_FILENAME

#: Dataset the archived experiments expect (not committed - see module docstring).
LEGACY_DATA_FILENAME = "cleaned_data_upd.csv"
LEGACY_DATA_PATH = PROJECT_ROOT / LEGACY_DATA_FILENAME

#: Where the maintained pipelines read and write model bundles.
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

#: Where archived experiments write their result CSVs. Gitignored.
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiment_results"


def build_parser(
    description: str,
    *,
    default_data_path: Path = DEFAULT_DATA_PATH,
    default_results_dir: Path = DEFAULT_RESULTS_DIR,
) -> argparse.ArgumentParser:
    """Build the shared experiment argument parser.

    ``--data-path`` and ``--results-dir`` accept either POSIX or Windows style
    separators; ``pathlib.Path`` normalises both on the running platform. A
    relative value is resolved against the current working directory, which is
    what a caller passing an explicit path expects, while the *defaults* stay
    anchored to the project root.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        metavar="CSV",
        help="Input dataset to train on.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        metavar="DIR",
        help="Directory for generated result CSVs; created on demand.",
    )
    return parser


def parse_args(
    description: str,
    argv: list[str] | None = None,
    *,
    default_data_path: Path = DEFAULT_DATA_PATH,
    default_results_dir: Path = DEFAULT_RESULTS_DIR,
) -> argparse.Namespace:
    """Parse experiment arguments. ``--help`` exits here, before any training."""
    parser = build_parser(
        description,
        default_data_path=default_data_path,
        default_results_dir=default_results_dir,
    )
    return parser.parse_args(argv)


def result_path(args: argparse.Namespace, filename: str) -> Path:
    """Return ``args.results_dir / filename``, creating the directory."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / filename


def require_direct_execution(module_name: str, script_name: str) -> None:
    """Refuse to run an experiment script as an imported module.

    These scripts execute top to bottom with no ``main()``; importing one would
    start a full training run as a side effect. Rather than restructure them
    (a separate task), fail loudly and point at the importable helpers here.
    """
    if module_name != "__main__":
        raise RuntimeError(
            f"{script_name} is a standalone experiment script and must be run "
            f"directly, e.g. `python {script_name} --help`. Import "
            f"experiment_config instead for its path helpers."
        )
