"""Portability contract: no machine-specific paths, no dependence on the CWD.

Every path in the repository must resolve from the project directory, so the
application and the training scripts behave identically wherever they are
launched. These tests are deliberately cheap: they exercise argument parsers
and path resolution, never a full training run.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import experiment_config
from conftest import REPO_ROOT

# The absolute path that used to be hardcoded across the experiment scripts.
PERSONAL_PATH_MARKERS = ("C:/Users/L/", "C:\\Users\\L\\", "/Users/L/Downloads", "Downloads/cleaned_data")

TEXT_SUFFIXES = {".py", ".toml", ".txt", ".yml", ".yaml", ".json", ".md", ".cfg", ".ini", ".lock"}

# Scripts DataFactor flagged, plus the two maintained pipelines.
FLAGGED_SCRIPTS = ["logreg+clustering.py", "xgboost_only.py", "qsvm.py"]
MAINTAINED_PIPELINES = ["logisticregression_only.py", "boostedtrees_ab.py"]


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def _tracked_text_files():
    return [p for p in _tracked_files() if p.suffix.lower() in TEXT_SUFFIXES and p.is_file()]


def _experiment_scripts():
    """Tracked scripts that route their paths through experiment_config."""
    scripts = []
    for path in _tracked_files():
        if path.suffix != ".py" or not path.is_file():
            continue
        if "experiment_config.require_direct_execution" in path.read_text(encoding="utf-8"):
            scripts.append(path)
    return sorted(scripts)


# ------------------------------------------------------- no personal paths

def test_no_tracked_file_contains_a_personal_absolute_path():
    """Proof for objective 1: the author's home directory is gone repo-wide."""
    offenders = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # This test file necessarily names the markers it searches for.
        if path.name == "test_portability.py":
            continue
        for marker in PERSONAL_PATH_MARKERS:
            if marker in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {marker}")
    assert offenders == []


def test_no_tracked_python_file_uses_a_windows_drive_letter_path():
    offenders = []
    for path in _tracked_text_files():
        if path.suffix != ".py" or path.name == "test_portability.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(f'"{d}:' in line or f"'{d}:" in line for d in "CDEF"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert offenders == []


def test_no_production_module_resolves_paths_from_the_working_directory():
    for name in ("app.py", "inference_db.py", "admin_auth.py", "admin_app.py", "streamlit_app.py"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "Path.cwd()" not in text, name
        assert "os.getcwd()" not in text, name
        assert "PROJECT_ROOT = Path(__file__).resolve().parent" in text, name


# --------------------------------------------------- experiment_config API

def test_project_root_is_the_repository_root():
    assert experiment_config.PROJECT_ROOT == REPO_ROOT


def test_default_paths_are_absolute_and_project_anchored():
    for path in (
        experiment_config.DEFAULT_DATA_PATH,
        experiment_config.LEGACY_DATA_PATH,
        experiment_config.ARTIFACTS_DIR,
        experiment_config.DEFAULT_RESULTS_DIR,
    ):
        assert path.is_absolute(), path
        assert path.parent == REPO_ROOT or path.is_relative_to(REPO_ROOT), path


def test_committed_dataset_is_the_maintained_default():
    assert experiment_config.DEFAULT_DATA_PATH.name == "cleaned_data.csv"
    assert experiment_config.DEFAULT_DATA_PATH.is_file()


def test_defaults_ignore_the_working_directory(foreign_cwd):
    """Same defaults from a directory outside the repository."""
    args = experiment_config.parse_args("probe", [])

    assert Path.cwd() == foreign_cwd
    assert args.data_path == experiment_config.DEFAULT_DATA_PATH
    assert args.results_dir == experiment_config.DEFAULT_RESULTS_DIR


def test_explicit_data_path_overrides_the_default(tmp_path):
    target = tmp_path / "my_data.csv"

    args = experiment_config.parse_args("probe", ["--data-path", str(target)])

    assert args.data_path == target


@pytest.mark.parametrize(
    "supplied",
    ["C:/somewhere/data.csv", r"C:\somewhere\data.csv", "/tmp/data.csv", "relative/data.csv"],
    ids=["windows-posix-sep", "windows-backslash-sep", "posix-absolute", "relative"],
)
def test_data_path_accepts_windows_and_posix_style_values(supplied):
    args = experiment_config.parse_args("probe", ["--data-path", supplied])

    assert isinstance(args.data_path, Path)
    assert args.data_path == Path(supplied)


def test_results_dir_override_and_result_path(tmp_path):
    args = experiment_config.parse_args("probe", ["--results-dir", str(tmp_path / "out")])

    resolved = experiment_config.result_path(args, "scores.csv")

    assert resolved == tmp_path / "out" / "scores.csv"
    assert resolved.parent.is_dir(), "result_path must create the directory"


def test_result_path_writes_only_under_the_requested_directory(tmp_path):
    """An overridden --results-dir must not leak output into the repository."""
    repo_results = REPO_ROOT / "experiment_results"
    existed_before = repo_results.exists()
    args = experiment_config.parse_args("probe", ["--results-dir", str(tmp_path / "out")])

    resolved = experiment_config.result_path(args, "scores.csv")

    assert resolved.is_relative_to(tmp_path)
    assert not resolved.is_relative_to(REPO_ROOT)
    assert repo_results.exists() == existed_before, "leaked into the repository"


def test_default_results_dir_is_gitignored():
    """The archived experiments write here by default; it must never be tracked."""
    ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "experiment_results/" in ignore_rules


def test_importing_an_experiment_script_refuses_instead_of_training():
    with pytest.raises(RuntimeError, match="standalone experiment script"):
        experiment_config.require_direct_execution("some.importing.module", "qsvm.py")


def test_direct_execution_is_permitted():
    assert experiment_config.require_direct_execution("__main__", "qsvm.py") is None


# ------------------------------------------------------------- script CLIs

def test_every_flagged_script_routes_through_experiment_config():
    scripts = {p.name for p in _experiment_scripts()}
    for name in FLAGGED_SCRIPTS:
        assert name in scripts, f"{name} still has hardcoded paths"


@pytest.mark.parametrize("script", FLAGGED_SCRIPTS)
def test_flagged_script_help_exposes_data_path_without_heavy_imports(script, tmp_path):
    """--help must work from outside the repo, with no optional deps installed.

    These archived experiments import plotly/kmodes/statsmodels, which are not
    in the lock, so the argument parsing has to happen before those imports.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-path" in result.stdout
    assert "--results-dir" in result.stdout


@pytest.mark.parametrize("script", MAINTAINED_PIPELINES)
def test_maintained_pipeline_exposes_data_path(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-path" in result.stdout
    assert "--artifacts-dir" in result.stdout
    assert str(REPO_ROOT / "cleaned_data.csv") in result.stdout.replace("\n", "").replace(" ", "")


@pytest.mark.parametrize("script", MAINTAINED_PIPELINES)
def test_maintained_pipeline_defaults_are_project_anchored(script):
    text = (REPO_ROOT / script).read_text(encoding="utf-8")

    assert 'PROJECT_ROOT = Path(__file__).resolve().parent' in text
    assert 'DATA_PATH = PROJECT_ROOT / "cleaned_data.csv"' in text
    assert 'ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"' in text


# --------------------------------------------------------- foreign-CWD API

def test_artifact_paths_resolve_from_a_foreign_cwd(foreign_cwd):
    import app

    assert app.MODEL_BUNDLE_PATH.is_absolute()
    assert app.MODEL_BUNDLE_PATH.is_file()
    assert app.BOOSTED_BUNDLE_PATH.is_file()
    assert app.SHAP_PATH_A.is_file()
    assert app.DRIFT_BASELINE_A.is_file()


def test_health_works_from_a_foreign_cwd(client, foreign_cwd):
    body = client.get("/health").json()

    assert Path.cwd() == foreign_cwd
    assert body["status"] == "ok"
    assert body["model_bundle_exists"] is True
    assert body["boosted_bundle_exists"] is True


def test_predict_works_from_a_foreign_cwd(client, valid_payload, foreign_cwd, isolated_db_path):
    response = client.post("/predict", json=valid_payload)

    assert Path.cwd() == foreign_cwd
    body = response.json()
    assert response.status_code == 200
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    # Logging still lands in the isolated temporary database, not the repo.
    assert isolated_db_path.is_file()
    assert isolated_db_path.is_relative_to(foreign_cwd.parent)


def test_admin_users_path_is_project_anchored_from_a_foreign_cwd(foreign_cwd):
    """Path resolution only. Credential semantics are the security track's work."""
    import admin_auth

    assert admin_auth.USERS_PATH == REPO_ROOT / "admin_users.json"
    assert admin_auth.USERS_PATH.is_absolute()


def test_running_from_a_foreign_cwd_leaves_the_repository_untouched(client, valid_payload, foreign_cwd):
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout

    client.get("/health")
    client.post("/predict", json=valid_payload)

    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    assert after == before
