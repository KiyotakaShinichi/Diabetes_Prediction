"""Deployment and environment configuration contracts.

These parse the actual config files rather than relying on visual review, so a
deployment surface cannot silently drift away from the tested runtime.
"""
import json
import re

import pytest
import yaml

from conftest import REPO_ROOT

# The canonical runtime. CI, the Dockerfile and Render must all agree.
CANONICAL_PYTHON = "3.11"
CANONICAL_PYTHON_FULL = "3.11.16"

RENDER_YAML = REPO_ROOT / "render.yaml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DEVCONTAINER = REPO_ROOT / ".devcontainer" / "devcontainer.json"
DOCKERFILE = REPO_ROOT / "Dockerfile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

PRODUCTION_LOCK = "requirements.lock"

# Environment variables the source actually reads as configuration. Verified by
# grepping for os.getenv / st.secrets; if one is removed from .env.example the
# contract test below fails.
REQUIRED_ENV_VARS = {
    "DATABASE_URL": "inference_db.py",
    "ADMIN_USERNAME": "admin_auth.py",
    "ADMIN_PASSWORD": "admin_auth.py",
}

# Service names that must survive any runtime realignment.
EXPECTED_RENDER_SERVICES = {"diabetes-api", "diabetes-app", "diabetes-admin"}


@pytest.fixture(scope="module")
def render_config() -> dict:
    return yaml.safe_load(RENDER_YAML.read_text(encoding="utf-8"))


def _python_services(config: dict) -> list[dict]:
    return [s for s in config["services"] if s.get("runtime") == "python"]


def _env_value(service: dict, key: str) -> str | None:
    for entry in service.get("envVars", []):
        if entry.get("key") == key:
            return entry.get("value")
    return None


# ------------------------------------------------------------ render.yaml

def test_render_defines_the_expected_services(render_config):
    names = {service["name"] for service in render_config["services"]}

    assert names == EXPECTED_RENDER_SERVICES


def test_every_python_service_pins_the_canonical_python(render_config):
    services = _python_services(render_config)

    assert services, "no python services found"
    for service in services:
        version = _env_value(service, "PYTHON_VERSION")
        assert version == CANONICAL_PYTHON_FULL, f"{service['name']} pins {version}"


def test_render_python_version_is_fully_qualified(render_config):
    """Render requires a full X.Y.Z when PYTHON_VERSION is set."""
    for service in _python_services(render_config):
        version = _env_value(service, "PYTHON_VERSION")
        assert re.fullmatch(r"\d+\.\d+\.\d+", version or ""), version


def test_every_python_service_builds_from_the_production_lock(render_config):
    for service in _python_services(render_config):
        build = service.get("buildCommand", "")
        assert PRODUCTION_LOCK in build, f"{service['name']}: {build!r}"


def test_no_service_installs_the_unpinned_requirements_file(render_config):
    """requirements.txt holds ranges; production must not resolve them freshly."""
    for service in render_config["services"]:
        build = service.get("buildCommand", "")
        assert not re.search(r"requirements\.txt(?!\S)", build), f"{service['name']}: {build!r}"


def test_render_start_commands_are_preserved(render_config):
    starts = {s["name"]: s.get("startCommand", "") for s in render_config["services"]}

    assert "uvicorn app:app" in starts["diabetes-api"]
    assert "streamlit run streamlit_app.py" in starts["diabetes-app"]
    assert "streamlit run admin_app.py" in starts["diabetes-admin"]
    for name, command in starts.items():
        assert "$PORT" in command, f"{name} no longer binds Render's port"


def test_api_service_keeps_its_health_check(render_config):
    api = next(s for s in render_config["services"] if s["name"] == "diabetes-api")

    assert api.get("healthCheckPath") == "/health"


def test_render_commits_no_secret_values(render_config):
    """envVars in a committed blueprint must never carry credentials."""
    forbidden = re.compile(r"postgres(ql)?://|password|secret|token|api[_-]?key", re.IGNORECASE)
    for service in render_config["services"]:
        for entry in service.get("envVars", []):
            value = str(entry.get("value", ""))
            assert not forbidden.search(value), f"{service['name']}.{entry.get('key')}"


# ------------------------------------------------------- runtime agreement

def test_dockerfile_uses_the_canonical_python_and_lock():
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert f"FROM python:{CANONICAL_PYTHON}-slim" in text
    assert f"-r {PRODUCTION_LOCK}" in text
    assert not re.search(r"-r requirements\.txt(?!\S)", text)


def test_ci_workflow_uses_the_canonical_python_and_locks():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    declared = str(workflow["env"]["PYTHON_VERSION"])

    assert declared in (CANONICAL_PYTHON, CANONICAL_PYTHON_FULL)
    text = WORKFLOW.read_text(encoding="utf-8")
    assert f"-r {PRODUCTION_LOCK}" in text
    assert "-r requirements-dev.lock" in text


def test_ci_exposes_static_training_and_coverage_gates():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    assert jobs["static"]["name"] == "Lint and typecheck"
    static_steps = {step["name"]: step.get("run", "") for step in jobs["static"]["steps"]}
    assert static_steps["Verify dependency source-to-lock contract"] == "python tools/verify_dependency_contract.py"
    assert static_steps["Lint governed scope"] == "ruff check ."
    assert static_steps["Typecheck owned stable modules"] == "python -m mypy"

    test_steps = {step["name"]: step.get("run", "") for step in jobs["test"]["steps"]}
    assert "tests/test_training_smoke.py" in test_steps["Run deterministic training smoke"]
    assert "--ignore=tests/test_training_smoke.py" in test_steps["Run remaining test suite"]
    assert test_steps["Enforce maintained-module coverage ratchet"] == "coverage report"


def test_devcontainer_uses_the_canonical_python_family_and_lock():
    raw = DEVCONTAINER.read_text(encoding="utf-8")
    config = json.loads(re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE))

    assert f"-{CANONICAL_PYTHON}-" in config["image"]
    update = config["updateContentCommand"]
    assert PRODUCTION_LOCK in update
    assert "requirements-dev.lock" in update
    assert not re.search(r"-r requirements\.txt(?!\S)", update)


# ------------------------------------------------------------ .env.example

def _documented_env_vars() -> set[str]:
    """Assignments of the form NAME=... at the start of a line."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))


def test_env_example_exists_and_is_tracked():
    assert ENV_EXAMPLE.is_file()


@pytest.mark.parametrize(("variable", "source"), sorted(REQUIRED_ENV_VARS.items()))
def test_required_env_var_is_documented(variable, source):
    """Fails if a variable the code reads is dropped from .env.example."""
    assert variable in _documented_env_vars(), f"{variable} (read by {source}) is undocumented"


@pytest.mark.parametrize(("variable", "source"), sorted(REQUIRED_ENV_VARS.items()))
def test_required_env_var_is_still_read_by_the_source(variable, source):
    """The other direction: .env.example must not document dead variables."""
    text = (REPO_ROOT / source).read_text(encoding="utf-8")

    assert f'"{variable}"' in text or f"'{variable}'" in text


def test_env_example_contains_no_real_credentials():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    # The removed default credential must never be suggested here. Assembled
    # rather than written literally so the string exists in exactly one place
    # (tests/test_admin_security.py owns the repository-wide sentinel check).
    assert "admin" + "12345" not in text
    # No populated connection string, only the documented placeholder shape.
    for match in re.findall(r"^DATABASE_URL=(.*)$", text, re.MULTILINE):
        assert match.strip() == "", f"DATABASE_URL must ship empty, got {match!r}"
    # No hex blobs that could be a hash, salt or token.
    assert not re.search(r"^[A-Z_]+=[0-9a-f]{32,}$", text, re.MULTILINE)


def test_env_example_uses_an_unmistakable_password_placeholder():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    value = re.search(r"^ADMIN_PASSWORD=(.*)$", text, re.MULTILINE)

    assert value, "ADMIN_PASSWORD is not documented"
    assert value.group(1).strip() == "change-me"


def test_dotenv_files_are_ignored_but_the_example_is_not():
    rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in rules
    assert "!.env.example" in rules


# ------------------------------------------------ CI gates that must not vanish
#
# The workflow already runs a vulnerability audit and a container smoke that
# exercises the real endpoints, but nothing pinned either one: both could be
# deleted and every test would still pass. These assert the gate exists and does
# the work its name claims, without depending on formatting or step ordering.


def workflow_jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def job_commands(job: dict) -> str:
    """Every shell command in a job, as one searchable block."""
    return "\n".join(step.get("run", "") or "" for step in job["steps"])


def test_ci_audits_both_dependency_locks():
    """A vulnerability gate that audits only one lock is half a gate."""
    commands = job_commands(workflow_jobs()["audit"])

    assert "pip-audit" in commands
    assert PRODUCTION_LOCK in commands
    assert "requirements-dev.lock" in commands


@pytest.mark.parametrize("endpoint", ["/health", "/ready", "/predict"])
def test_ci_container_smoke_exercises_the_real_endpoint(endpoint):
    """The smoke must reach the running container, not just build an image."""
    commands = job_commands(workflow_jobs()["docker"])

    assert "docker build" in commands
    assert endpoint in commands, f"the container smoke never calls {endpoint}"


def test_ci_container_smoke_asserts_the_prediction_response():
    """Calling /predict proves routing; reading the body proves it served."""
    commands = job_commands(workflow_jobs()["docker"])

    assert "probability" in commands or "risk_category" in commands


# -------------------------------------------- the experiment inventory stays true

EXPERIMENT_INVENTORY = REPO_ROOT / "experiments" / "README.md"

#: Modules governed by lint/typecheck or otherwise maintained, so they are not
#: archival studies and are deliberately absent from the historical inventory.
MAINTAINED_TOP_LEVEL = {
    "app.py", "streamlit_app.py", "admin_app.py", "admin_auth.py",
    "inference_db.py", "conftest.py", "create_admin_user.py",
    "experiment_config.py", "logisticregression_only.py", "boostedtrees_ab.py",
}


def tracked_top_level_scripts() -> set[str]:
    return {
        path.name
        for path in REPO_ROOT.glob("*.py")
        if path.name not in MAINTAINED_TOP_LEVEL
    }


def test_every_archival_script_is_classified():
    """A new stray script at the root must be classified, not silently ignored.

    The inventory is the ownership record for code outside the lint and
    typecheck boundary. If it drifts out of date, "unlinted but inventoried"
    quietly becomes "unlinted and forgotten".
    """
    inventory = EXPERIMENT_INVENTORY.read_text(encoding="utf-8")

    unlisted = sorted(
        name for name in tracked_top_level_scripts() if f"`{name}`" not in inventory
    )
    assert unlisted == [], f"not classified in experiments/README.md: {unlisted}"


def inventory_table_rows() -> list[str]:
    """Only the classification table - the surrounding prose is not a contract."""
    return [
        line for line in EXPERIMENT_INVENTORY.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `") and line.count("|") >= 4
    ]


def test_the_inventory_classifies_the_maintained_pipelines_as_maintained():
    rows = {row.split("|")[1].strip(): row.split("|")[2].strip() for row in inventory_table_rows()}

    assert rows["`logisticregression_only.py`"] == "MAINTAINED"
    assert rows["`boostedtrees_ab.py`"] == "MAINTAINED"


def test_the_inventory_uses_only_the_declared_classifications():
    """Free-text statuses would make the inventory unauditable."""
    table_rows = inventory_table_rows()

    assert table_rows, "expected a classification table"
    for row in table_rows:
        status = row.split("|")[2].strip()
        assert status in {"MAINTAINED", "HISTORICAL-VALUABLE", "REDUNDANT/DEAD"}, row


# ----------------------------------------------------------------- changelog

CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def test_the_changelog_exists_and_is_tracked():
    assert CHANGELOG.is_file()


def test_the_changelog_claims_no_release_that_has_not_happened():
    """No tag exists, so the file must not imply a shipped version.

    A fabricated release line is the cheapest way to inflate a maturity signal,
    and the least honest. This fails if a semantic version heading appears.
    """
    text = CHANGELOG.read_text(encoding="utf-8")

    assert "Unreleased" in text
    versioned = re.findall(r"^#+\s*\[?v?\d+\.\d+\.\d+", text, re.MULTILINE)
    assert versioned == [], f"changelog announces releases: {versioned}"


def test_every_changelog_commit_reference_exists():
    """Each milestone cites a real commit, so the history can be audited."""
    import subprocess

    text = CHANGELOG.read_text(encoding="utf-8")
    referenced = set(re.findall(r"`([0-9a-f]{7})`", text))

    assert referenced, "expected commit references"
    for sha in sorted(referenced):
        result = subprocess.run(
            ["git", "cat-file", "-t", sha],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.stdout.strip() == "commit", f"{sha} is not a commit in this repository"
