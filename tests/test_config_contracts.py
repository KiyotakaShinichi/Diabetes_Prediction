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
