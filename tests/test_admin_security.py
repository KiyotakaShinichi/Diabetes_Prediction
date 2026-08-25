"""Admin authentication security contract.

The rule this suite defends: authentication is fail-closed. With nothing
explicitly configured, no account exists, no login succeeds, and importing any
module never creates one.

Every test redirects the credential store into tmp_path and clears the admin
environment variables, so nothing here can read or write real credentials.
"""
import importlib
import json
import logging
import subprocess
import sys

import pytest

import admin_auth
from conftest import REPO_ROOT

# The credential pair that used to be created implicitly. Named here ONLY as a
# forbidden sentinel: test_no_tracked_file_contains_the_former_default_password
# asserts it appears nowhere in tracked source, config or documentation.
FORBIDDEN_DEFAULT_USERNAME = "admin"
FORBIDDEN_DEFAULT_PASSWORD = "admin" + "12345"

GOOD_PASSWORD = "correct-horse-battery-staple"

#: The production store location, captured before the autouse fixture below
#: redirects USERS_PATH into tmp_path.
REAL_USERS_PATH = admin_auth.USERS_PATH

# Keeping the real 600k-iteration cost would add seconds to every auth test.
# test_password_hashing_uses_the_declared_work_factor exercises the real value.
FAST_ITERATIONS = 1_000

TEXT_SUFFIXES = {".py", ".toml", ".txt", ".yml", ".yaml", ".json", ".md", ".cfg", ".ini", ".lock", ".example"}


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Redirect the credential store and clear admin env vars for every test."""
    monkeypatch.setattr(admin_auth, "USERS_PATH", tmp_path / "data" / "admin_users.json")
    monkeypatch.delenv(admin_auth.ENV_USERNAME, raising=False)
    monkeypatch.delenv(admin_auth.ENV_PASSWORD, raising=False)
    return tmp_path / "data" / "admin_users.json"


@pytest.fixture
def fast_hashing(monkeypatch):
    monkeypatch.setattr(admin_auth, "PBKDF2_ITERATIONS", FAST_ITERATIONS)


def _tracked_text_files():
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for name in listing:
        path = REPO_ROOT / name
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            yield path


# --------------------------------------------------- 1, 20: no default creds

def test_no_default_credential_fallback_exists_in_admin_auth():
    source = (REPO_ROOT / "admin_auth.py").read_text(encoding="utf-8")

    assert "ensure_default_admin" not in source
    assert FORBIDDEN_DEFAULT_PASSWORD not in source
    # os.environ.get must never supply a usable credential default.
    assert 'os.environ.get(ENV_USERNAME, "")' in source
    assert 'os.environ.get(ENV_PASSWORD, "")' in source


def test_no_tracked_file_contains_the_former_default_password():
    """The literal must be gone from source, config and docs alike."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _tracked_text_files()
        if path.name != "test_admin_security.py"
        and FORBIDDEN_DEFAULT_PASSWORD in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == []


def test_ensure_default_admin_is_gone_repository_wide():
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _tracked_text_files()
        if path.name != "test_admin_security.py"
        and "ensure_default_admin" in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == []


# ------------------------------------------- 2, 3, 4, 5: import side effects

@pytest.mark.parametrize("module", ["admin_auth", "admin_app", "streamlit_app", "create_admin_user"])
def test_importing_a_module_writes_nothing(module, tmp_path, monkeypatch):
    """Importing must not create a credential store or any other file."""
    store_root = tmp_path / "probe"
    store_root.mkdir()
    monkeypatch.setattr(admin_auth, "USERS_PATH", store_root / "admin_users.json")

    before = set(store_root.rglob("*"))
    imported = importlib.import_module(module)
    importlib.reload(imported)

    assert set(store_root.rglob("*")) == before
    assert not (store_root / "admin_users.json").exists()


def test_importing_admin_app_creates_no_account(isolated_store):
    import admin_app

    importlib.reload(admin_app)

    assert not isolated_store.exists()
    assert admin_auth._load_users() == []


def test_missing_store_does_not_create_one(isolated_store):
    assert not isolated_store.exists()

    assert admin_auth._load_users() == []
    assert admin_auth.has_stored_users() is False

    assert not isolated_store.exists(), "reading the store must never create it"


def test_unconfigured_state_reports_itself():
    assert admin_auth.is_configured() is False
    assert admin_auth.authentication_status() == "unconfigured"


# ------------------------------------------------------ 6: fail closed

@pytest.mark.parametrize(
    ("username", "password"),
    [
        (FORBIDDEN_DEFAULT_USERNAME, FORBIDDEN_DEFAULT_PASSWORD),
        ("admin", "admin"),
        ("", ""),
        ("anyone", "anything"),
    ],
)
def test_authentication_fails_when_nothing_is_configured(username, password):
    assert admin_auth.authenticate_user(username, password) is False


def test_the_former_default_credential_no_longer_authenticates(fast_hashing):
    """Even with another admin configured, the old default must not work."""
    admin_auth.create_or_update_user("realadmin", GOOD_PASSWORD)

    assert admin_auth.authenticate_user(
        FORBIDDEN_DEFAULT_USERNAME, FORBIDDEN_DEFAULT_PASSWORD
    ) is False


# ------------------------------------------- 7, 8, 9, 10, 11: env provider

def test_only_username_configured_fails_closed(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")

    with pytest.raises(admin_auth.AdminAuthConfigurationError, match=admin_auth.ENV_PASSWORD):
        admin_auth.env_credentials()
    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is False
    assert admin_auth.is_configured() is False
    assert "misconfigured" in admin_auth.authentication_status()


def test_only_password_configured_fails_closed(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, GOOD_PASSWORD)

    with pytest.raises(admin_auth.AdminAuthConfigurationError, match=admin_auth.ENV_USERNAME):
        admin_auth.env_credentials()
    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is False
    assert admin_auth.is_configured() is False


def test_configured_environment_pair_authenticates(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, GOOD_PASSWORD)

    assert admin_auth.is_configured() is True
    assert admin_auth.authentication_status() == "configured via environment"
    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is True


def test_wrong_environment_password_is_rejected(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, GOOD_PASSWORD)

    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD + "x") is False


def test_wrong_environment_username_is_rejected(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, GOOD_PASSWORD)

    assert admin_auth.authenticate_user("bob", GOOD_PASSWORD) is False


def test_environment_credentials_are_not_created_as_a_stored_account(monkeypatch, isolated_store):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, GOOD_PASSWORD)

    admin_auth.authenticate_user("alice", GOOD_PASSWORD)

    assert not isolated_store.exists(), "env auth must not materialise a file account"


# ------------------------------- 12, 13, 14, 15: explicit file-backed store

def test_explicit_user_creation_writes_the_store(isolated_store, fast_hashing):
    assert not isolated_store.exists()

    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)

    assert isolated_store.is_file()
    assert admin_auth.has_stored_users() is True
    assert admin_auth.authentication_status() == "configured via user store"


def test_password_is_stored_as_a_salted_hash_not_plaintext(isolated_store, fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)

    raw = isolated_store.read_text(encoding="utf-8")
    record = json.loads(raw)["users"][0]

    assert GOOD_PASSWORD not in raw
    assert record["username"] == "alice"
    assert record["algorithm"] == admin_auth.HASH_ALGORITHM
    assert record["iterations"] == FAST_ITERATIONS
    assert len(record["salt"]) == 32
    assert record["password_hash"] != GOOD_PASSWORD
    assert len(record["password_hash"]) == 64


def test_salt_is_unique_per_user(isolated_store, fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)
    admin_auth.create_or_update_user("bob", GOOD_PASSWORD)

    users = json.loads(isolated_store.read_text(encoding="utf-8"))["users"]
    assert users[0]["salt"] != users[1]["salt"]
    assert users[0]["password_hash"] != users[1]["password_hash"]


def test_stored_credential_authenticates(fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)

    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is True


def test_wrong_stored_password_is_rejected(fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)

    assert admin_auth.authenticate_user("alice", "not-the-password") is False


def test_unknown_username_is_rejected(fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)

    assert admin_auth.authenticate_user("mallory", GOOD_PASSWORD) is False


def test_updating_a_user_replaces_rather_than_duplicates(isolated_store, fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)
    admin_auth.create_or_update_user("alice", "a-different-password")

    users = json.loads(isolated_store.read_text(encoding="utf-8"))["users"]
    assert len(users) == 1
    assert admin_auth.authenticate_user("alice", "a-different-password") is True
    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is False


@pytest.mark.parametrize(
    ("username", "password", "match"),
    [
        ("", GOOD_PASSWORD, "Username"),
        ("   ", GOOD_PASSWORD, "Username"),
        ("alice", "", "Password"),
        ("alice", "       ", "Password"),
        ("alice", "short", "at least"),
    ],
)
def test_invalid_bootstrap_input_is_refused(username, password, match, isolated_store, fast_hashing):
    with pytest.raises(ValueError, match=match):
        admin_auth.create_or_update_user(username, password)

    assert not isolated_store.exists(), "a refused request must write nothing"


def test_password_hashing_uses_the_declared_work_factor():
    """Exercises the real cost parameter, not the fast test override."""
    assert admin_auth.PBKDF2_ITERATIONS >= 600_000
    assert admin_auth.HASH_ALGORITHM == "pbkdf2_sha256"

    salt = "ab" * 16
    digest = admin_auth._hash_password(GOOD_PASSWORD, salt, iterations=1_000)

    assert digest != GOOD_PASSWORD
    assert admin_auth._hash_password(GOOD_PASSWORD, salt, iterations=1_000) == digest
    assert admin_auth._hash_password("other", salt, iterations=1_000) != digest


# ------------------------------------------------ 16: malformed store safety

@pytest.mark.parametrize(
    "contents",
    ["not json at all", "[]", '{"users": "not-a-list"}', '{"nope": 1}', "null", '{"users": [1, 2, 3]}'],
    ids=["invalid-json", "top-level-list", "users-not-list", "no-users-key", "null", "non-dict-records"],
)
def test_malformed_store_fails_closed(contents, isolated_store, fast_hashing):
    isolated_store.parent.mkdir(parents=True, exist_ok=True)
    isolated_store.write_text(contents, encoding="utf-8")

    assert admin_auth._load_users() == []
    assert admin_auth.has_stored_users() is False
    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is False
    assert admin_auth.authenticate_user(FORBIDDEN_DEFAULT_USERNAME, FORBIDDEN_DEFAULT_PASSWORD) is False


@pytest.mark.parametrize(
    "record",
    [
        {"username": "alice", "salt": "zzzz", "password_hash": "x", "iterations": 10},
        {"username": "alice", "salt": "ab" * 16, "password_hash": "x", "iterations": 0},
        {"username": "alice", "salt": "ab" * 16, "password_hash": "x", "iterations": "many"},
        {"username": "alice", "salt": "ab" * 16, "password_hash": "x", "algorithm": "rot13"},
        {"username": "alice"},
    ],
    ids=["bad-salt", "zero-iterations", "non-int-iterations", "unknown-algorithm", "no-hash"],
)
def test_corrupt_record_never_authenticates(record, isolated_store):
    isolated_store.parent.mkdir(parents=True, exist_ok=True)
    isolated_store.write_text(json.dumps({"users": [record]}), encoding="utf-8")

    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is False
    assert admin_auth.authenticate_user("alice", "") is False


def test_empty_credentials_are_refused_without_touching_the_store(fast_hashing):
    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)

    assert admin_auth.authenticate_user("", "") is False
    assert admin_auth.authenticate_user("alice", "") is False
    assert admin_auth.authenticate_user("", GOOD_PASSWORD) is False


# ------------------------------------------------ 17, 18: store is untracked

def test_admin_users_json_is_not_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    assert "admin_users.json" not in tracked
    assert "data/admin_users.json" not in tracked
    assert not [name for name in tracked if name.endswith("admin_users.json")]


@pytest.mark.parametrize("candidate", ["admin_users.json", "data/admin_users.json"])
def test_credential_store_locations_are_gitignored(candidate):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", candidate], cwd=REPO_ROOT, check=False
    )

    assert result.returncode == 0, f"{candidate} is not gitignored"


def test_runtime_store_lives_under_the_ignored_data_directory():
    assert REAL_USERS_PATH == REPO_ROOT / "data" / "admin_users.json"
    assert REAL_USERS_PATH.parent.name == "data"


# ------------------------------------------------------- 19: no secret leaks

def test_nothing_logs_the_password_or_hash(isolated_store, fast_hashing, caplog):
    caplog.set_level(logging.DEBUG)

    admin_auth.create_or_update_user("alice", GOOD_PASSWORD)
    admin_auth.authenticate_user("alice", GOOD_PASSWORD)
    admin_auth.authenticate_user("alice", "wrong-password-value")

    record = json.loads(isolated_store.read_text(encoding="utf-8"))["users"][0]
    logged = caplog.text
    assert GOOD_PASSWORD not in logged
    assert "wrong-password-value" not in logged
    assert record["password_hash"] not in logged
    assert record["salt"] not in logged


def test_malformed_store_logging_does_not_leak_contents(isolated_store, caplog):
    caplog.set_level(logging.DEBUG)
    secret_ish = "s3cr3t-lookalike-value"
    isolated_store.parent.mkdir(parents=True, exist_ok=True)
    isolated_store.write_text(f'{{"users": "{secret_ish}"}}', encoding="utf-8")

    admin_auth._load_users()

    assert secret_ish not in caplog.text


def test_configuration_error_message_names_the_variable_not_the_value(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, "")

    with pytest.raises(admin_auth.AdminAuthConfigurationError) as excinfo:
        admin_auth.env_credentials()

    assert admin_auth.ENV_PASSWORD in str(excinfo.value)
    assert GOOD_PASSWORD not in str(excinfo.value)


def test_status_string_never_contains_credentials(monkeypatch):
    monkeypatch.setenv(admin_auth.ENV_USERNAME, "alice")
    monkeypatch.setenv(admin_auth.ENV_PASSWORD, GOOD_PASSWORD)

    status = admin_auth.authentication_status()

    assert GOOD_PASSWORD not in status
    assert "alice" not in status


# ------------------------------------------------------- bootstrap CLI

def test_bootstrap_cli_requires_a_username():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "create_admin_user.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )

    assert result.returncode != 0
    assert "--username" in result.stderr


def test_bootstrap_cli_does_not_accept_a_password_argument():
    """A CLI password would leak into shell history and the process list."""
    source = (REPO_ROOT / "create_admin_user.py").read_text(encoding="utf-8")

    assert "getpass" in source
    assert '"--password"' not in source
    assert "'--password'" not in source


def test_bootstrap_cli_help_is_safe(isolated_store):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "create_admin_user.py"), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0
    assert "--username" in result.stdout
    assert not isolated_store.exists()


def test_bootstrap_prompt_requires_matching_confirmation():
    import create_admin_user

    answers = iter(["first-password", "second-password"])
    with pytest.raises(ValueError, match="did not match"):
        create_admin_user.prompt_password(prompt_fn=lambda _prompt: next(answers))


def test_bootstrap_main_creates_the_account(isolated_store, fast_hashing, monkeypatch, capsys):
    import create_admin_user

    monkeypatch.setattr(create_admin_user, "prompt_password", lambda **_kw: GOOD_PASSWORD)

    exit_code = create_admin_user.main(["--username", "alice"])

    assert exit_code == 0
    assert isolated_store.is_file()
    assert admin_auth.authenticate_user("alice", GOOD_PASSWORD) is True
    assert GOOD_PASSWORD not in capsys.readouterr().out


def test_bootstrap_main_refuses_a_short_password(isolated_store, fast_hashing, monkeypatch, capsys):
    import create_admin_user

    monkeypatch.setattr(create_admin_user, "prompt_password", lambda **_kw: "short")

    exit_code = create_admin_user.main(["--username", "alice"])

    assert exit_code == 2
    assert not isolated_store.exists()
    assert "short" not in capsys.readouterr().err.replace("Refused: Password must be at least 8 characters", "")


# --------------------------------------------------- docs stay truthful

@pytest.mark.parametrize("document", ["README.md", "README_DEPLOY.md", ".env.example"])
def test_documentation_does_not_promise_automatic_admin_creation(document):
    path = REPO_ROOT / document
    if not path.is_file():
        pytest.skip(f"{document} not present")
    text = path.read_text(encoding="utf-8").lower()

    assert "auto-created" not in text
    assert "automatically created" not in text
    assert FORBIDDEN_DEFAULT_PASSWORD not in text
