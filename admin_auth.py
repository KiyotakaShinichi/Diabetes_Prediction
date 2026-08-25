"""Admin authentication.

Fail-closed by design. There is NO default administrator: if neither credential
provider is explicitly configured, every authentication attempt is refused and
no account is ever created implicitly.

Two providers, checked in order:

1. Environment - ``ADMIN_USERNAME`` and ``ADMIN_PASSWORD``. Both must be set.
   Setting exactly one is a configuration error and fails closed rather than
   silently degrading. This is the provider a stateless deployment (Render,
   a container) should use; the values must be managed as platform secrets.

2. Runtime user store - a JSON file under the gitignored ``data/`` directory,
   written only by an explicit ``create_or_update_user()`` call (normally via
   ``create_admin_user.py``). Importing this module never touches it.

Passwords are stored as PBKDF2-HMAC-SHA256 with a 16-byte random salt. The
algorithm and iteration count are recorded per record so the cost can be raised
later without invalidating existing credentials. Nothing here ever logs a
password, a salt or a hash.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

#: Runtime credential store. Under data/, which is gitignored, and created only
#: by an explicit user-management operation - never on import.
USERS_PATH = PROJECT_ROOT / "data" / "admin_users.json"

HASH_ALGORITHM = "pbkdf2_sha256"

#: OWASP's recommended work factor for PBKDF2-HMAC-SHA256.
PBKDF2_ITERATIONS = 600_000

#: Records written before the iteration count was stored used this value.
LEGACY_PBKDF2_ITERATIONS = 120_000

#: Pre-existing repository rule, preserved rather than invented.
MIN_PASSWORD_LENGTH = 8

ENV_USERNAME = "ADMIN_USERNAME"
ENV_PASSWORD = "ADMIN_PASSWORD"  # noqa: S105 - variable name, not a secret

#: Salt used only to burn a comparable amount of time on a failed lookup, so an
#: unknown username is not distinguishable from a wrong password by timing.
_DUMMY_SALT = "00" * 16


class AdminAuthConfigurationError(RuntimeError):
    """Raised when the environment provider is only half-configured."""


def _hash_password(password: str, salt: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    )
    return derived.hex()


def _load_users() -> list[dict[str, Any]]:
    """Return stored user records, or an empty list.

    A missing store means "no file-backed users", never "create a default". A
    malformed store is refused rather than partially trusted, so a corrupted or
    tampered file cannot authenticate anyone.
    """
    if not USERS_PATH.exists():
        return []
    try:
        raw = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("Admin credential store at %s is unreadable: %s", USERS_PATH, type(exc).__name__)
        return []

    if not isinstance(raw, dict):
        logger.error("Admin credential store at %s is not a JSON object.", USERS_PATH)
        return []

    users = raw.get("users")
    if not isinstance(users, list):
        logger.error("Admin credential store at %s has no 'users' list.", USERS_PATH)
        return []

    valid = [record for record in users if isinstance(record, dict) and record.get("username")]
    if len(valid) != len(users):
        logger.warning(
            "Ignored %d malformed record(s) in the admin credential store.", len(users) - len(valid)
        )
    return valid


def _save_users(users: list[dict[str, Any]]) -> None:
    """Persist the store, creating data/ only for this explicit operation."""
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")


def env_credentials() -> tuple[str, str] | None:
    """Return the configured environment credentials, or None if disabled.

    Raises AdminAuthConfigurationError when exactly one of the pair is set:
    half-configured auth is a mistake worth surfacing, not something to paper
    over by falling back to another provider.
    """
    username = os.environ.get(ENV_USERNAME, "").strip()
    password = os.environ.get(ENV_PASSWORD, "")

    if not username and not password:
        return None
    if not username or not password:
        missing = ENV_USERNAME if not username else ENV_PASSWORD
        raise AdminAuthConfigurationError(
            f"{missing} is not set. Set both {ENV_USERNAME} and {ENV_PASSWORD}, or neither."
        )
    return username, password


def has_stored_users() -> bool:
    """True when the runtime store holds at least one usable record."""
    return bool(_load_users())


def is_configured() -> bool:
    """True when at least one credential provider is usable."""
    try:
        if env_credentials() is not None:
            return True
    except AdminAuthConfigurationError:
        return False
    return has_stored_users()


def authentication_status() -> str:
    """Human-readable state for the dashboard. Never includes secret values."""
    try:
        env = env_credentials()
    except AdminAuthConfigurationError as exc:
        return f"misconfigured: {exc}"
    providers = []
    if env is not None:
        providers.append("environment")
    if has_stored_users():
        providers.append("user store")
    if not providers:
        return "unconfigured"
    return "configured via " + " and ".join(providers)


def create_or_update_user(username: str, password: str) -> None:
    """Create or replace a stored administrator. Explicit action only."""
    username = username.strip()
    if not username:
        raise ValueError("Username must not be empty")
    if not password or not password.strip():
        raise ValueError("Password must not be empty")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    salt = os.urandom(16).hex()
    record = {
        "username": username,
        "algorithm": HASH_ALGORITHM,
        "iterations": PBKDF2_ITERATIONS,
        "salt": salt,
        "password_hash": _hash_password(password, salt, PBKDF2_ITERATIONS),
    }

    users = _load_users()
    for index, existing in enumerate(users):
        if existing.get("username") == username:
            users[index] = record
            break
    else:
        users.append(record)

    _save_users(users)
    logger.info("Stored administrator %r in the runtime credential store.", username)


def _verify_record(record: dict[str, Any], password: str) -> bool:
    """Constant-time check of a password against one stored record."""
    algorithm = record.get("algorithm", HASH_ALGORITHM)
    if algorithm != HASH_ALGORITHM:
        logger.error("Stored record uses unsupported algorithm %r; refusing.", algorithm)
        return False

    salt = record.get("salt", "")
    expected = record.get("password_hash", "")
    iterations = record.get("iterations", LEGACY_PBKDF2_ITERATIONS)
    if not isinstance(iterations, int) or iterations < 1:
        logger.error("Stored record has an invalid iteration count; refusing.")
        return False

    try:
        provided = _hash_password(password, salt, iterations)
    except ValueError:
        # bytes.fromhex() rejected the stored salt - treat as unusable, not as a
        # reason to fall through to any other credential.
        logger.error("Stored record has a malformed salt; refusing.")
        return False

    return hmac.compare_digest(expected, provided)


def authenticate_user(username: str, password: str) -> bool:
    """Return True only for an explicitly configured, matching credential."""
    username = (username or "").strip()
    if not username or not password:
        return False

    try:
        env = env_credentials()
    except AdminAuthConfigurationError as exc:
        logger.error("Admin authentication is misconfigured, refusing: %s", exc)
        return False

    if env is not None:
        env_user, env_password = env
        # Compare both halves in constant time, and always compare both so the
        # result does not reveal which half was wrong.
        user_ok = hmac.compare_digest(username, env_user)
        password_ok = hmac.compare_digest(password, env_password)
        if user_ok and password_ok:
            return True

    for record in _load_users():
        if hmac.compare_digest(str(record.get("username", "")), username):
            return _verify_record(record, password)

    # No matching record: still derive a hash so an unknown username costs about
    # the same as a wrong password.
    _hash_password(password, _DUMMY_SALT, LEGACY_PBKDF2_ITERATIONS)
    return False
