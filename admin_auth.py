import hashlib
import hmac
import json
import os
from pathlib import Path


USERS_PATH = Path("admin_users.json")


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120000)
    return dk.hex()


def _load_users() -> dict:
    if not USERS_PATH.exists():
        return {"users": []}
    with open(USERS_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_users(data: dict) -> None:
    with open(USERS_PATH, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def ensure_default_admin() -> None:
    data = _load_users()
    if data.get("users"):
        return

    default_user = os.getenv("ADMIN_USERNAME", "admin")
    default_password = os.getenv("ADMIN_PASSWORD", "admin12345")
    create_or_update_user(default_user, default_password)


def create_or_update_user(username: str, password: str) -> None:
    username = username.strip()
    if not username:
        raise ValueError("Username must not be empty")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    data = _load_users()
    users = data.get("users", [])
    salt = os.urandom(16).hex()
    record = {
        "username": username,
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }

    replaced = False
    for idx, existing in enumerate(users):
        if existing.get("username") == username:
            users[idx] = record
            replaced = True
            break

    if not replaced:
        users.append(record)

    data["users"] = users
    _save_users(data)


def authenticate_user(username: str, password: str) -> bool:
    data = _load_users()
    for user in data.get("users", []):
        if user.get("username") != username:
            continue
        salt = user.get("salt", "")
        expected = user.get("password_hash", "")
        provided = _hash_password(password, salt)
        return hmac.compare_digest(expected, provided)
    return False
