"""Explicitly create or update an administrator in the runtime credential store.

    python create_admin_user.py --username alice

The password is read interactively with getpass, never from an argument: a
command-line password would be visible in shell history and in the process list
on a shared machine.

This is the ONLY way an account is created. Nothing in the application seeds an
administrator implicitly, and importing any module never writes to the store.
"""
from __future__ import annotations

import argparse
import getpass
import sys

import admin_auth


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update an administrator account.",
        epilog=(
            "The store is written to data/admin_users.json, which is gitignored. "
            "For a stateless deployment, set ADMIN_USERNAME and ADMIN_PASSWORD instead."
        ),
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Administrator username. An existing entry with this name is replaced.",
    )
    return parser.parse_args(argv)


def prompt_password(prompt_fn=getpass.getpass) -> str:
    """Read and confirm a password without echoing it."""
    password = prompt_fn(f"Password (min {admin_auth.MIN_PASSWORD_LENGTH} characters): ")
    confirmation = prompt_fn("Confirm password: ")
    if password != confirmation:
        raise ValueError("Passwords did not match")
    return password


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    username = args.username.strip()
    if not username:
        print("Username must not be empty.", file=sys.stderr)
        return 2

    existing = any(record.get("username") == username for record in admin_auth._load_users())

    try:
        password = prompt_password()
        admin_auth.create_or_update_user(username, password)
    except ValueError as exc:
        # Only the failure reason is printed - never the password itself.
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nAborted; nothing was written.", file=sys.stderr)
        return 130

    action = "Updated" if existing else "Created"
    print(f"{action} administrator {username!r} in {admin_auth.USERS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
