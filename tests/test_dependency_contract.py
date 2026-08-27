"""Behavioral tests for the source-to-lock dependency contract."""
from __future__ import annotations

from pathlib import Path

from tools.verify_dependency_contract import validate_dependency_contract


def _write_contract(root: Path, *, production_lock: str, development_lock: str) -> None:
    (root / "requirements.txt").write_text("requests>=2,<3\n", encoding="utf-8")
    (root / "requirements-dev.txt").write_text("pytest>=8,<10\nrequests>=2,<3\n", encoding="utf-8")
    (root / "requirements.lock").write_text(
        "# generated\n# --output-file requirements.lock requirements.txt\n" + production_lock,
        encoding="utf-8",
    )
    (root / "requirements-dev.lock").write_text(
        "# generated\n# --output-file requirements-dev.lock requirements-dev.txt\n" + development_lock,
        encoding="utf-8",
    )


def _fake_git(monkeypatch, names: set[str]) -> None:
    class Result:
        stdout = "\n".join(sorted(names))

    monkeypatch.setattr(
        "tools.verify_dependency_contract.subprocess.run", lambda *args, **kwargs: Result()
    )


def test_repository_dependency_contract_is_synchronized():
    assert validate_dependency_contract() == []


def test_contract_detects_a_missing_direct_pin(tmp_path, monkeypatch):
    _write_contract(
        tmp_path,
        production_lock="urllib3==2.7.0\n",
        development_lock="pytest==9.1.1\nrequests==2.34.2\n",
    )
    _fake_git(monkeypatch, {"requirements.txt", "requirements-dev.txt", "requirements.lock", "requirements-dev.lock"})

    problems = validate_dependency_contract(tmp_path)

    assert any("does not pin direct dependency requests" in problem for problem in problems)


def test_contract_detects_shared_pin_drift(tmp_path, monkeypatch):
    _write_contract(
        tmp_path,
        production_lock="requests==2.34.2\n",
        development_lock="pytest==9.1.1\nrequests==2.33.0\n",
    )
    _fake_git(monkeypatch, {"requirements.txt", "requirements-dev.txt", "requirements.lock", "requirements-dev.lock"})

    problems = validate_dependency_contract(tmp_path)

    assert any("shared dependency requests differs" in problem for problem in problems)
