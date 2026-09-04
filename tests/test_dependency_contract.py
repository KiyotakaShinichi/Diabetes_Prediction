"""Behavioral tests for the source-to-lock dependency contract."""
from __future__ import annotations

import re
from pathlib import Path

from conftest import REPO_ROOT
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


def test_requirements_parsing_ignores_pip_option_lines(tmp_path):
    """Option lines are configuration, not dependencies.

    requirements-dev.txt carries --extra-index-url so PyTorch resolves from its
    CPU wheel index; feeding that to a specifier parser is a crash, not a
    contract violation.
    """
    from tools.verify_dependency_contract import _requirements

    path = tmp_path / "requirements.txt"
    path.write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "--index-url https://pypi.org/simple\n"
        "-c other.txt\n"
        "\n"
        "# a comment\n"
        "torch>=2.4,<3\n",
        encoding="utf-8",
    )

    parsed = _requirements(path)

    assert [requirement.name for requirement in parsed] == ["torch"]


def test_torch_is_a_declared_development_dependency():
    """Research-only: production serving loads no neural network."""
    dev = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    runtime = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert re.search(r"^torch[><=~]", dev, re.MULTILINE)
    assert not re.search(r"^torch[><=~]", runtime, re.MULTILINE), (
        "torch must not become a runtime dependency while production serves no DL model"
    )


def test_the_development_lock_pins_a_cpu_only_torch():
    """CI must not download a CUDA build; the brief forbids requiring CUDA."""
    lock = (REPO_ROOT / "requirements-dev.lock").read_text(encoding="utf-8")

    assert "torch==" in lock
    assert "+cpu" in lock, "the lock must pin the CPU wheel"
    for cuda_package in ("nvidia-", "triton=="):
        assert cuda_package not in lock, f"{cuda_package} would pull a CUDA stack into CI"


def test_the_development_lock_carries_its_own_index_urls():
    """`pip install -r requirements-dev.lock` must resolve +cpu unaided.

    PyPI does not host local-version wheels, so without the emitted index the
    lock would be uninstallable on a clean machine.
    """
    lock = (REPO_ROOT / "requirements-dev.lock").read_text(encoding="utf-8")

    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lock
