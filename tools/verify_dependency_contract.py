"""Verify the repository's canonical requirements/lock ownership contract."""
from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "requirements.txt": "requirements.lock",
    "requirements-dev.txt": "requirements-dev.lock",
}


def _requirements(path: Path) -> list[Requirement]:
    parsed: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            parsed.append(Requirement(line))
    return parsed


def _pins(requirements: list[Requirement]) -> dict[str, set[str]]:
    pins: dict[str, set[str]] = defaultdict(set)
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError(f"lock entry for {name!r} is not an exact pin: {requirement}")
        pins[name].add(specifiers[0].version)
    return dict(pins)


def validate_dependency_contract(root: Path = PROJECT_ROOT) -> list[str]:
    """Return actionable contract violations; an empty list means valid."""
    problems: list[str] = []
    git_executable = shutil.which("git")
    if git_executable is None:
        return ["git executable not found; tracked dependency files cannot be verified"]
    tracked = set(
        # Executable is resolved by shutil.which; argv has no caller-controlled options.
        subprocess.run(  # noqa: S603
            [git_executable, "ls-files"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )
    parsed_sources: dict[str, list[Requirement]] = {}
    parsed_locks: dict[str, list[Requirement]] = {}

    for source_name, lock_name in SOURCES.items():
        for name in (source_name, lock_name):
            path = root / name
            if name not in tracked:
                problems.append(f"{name} is not tracked by git")
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                problems.append(f"{name} is missing or empty")

        source_path = root / source_name
        lock_path = root / lock_name
        if not source_path.is_file() or not lock_path.is_file():
            continue
        try:
            source_requirements = _requirements(source_path)
            lock_requirements = _requirements(lock_path)
            lock_pins = _pins(lock_requirements)
        except (OSError, ValueError) as exc:
            problems.append(f"cannot parse {source_name}/{lock_name}: {exc}")
            continue

        parsed_sources[source_name] = source_requirements
        parsed_locks[lock_name] = lock_requirements
        header = "\n".join(lock_path.read_text(encoding="utf-8").splitlines()[:3])
        expected_command = f"--output-file {lock_name} {source_name}"
        if expected_command not in header:
            problems.append(f"{lock_name} does not identify its canonical input/output")

        for direct in source_requirements:
            name = canonicalize_name(direct.name)
            versions = lock_pins.get(name)
            if not versions:
                problems.append(f"{lock_name} does not pin direct dependency {name}")
                continue
            for version in versions:
                if version not in direct.specifier:
                    problems.append(
                        f"{lock_name} pins {name}=={version}, outside {source_name} constraint {direct.specifier}"
                    )

    if {"requirements.lock", "requirements-dev.lock"} <= parsed_locks.keys():
        production = _pins(parsed_locks["requirements.lock"])
        development = _pins(parsed_locks["requirements-dev.lock"])
        for name in sorted(production.keys() & development.keys()):
            if production[name] != development[name]:
                problems.append(
                    f"shared dependency {name} differs: production={sorted(production[name])}, "
                    f"development={sorted(development[name])}"
                )

    runtime_names = {
        canonicalize_name(requirement.name)
        for requirement in parsed_sources.get("requirements.txt", [])
    }
    if "requests" not in runtime_names:
        problems.append("requests must be a direct runtime dependency for the public API client")
    return problems


def main() -> int:
    problems = validate_dependency_contract()
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print("Dependency contract valid: direct sources and deterministic locks are tracked and synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
