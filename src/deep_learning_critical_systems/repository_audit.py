"""Small repository-hygiene audit for accidental local artifacts and secrets."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024

FORBIDDEN_PARTS = {
    ".DS_Store",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".pt", ".pth", ".pyc"}
RAW_DATA_PREFIXES = ("data/raw/", "data/interim/", "data/processed/")
ALLOWED_DATA_MARKERS = {
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
}

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
}


@dataclass(frozen=True)
class AuditFinding:
    """One repository-audit finding."""

    path: str
    reason: str


def tracked_files(root: Path) -> list[str]:
    """Return tracked and non-ignored repository files from the working tree."""

    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [item.decode() for item in result.stdout.split(b"\0") if item]


def audit_paths(root: Path, paths: list[str]) -> list[AuditFinding]:
    """Audit a known list of repository-relative tracked files."""

    findings: list[AuditFinding] = []
    for relative_path in paths:
        path = root / relative_path
        parts = set(Path(relative_path).parts)

        if parts & FORBIDDEN_PARTS:
            findings.append(
                AuditFinding(relative_path, "local metadata/cache is tracked")
            )
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(
                AuditFinding(relative_path, "checkpoint/cache binary is tracked")
            )
        if (
            relative_path.startswith(RAW_DATA_PREFIXES)
            and relative_path not in ALLOWED_DATA_MARKERS
        ):
            findings.append(
                AuditFinding(relative_path, "raw or processed data is tracked")
            )
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            findings.append(AuditFinding(relative_path, "tracked file exceeds 5 MiB"))

        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(AuditFinding(relative_path, f"possible {label}"))

    return findings


def audit_repository(root: Path) -> list[AuditFinding]:
    """Audit tracked and prospective repository contents.

    This narrow guard catches common accidental commits. It is not a replacement
    for a dedicated secret scanner, malware scanner or software-composition audit.
    """

    return audit_paths(root, tracked_files(root))


def main(root: Path) -> int:
    """Print findings and return a process-compatible status code."""

    findings = audit_repository(root)
    if findings:
        print("Repository audit failed:")
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}")
        return 1

    print(
        "Repository audit passed: no tracked local artifacts or obvious secrets found."
    )
    print("Note: this lightweight check is not a professional secret scanner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/audit_repository.py with an explicit repository path."
    )
