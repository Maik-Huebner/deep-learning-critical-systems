"""Run the package-owned repository audit from a source checkout."""

from pathlib import Path

from deep_learning_critical_systems.repository_audit import main

REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    raise SystemExit(main(REPO_ROOT))
