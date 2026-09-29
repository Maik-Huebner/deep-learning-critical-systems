"""Validate versioned evidence or regenerate its model-comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_learning_critical_systems.evaluation.evidence import verify_evidence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIRECTORY = REPO_ROOT / "reports" / "evidence"
COMPARISON_FIGURE = REPO_ROOT / "reports" / "figures" / "model_comparison.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate reports/figures/model_comparison.png after validation.",
    )
    arguments = parser.parse_args()

    result = verify_evidence(EVIDENCE_DIRECTORY)
    print(f"Validated {result.files_checked} evidence files.")
    if arguments.write:
        from deep_learning_critical_systems.evaluation.compare_models import (
            main as write_comparison,
        )

        write_comparison(EVIDENCE_DIRECTORY, COMPARISON_FIGURE)


if __name__ == "__main__":
    main()
