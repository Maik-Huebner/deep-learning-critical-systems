"""Validate versioned evidence or regenerate its model-comparison figure."""

from __future__ import annotations

import argparse

from deep_learning_critical_systems.evaluation.evidence import verify_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate reports/figures/model_comparison.png after validation.",
    )
    arguments = parser.parse_args()

    result = verify_evidence()
    print(f"Validated {result.files_checked} evidence files.")
    if arguments.write:
        from deep_learning_critical_systems.evaluation.compare_models import (
            main as write_comparison,
        )

        write_comparison()


if __name__ == "__main__":
    main()
