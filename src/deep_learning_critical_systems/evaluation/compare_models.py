"""Create the final comparison from validated, versioned test evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from deep_learning_critical_systems.evaluation.evidence import (
    load_json,
    verify_evidence,
)


def load_comparison_data(
    evidence_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load model and baseline results from the audited evidence bundle."""

    verify_evidence(evidence_directory)
    payload = load_json(evidence_directory / "final_metrics.json")
    models = [
        {"model": model_name, **model_result["metrics"]}
        for model_name, model_result in payload["models"].items()
    ]
    baseline = {
        "model": "Majority Baseline",
        **payload["majority_baseline"]["metrics"],
    }
    return models, baseline


def print_results(models: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    """Print the final model comparison."""

    print("\n=== FINAL MODEL COMPARISON ===\n")
    print("Model             | Accuracy | Macro-F1 | Increase Recall")
    print("-" * 61)
    for result in [*models, baseline]:
        print(
            f"{result['model']:<17} | "
            f"{result['accuracy'] * 100:>7.2f}% | "
            f"{result['macro_f1'] * 100:>7.2f}% | "
            f"{result['stress_increase_recall'] * 100:>14.2f}%"
        )


def create_comparison_plot(
    models: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_path: Path,
) -> None:
    """Create a grouped comparison plot for the three neural models."""

    model_names = [result["model"] for result in models]
    positions = list(range(len(model_names)))
    bar_width = 0.24
    figure, axis = plt.subplots(figsize=(10, 6))

    bars_by_metric = []
    for offset, metric, label in (
        (-bar_width, "accuracy", "Accuracy"),
        (0, "macro_f1", "Macro-F1"),
        (bar_width, "stress_increase_recall", "Stress Increase Recall"),
    ):
        bars_by_metric.append(
            axis.bar(
                [position + offset for position in positions],
                [result[metric] * 100 for result in models],
                width=bar_width,
                label=label,
            )
        )

    axis.axhline(
        baseline["accuracy"] * 100,
        linestyle="--",
        label="Majority Baseline Accuracy",
    )
    axis.set(
        title="Final Model Comparison on the Held-Out Test Set",
        xlabel="Model",
        ylabel="Score (%)",
        xticks=positions,
        xticklabels=model_names,
        ylim=(0, 55),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    for bars in bars_by_metric:
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.6,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def main(evidence_directory: Path, output_path: Path) -> None:
    """Validate evidence and regenerate the final comparison figure."""

    models, baseline = load_comparison_data(evidence_directory)
    print_results(models, baseline)
    create_comparison_plot(models, baseline, output_path)
    print("\nModel comparison completed from versioned evidence.")


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/reproduce_evidence.py so repository paths are explicit."
    )
