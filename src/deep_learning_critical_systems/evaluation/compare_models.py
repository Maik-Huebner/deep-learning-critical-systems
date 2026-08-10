"""Compare the final test results of all financial-stress models.

The values used here are the frozen test results from the completed
MLP, LSTM and tuned Transformer evaluations.

This script is for reporting and visualization only. It does not
perform model selection or hyperparameter tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "model_comparison.png"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "model_comparison_results.json"
)


MODEL_RESULTS = [
    {
        "model": "MLP",
        "accuracy": 0.4392,
        "macro_precision": 0.6436,
        "macro_recall": 0.4124,
        "macro_f1": 0.3344,
        "stress_increase_recall": 0.0037,
    },
    {
        "model": "LSTM",
        "accuracy": 0.4433,
        "macro_precision": 0.4326,
        "macro_recall": 0.4312,
        "macro_f1": 0.3790,
        "stress_increase_recall": 0.0615,
    },
    {
        "model": "Transformer",
        "accuracy": 0.4244,
        "macro_precision": 0.4068,
        "macro_recall": 0.4241,
        "macro_f1": 0.3738,
        "stress_increase_recall": 0.0819,
    },
]


MAJORITY_BASELINE = {
    "model": "Majority Baseline",
    "accuracy": 0.3064,
    "macro_precision": 0.1021,
    "macro_recall": 0.3333,
    "macro_f1": 0.1563,
    "stress_increase_recall": 0.0,
}


def print_results() -> None:
    """Print the final model comparison."""

    print()
    print(
        "=== FINAL MODEL COMPARISON ==="
    )

    print()

    print(
        "Model        | Accuracy | Macro-F1 | "
        "Increase Recall"
    )

    print(
        "-" * 58
    )

    for result in MODEL_RESULTS:
        print(
            f"{result['model']:<12} | "
            f"{result['accuracy'] * 100:>7.2f}% | "
            f"{result['macro_f1'] * 100:>7.2f}% | "
            f"{result['stress_increase_recall'] * 100:>14.2f}%"
        )

    print(
        f"{MAJORITY_BASELINE['model']:<12} | "
        f"{MAJORITY_BASELINE['accuracy'] * 100:>7.2f}% | "
        f"{MAJORITY_BASELINE['macro_f1'] * 100:>7.2f}% | "
        f"{MAJORITY_BASELINE['stress_increase_recall'] * 100:>14.2f}%"
    )


def save_results() -> None:
    """Save model comparison metrics as JSON."""

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "evaluation_split": "held_out_test_set",
        "test_samples": 1694,
        "models": MODEL_RESULTS,
        "majority_baseline": MAJORITY_BASELINE,
        "summary": {
            "highest_accuracy": "LSTM",
            "highest_macro_f1": "LSTM",
            "highest_stress_increase_recall": "Transformer",
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved:",
        RESULTS_PATH,
    )


def create_comparison_plot() -> None:
    """Create a grouped comparison plot for the three neural models."""

    model_names = [
        result[
            "model"
        ]
        for result in MODEL_RESULTS
    ]

    accuracy_values = [
        result[
            "accuracy"
        ]
        * 100
        for result in MODEL_RESULTS
    ]

    macro_f1_values = [
        result[
            "macro_f1"
        ]
        * 100
        for result in MODEL_RESULTS
    ]

    increase_recall_values = [
        result[
            "stress_increase_recall"
        ]
        * 100
        for result in MODEL_RESULTS
    ]

    positions = list(
        range(
            len(
                model_names
            )
        )
    )

    bar_width = 0.24

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    accuracy_bars = axis.bar(
        [
            position
            - bar_width
            for position in positions
        ],
        accuracy_values,
        width=bar_width,
        label="Accuracy",
    )

    macro_f1_bars = axis.bar(
        positions,
        macro_f1_values,
        width=bar_width,
        label="Macro-F1",
    )

    increase_recall_bars = axis.bar(
        [
            position
            + bar_width
            for position in positions
        ],
        increase_recall_values,
        width=bar_width,
        label="Stress Increase Recall",
    )

    axis.axhline(
        MAJORITY_BASELINE[
            "accuracy"
        ]
        * 100,
        linestyle="--",
        label=(
            "Majority Baseline Accuracy"
        ),
    )

    axis.set_title(
        "Final Model Comparison on the Held-Out Test Set"
    )

    axis.set_xlabel(
        "Model"
    )

    axis.set_ylabel(
        "Score (%)"
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        model_names
    )

    axis.set_ylim(
        0,
        55,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    for bars in (
        accuracy_bars,
        macro_f1_bars,
        increase_recall_bars,
    ):
        for bar in bars:
            value = bar.get_height()

            axis.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value
                + 0.6,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved:",
        FIGURE_PATH,
    )


def main() -> None:
    """Generate the final model comparison artifacts."""

    print_results()

    save_results()

    create_comparison_plot()

    print()

    print(
        "Model comparison completed."
    )


if __name__ == "__main__":
    main()