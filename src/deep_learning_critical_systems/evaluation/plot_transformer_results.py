"""Create plots for Transformer training and hyperparameter tuning."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

TUNING_RESULTS_PATH = (
    LOG_DIR
    / "transformer_tuning_results_final.json"
)

TRAINING_HISTORY_PATH = (
    LOG_DIR
    / "transformer_tuned_training_history.json"
)


def load_json(
    path: Path,
):
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return json.loads(
        path.read_text()
    )


def save_figure(
    figure,
    filename: str,
) -> None:
    """Save one figure to the report directory."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        FIGURE_DIR
        / filename
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved:",
        path,
    )


def plot_training_loss(
    history: dict,
) -> None:
    """Plot training and validation loss for the selected Transformer."""

    epochs = range(
        1,
        len(
            history[
                "train_loss"
            ]
        )
        + 1,
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    axis.plot(
        epochs,
        history[
            "train_loss"
        ],
        label="Training Loss",
    )

    axis.plot(
        epochs,
        history[
            "validation_loss"
        ],
        label="Validation Loss",
    )

    axis.axvline(
        history[
            "best_epoch"
        ],
        linestyle="--",
        label=(
            "Selected Epoch "
            f"{history['best_epoch']}"
        ),
    )

    axis.set_title(
        "Transformer Training and Validation Loss"
    )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Cross-Entropy Loss"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    save_figure(
        figure,
        "transformer_training_loss.png",
    )


def plot_training_accuracy(
    history: dict,
) -> None:
    """Plot training and validation accuracy."""

    epochs = range(
        1,
        len(
            history[
                "train_accuracy"
            ]
        )
        + 1,
    )

    train_accuracy = [
        value * 100
        for value in history[
            "train_accuracy"
        ]
    ]

    validation_accuracy = [
        value * 100
        for value in history[
            "validation_accuracy"
        ]
    ]

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    axis.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy",
    )

    axis.plot(
        epochs,
        validation_accuracy,
        label="Validation Accuracy",
    )

    axis.axvline(
        history[
            "best_epoch"
        ],
        linestyle="--",
        label=(
            "Selected Epoch "
            f"{history['best_epoch']}"
        ),
    )

    axis.set_title(
        "Transformer Training and Validation Accuracy"
    )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Accuracy (%)"
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    save_figure(
        figure,
        "transformer_training_accuracy.png",
    )


def plot_tuning_metric(
    results: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Create one tuning comparison bar chart."""

    run_ids = [
        result[
            "run_id"
        ]
        for result in results
    ]

    values = [
        result[
            metric_key
        ]
        * 100
        for result in results
    ]

    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    bars = axis.bar(
        run_ids,
        values,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Transformer Tuning Run"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
        strict=True,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        figure,
        filename,
    )


def plot_validation_loss(
    results: list[dict],
) -> None:
    """Compare best validation loss across tuning runs."""

    run_ids = [
        result[
            "run_id"
        ]
        for result in results
    ]

    values = [
        result[
            "validation_loss"
        ]
        for result in results
    ]

    figure, axis = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    bars = axis.bar(
        run_ids,
        values,
    )

    axis.set_title(
        "Transformer Hyperparameter Tuning: Validation Loss"
    )

    axis.set_xlabel(
        "Transformer Tuning Run"
    )

    axis.set_ylabel(
        "Validation Loss"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
        strict=True,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        figure,
        "transformer_tuning_validation_loss.png",
    )


def print_tuning_summary(
    results: list[dict],
) -> None:
    """Print a compact tuning table for documentation."""

    ranked_results = sorted(
        results,
        key=lambda result: (
            -result[
                "validation_macro_f1"
            ],
            result[
                "validation_loss"
            ],
        ),
    )

    print()
    print(
        "=== TRANSFORMER TUNING SUMMARY ==="
    )

    print()

    print(
        "Run | Macro-F1 | Accuracy | "
        "Increase Recall | Val Loss"
    )

    print(
        "-" * 60
    )

    for result in ranked_results:
        print(
            f"{result['run_id']:<3} | "
            f"{result['validation_macro_f1'] * 100:>7.2f}% | "
            f"{result['validation_accuracy'] * 100:>7.2f}% | "
            f"{result['recall_stress_increase'] * 100:>14.2f}% | "
            f"{result['validation_loss']:.4f}"
        )


def main() -> None:
    """Generate all Transformer training and tuning plots."""

    tuning_results = load_json(
        TUNING_RESULTS_PATH
    )

    training_history = load_json(
        TRAINING_HISTORY_PATH
    )

    print()
    print(
        "=== TRANSFORMER PLOT GENERATION ==="
    )

    print(
        "Tuning runs:",
        len(
            tuning_results
        ),
    )

    print(
        "Selected best epoch:",
        training_history[
            "best_epoch"
        ],
    )

    print()

    plot_training_loss(
        training_history
    )

    plot_training_accuracy(
        training_history
    )

    plot_tuning_metric(
        results=tuning_results,
        metric_key="validation_macro_f1",
        ylabel="Validation Macro-F1 (%)",
        title=(
            "Transformer Hyperparameter Tuning: "
            "Validation Macro-F1"
        ),
        filename=(
            "transformer_tuning_macro_f1.png"
        ),
    )

    plot_tuning_metric(
        results=tuning_results,
        metric_key="validation_accuracy",
        ylabel="Validation Accuracy (%)",
        title=(
            "Transformer Hyperparameter Tuning: "
            "Validation Accuracy"
        ),
        filename=(
            "transformer_tuning_accuracy.png"
        ),
    )

    plot_tuning_metric(
        results=tuning_results,
        metric_key="recall_stress_increase",
        ylabel="Stress Increase Recall (%)",
        title=(
            "Transformer Hyperparameter Tuning: "
            "Stress Increase Recall"
        ),
        filename=(
            "transformer_tuning_increase_recall.png"
        ),
    )

    plot_validation_loss(
        tuning_results
    )

    print_tuning_summary(
        tuning_results
    )

    print()
    print(
        "Transformer plots completed."
    )


if __name__ == "__main__":
    main()