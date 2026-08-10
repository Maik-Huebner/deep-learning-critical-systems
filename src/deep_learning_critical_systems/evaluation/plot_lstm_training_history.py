"""Create training-history plots for the LSTM model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]

HISTORY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "lstm_training_history.json"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

LOSS_FIGURE_PATH = (
    FIGURES_DIR
    / "lstm_training_loss.png"
)

ACCURACY_FIGURE_PATH = (
    FIGURES_DIR
    / "lstm_training_accuracy.png"
)


def load_training_history() -> dict:
    """Load the saved LSTM training history."""

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    return json.loads(
        HISTORY_PATH.read_text()
    )


def plot_loss(
    history: dict,
) -> None:
    """Plot training and validation loss across epochs."""

    train_loss = history[
        "train_loss"
    ]

    validation_loss = history[
        "validation_loss"
    ]

    best_epoch = history[
        "best_epoch"
    ]

    epochs = range(
        1,
        len(train_loss) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss",
    )

    axis.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation Loss",
    )

    axis.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.2,
        label=f"Best Epoch ({best_epoch})",
    )

    axis.set_title(
        "LSTM - Training and Validation Loss"
    )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Cross-Entropy Loss"
    )

    axis.set_xticks(
        list(epochs)
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        LOSS_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {LOSS_FIGURE_PATH}"
    )


def plot_accuracy(
    history: dict,
) -> None:
    """Plot training and validation accuracy across epochs."""

    train_accuracy = history[
        "train_accuracy"
    ]

    validation_accuracy = history[
        "validation_accuracy"
    ]

    best_epoch = history[
        "best_epoch"
    ]

    epochs = range(
        1,
        len(train_accuracy) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Training Accuracy",
    )

    axis.plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    axis.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.2,
        label=f"Best Epoch ({best_epoch})",
    )

    axis.set_title(
        "LSTM - Training and Validation Accuracy"
    )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Accuracy"
    )

    axis.set_xticks(
        list(epochs)
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        ACCURACY_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {ACCURACY_FIGURE_PATH}"
    )


def main() -> None:
    """Create all LSTM training-history figures."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_training_history()

    print(
        "=== LSTM TRAINING PLOTS ==="
    )

    print(
        "Epochs trained:",
        history["epochs_trained"],
    )

    print(
        "Best epoch:",
        history["best_epoch"],
    )

    plot_loss(
        history
    )

    plot_accuracy(
        history
    )

    print()
    print(
        "Training plots completed."
    )


if __name__ == "__main__":
    main()