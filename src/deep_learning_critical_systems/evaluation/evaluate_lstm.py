"""Evaluate the final tuned LSTM financial-stress model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from deep_learning_critical_systems.data.datasets import (
    create_data_loaders,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    CLASS_NAMES,
    prepare_ofr_data,
)
from deep_learning_critical_systems.models.lstm import (
    FinancialStressLSTM,
)
from deep_learning_critical_systems.training.trainer import (
    select_device,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "lstm_tuned_model.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "lstm_test_metrics.json"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "lstm_confusion_matrix.png"
)

BATCH_SIZE = 64


def load_model(
    device: torch.device,
) -> tuple[
    FinancialStressLSTM,
    dict,
]:
    """Load the final validation-selected LSTM model."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model = FinancialStressLSTM(
        feature_count=checkpoint[
            "feature_count"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        classifier_hidden_size=checkpoint[
            "classifier_hidden_size"
        ],
        class_count=checkpoint[
            "class_count"
        ],
        dropout=checkpoint[
            "dropout"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


def predict(
    model: FinancialStressLSTM,
    test_loader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Predict all classes in the chronological test set."""

    predictions = []
    targets = []

    with torch.no_grad():
        for features, batch_targets in test_loader:
            features = features.to(
                device
            )

            logits = model(
                features
            )

            batch_predictions = logits.argmax(
                dim=1
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
                .tolist()
            )

            targets.extend(
                batch_targets
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            targets,
            dtype=np.int64,
        ),
        np.asarray(
            predictions,
            dtype=np.int64,
        ),
    )


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate the main classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "macro_precision": float(
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def create_majority_baseline(
    training_targets: np.ndarray,
    test_targets: np.ndarray,
) -> np.ndarray:
    """Predict the most frequent training class for every test sample."""

    class_counts = np.bincount(
        training_targets,
        minlength=len(
            CLASS_NAMES
        ),
    )

    majority_class = int(
        class_counts.argmax()
    )

    return np.full(
        shape=len(
            test_targets
        ),
        fill_value=majority_class,
        dtype=np.int64,
    )


def plot_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Create and save the final LSTM test confusion matrix."""

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_xticks(
        np.arange(
            len(CLASS_NAMES)
        )
    )

    axis.set_yticks(
        np.arange(
            len(CLASS_NAMES)
        )
    )

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=30,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_title(
        "Tuned LSTM - Test Confusion Matrix"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            axis.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {CONFUSION_MATRIX_PATH}"
    )


def save_metrics(
    lstm_metrics: dict[str, float],
    majority_metrics: dict[str, float],
    checkpoint: dict,
) -> None:
    """Save final LSTM, baseline and model-selection metadata as JSON."""

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "lstm": lstm_metrics,
        "majority_baseline": majority_metrics,
        "model_selection": {
            "selected_run": checkpoint.get(
                "selected_run"
            ),
            "selection_method": checkpoint.get(
                "selection_method"
            ),
            "selection_metric": checkpoint.get(
                "selection_metric"
            ),
            "selection_tie_breaker": checkpoint.get(
                "selection_tie_breaker"
            ),
            "canonical_model_seed": checkpoint.get(
                "canonical_model_seed"
            ),
            "stability_seeds": checkpoint.get(
                "stability_seeds"
            ),
            "hidden_size": checkpoint.get(
                "hidden_size"
            ),
            "classifier_hidden_size": checkpoint.get(
                "classifier_hidden_size"
            ),
            "dropout": checkpoint.get(
                "dropout"
            ),
            "learning_rate": checkpoint.get(
                "learning_rate"
            ),
        },
    }

    METRICS_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print(
        f"Saved: {METRICS_PATH}"
    )


def print_metric_block(
    name: str,
    metrics: dict[str, float],
) -> None:
    """Print one compact metrics block."""

    print()
    print(
        name
    )

    print(
        "Accuracy:",
        round(
            metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Macro Precision:",
        round(
            metrics[
                "macro_precision"
            ],
            4,
        ),
    )

    print(
        "Macro Recall:",
        round(
            metrics[
                "macro_recall"
            ],
            4,
        ),
    )

    print(
        "Macro F1:",
        round(
            metrics[
                "macro_f1"
            ],
            4,
        ),
    )


def main() -> None:
    """Run the final tuned LSTM test evaluation."""

    prepared_data = prepare_ofr_data()

    (
        _,
        _,
        test_loader,
    ) = create_data_loaders(
        prepared_data,
        batch_size=BATCH_SIZE,
    )

    device = select_device()

    (
        model,
        checkpoint,
    ) = load_model(
        device
    )

    (
        test_targets,
        lstm_predictions,
    ) = predict(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    majority_predictions = (
        create_majority_baseline(
            training_targets=prepared_data.y_train,
            test_targets=test_targets,
        )
    )

    lstm_metrics = calculate_metrics(
        test_targets,
        lstm_predictions,
    )

    majority_metrics = calculate_metrics(
        test_targets,
        majority_predictions,
    )

    print()
    print(
        "=== FINAL TUNED LSTM TEST EVALUATION ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH.name,
    )

    print(
        "Selected run:",
        checkpoint.get(
            "selected_run"
        ),
    )

    print(
        "Selection method:",
        checkpoint.get(
            "selection_method"
        ),
    )

    print(
        "Selection metric:",
        checkpoint.get(
            "selection_metric"
        ),
    )

    print(
        "Canonical model seed:",
        checkpoint.get(
            "canonical_model_seed"
        ),
    )

    print(
        "Hidden size:",
        checkpoint[
            "hidden_size"
        ],
    )

    print(
        "Classifier hidden size:",
        checkpoint[
            "classifier_hidden_size"
        ],
    )

    print(
        "Dropout:",
        checkpoint[
            "dropout"
        ],
    )

    print(
        "Learning rate:",
        checkpoint[
            "learning_rate"
        ],
    )

    print(
        "Test samples:",
        len(
            test_targets
        ),
    )

    print_metric_block(
        "Tuned LSTM:",
        lstm_metrics,
    )

    print_metric_block(
        "Majority-class baseline:",
        majority_metrics,
    )

    print()
    print(
        "=== CLASSIFICATION REPORT ==="
    )

    print(
        classification_report(
            test_targets,
            lstm_predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "=== CONFUSION MATRIX ==="
    )

    matrix = confusion_matrix(
        test_targets,
        lstm_predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    print(
        matrix
    )

    print()

    plot_confusion_matrix(
        test_targets,
        lstm_predictions,
    )

    save_metrics(
        lstm_metrics,
        majority_metrics,
        checkpoint,
    )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()
