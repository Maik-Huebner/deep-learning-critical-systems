"""Final test evaluation for the tuned Transformer model."""

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
from deep_learning_critical_systems.models.transformer import (
    FinancialStressTransformer,
)
from deep_learning_critical_systems.training.trainer import (
    select_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "transformer_tuned_model.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "transformer_test_metrics.json"
)

CONFUSION_MATRIX_FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "transformer_confusion_matrix.png"
)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FinancialStressTransformer, dict]:
    """Load the tuned Transformer checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = FinancialStressTransformer(
        feature_count=checkpoint["feature_count"],
        model_dimension=checkpoint["model_dimension"],
        num_heads=checkpoint["num_heads"],
        feed_forward_size=checkpoint["feed_forward_size"],
        num_layers=checkpoint["num_layers"],
        classifier_hidden_size=checkpoint[
            "classifier_hidden_size"
        ],
        class_count=checkpoint["class_count"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


def collect_predictions(
    model: FinancialStressTransformer,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model and collect targets and predictions."""

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(
                device
            )

            logits = model(
                features
            )

            predictions = logits.argmax(
                dim=1
            )

            all_targets.extend(
                targets.numpy().tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

    targets_array = np.asarray(
        all_targets,
        dtype=np.int64,
    )

    predictions_array = np.asarray(
        all_predictions,
        dtype=np.int64,
    )

    return (
        targets_array,
        predictions_array,
    )


def determine_training_majority_class(
    training_targets: np.ndarray,
) -> int:
    """Determine the majority class using training labels only."""

    if training_targets.ndim != 1:
        raise ValueError(
            "Training targets must be one-dimensional."
        )

    if len(training_targets) == 0:
        raise ValueError(
            "Training targets must not be empty."
        )

    return int(
        np.bincount(
            training_targets.astype(
                np.int64
            )
        ).argmax()
    )


def create_majority_baseline(
    targets: np.ndarray,
    majority_class: int,
) -> np.ndarray:
    """Predict the training-set majority class for every sample."""

    return np.full_like(
        targets,
        fill_value=majority_class,
    )


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate standard multi-class evaluation metrics."""

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


def save_confusion_matrix_plot(
    matrix: np.ndarray,
) -> None:
    """Save the Transformer confusion matrix."""

    CONFUSION_MATRIX_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_title(
        "Transformer Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted Class"
    )

    axis.set_ylabel(
        "True Class"
    )

    axis.set_xticks(
        range(
            len(
                CLASS_NAMES
            )
        )
    )

    axis.set_yticks(
        range(
            len(
                CLASS_NAMES
            )
        )
    )

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=20,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    threshold = (
        matrix.max()
        / 2.0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            axis.text(
                column_index,
                row_index,
                str(
                    value
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_FIGURE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved:",
        CONFUSION_MATRIX_FIGURE_PATH,
    )


def save_metrics(
    checkpoint: dict,
    transformer_metrics: dict[str, float],
    majority_metrics: dict[str, float],
    majority_class: int,
    report: str,
    matrix: np.ndarray,
    sample_count: int,
) -> None:
    """Save the final Transformer test metrics."""

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_name": (
            "FinancialStressTransformer"
        ),
        "selected_run": checkpoint.get(
            "selected_run"
        ),
        "selection_metric": checkpoint.get(
            "selection_metric"
        ),
        "selection_tie_breaker": checkpoint.get(
            "selection_tie_breaker"
        ),
        "test_samples": sample_count,
        "transformer_metrics": transformer_metrics,
        "majority_baseline": {
            "source": (
                "training_set_majority_class"
            ),
            "class_id": majority_class,
            "class_name": CLASS_NAMES[
                majority_class
            ],
            "metrics": majority_metrics,
        },
        "classification_report": report,
        "confusion_matrix": (
            matrix.tolist()
        ),
        "hyperparameters": {
            "model_dimension": checkpoint[
                "model_dimension"
            ],
            "num_heads": checkpoint[
                "num_heads"
            ],
            "feed_forward_size": checkpoint[
                "feed_forward_size"
            ],
            "num_layers": checkpoint[
                "num_layers"
            ],
            "classifier_hidden_size": checkpoint[
                "classifier_hidden_size"
            ],
            "dropout": checkpoint[
                "dropout"
            ],
            "learning_rate": checkpoint[
                "learning_rate"
            ],
            "batch_size": checkpoint[
                "batch_size"
            ],
            "best_epoch": checkpoint[
                "best_epoch"
            ],
        },
    }

    METRICS_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    print(
        "Saved:",
        METRICS_PATH,
    )


def main() -> None:
    """Evaluate the tuned Transformer on the held-out test set."""

    prepared_data = prepare_ofr_data()

    _, _, test_loader = (
        create_data_loaders(
            prepared_data,
            batch_size=64,
        )
    )

    device = select_device()

    model, checkpoint = load_model(
        CHECKPOINT_PATH,
        device,
    )

    targets, predictions = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    majority_class = (
        determine_training_majority_class(
            prepared_data.y_train
        )
    )

    majority_predictions = (
        create_majority_baseline(
            targets=targets,
            majority_class=majority_class,
        )
    )

    transformer_metrics = (
        calculate_metrics(
            targets,
            predictions,
        )
    )

    majority_metrics = (
        calculate_metrics(
            targets,
            majority_predictions,
        )
    )

    report = classification_report(
        targets,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    print()
    print(
        "=== TRANSFORMER TEST EVALUATION ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Test samples:",
        len(
            targets
        ),
    )

    print(
        "Selected run:",
        checkpoint.get(
            "selected_run"
        ),
    )

    print()

    print(
        "Transformer:"
    )

    print(
        "Accuracy:",
        round(
            transformer_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Macro Precision:",
        round(
            transformer_metrics[
                "macro_precision"
            ],
            4,
        ),
    )

    print(
        "Macro Recall:",
        round(
            transformer_metrics[
                "macro_recall"
            ],
            4,
        ),
    )

    print(
        "Macro F1:",
        round(
            transformer_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print()

    print(
        "Training-set majority baseline:"
    )

    print(
        "Majority class:",
        majority_class,
        "-",
        CLASS_NAMES[
            majority_class
        ],
    )

    print(
        "Accuracy:",
        round(
            majority_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Macro Precision:",
        round(
            majority_metrics[
                "macro_precision"
            ],
            4,
        ),
    )

    print(
        "Macro Recall:",
        round(
            majority_metrics[
                "macro_recall"
            ],
            4,
        ),
    )

    print(
        "Macro F1:",
        round(
            majority_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print()

    print(
        "=== CLASSIFICATION REPORT ==="
    )

    print(
        report
    )

    print(
        "=== CONFUSION MATRIX ==="
    )

    print(
        matrix
    )

    save_confusion_matrix_plot(
        matrix
    )

    save_metrics(
        checkpoint=checkpoint,
        transformer_metrics=transformer_metrics,
        majority_metrics=majority_metrics,
        majority_class=majority_class,
        report=report,
        matrix=matrix,
        sample_count=len(
            targets
        ),
    )

    print()

    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()