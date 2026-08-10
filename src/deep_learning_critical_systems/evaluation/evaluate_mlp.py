"""Evaluate the trained MLP financial-stress baseline."""

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
from deep_learning_critical_systems.models.mlp import (
    FinancialStressMLP,
)
from deep_learning_critical_systems.training.trainer import (
    select_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "mlp_baseline.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "mlp_test_metrics.json"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "mlp_confusion_matrix.png"
)

BATCH_SIZE = 64


def load_model(
    device: torch.device,
) -> FinancialStressMLP:
    """Load the best saved MLP model."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"MLP checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model = FinancialStressMLP(
        sequence_length=checkpoint[
            "sequence_length"
        ],
        feature_count=checkpoint[
            "feature_count"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        second_hidden_size=checkpoint[
            "second_hidden_size"
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

    return model


def predict(
    model: FinancialStressMLP,
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
    """Create and save the MLP test confusion matrix."""

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
        "MLP Baseline - Test Confusion Matrix"
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
    mlp_metrics: dict[str, float],
    majority_metrics: dict[str, float],
) -> None:
    """Save MLP and naive-baseline metrics as JSON."""

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "mlp": mlp_metrics,
        "majority_baseline": majority_metrics,
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
    """Run the complete MLP test evaluation."""

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

    model = load_model(
        device
    )

    (
        test_targets,
        mlp_predictions,
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

    mlp_metrics = calculate_metrics(
        test_targets,
        mlp_predictions,
    )

    majority_metrics = calculate_metrics(
        test_targets,
        majority_predictions,
    )

    print()
    print(
        "=== MLP TEST EVALUATION ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Test samples:",
        len(
            test_targets
        ),
    )

    print_metric_block(
        "MLP baseline:",
        mlp_metrics,
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
            mlp_predictions,
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
        mlp_predictions,
    )

    print(
        matrix
    )

    print()

    plot_confusion_matrix(
        test_targets,
        mlp_predictions,
    )

    save_metrics(
        mlp_metrics,
        majority_metrics,
    )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()