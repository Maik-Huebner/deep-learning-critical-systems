"""Explainability and prediction-error analysis for the final Transformer.

The final tuned Transformer is not modified in this script.

The analysis uses the already frozen model to investigate:

- correct and incorrect predictions
- class-wise error behavior
- attention patterns for representative examples
- attention concentration over the 60-day historical window

Attention weights are treated as an interpretability aid only.
They must not be interpreted as causal explanations or direct
feature importance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from deep_learning_critical_systems.data.datasets import (
    create_data_loaders,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    CLASS_NAMES,
    WINDOW_SIZE,
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

LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "transformer_explainability.json"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

CORRECT_ATTENTION_PATH = (
    FIGURE_DIR
    / "transformer_attention_correct_increase.png"
)

INCORRECT_ATTENTION_PATH = (
    FIGURE_DIR
    / "transformer_attention_misclassified_increase.png"
)

ERROR_DISTRIBUTION_PATH = (
    FIGURE_DIR
    / "transformer_correct_vs_incorrect_by_class.png"
)


def load_model(
    device: torch.device,
) -> tuple[
    FinancialStressTransformer,
    dict,
]:
    """Load the frozen tuned Transformer."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model = FinancialStressTransformer(
        feature_count=checkpoint[
            "feature_count"
        ],
        model_dimension=checkpoint[
            "model_dimension"
        ],
        num_heads=checkpoint[
            "num_heads"
        ],
        feed_forward_size=checkpoint[
            "feed_forward_size"
        ],
        num_layers=checkpoint[
            "num_layers"
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


def collect_predictions(
    model: FinancialStressTransformer,
    test_loader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Collect targets, predictions and confidence scores."""

    targets = []
    predictions = []
    confidences = []

    with torch.no_grad():
        for features, batch_targets in test_loader:
            features = features.to(
                device
            )

            logits = model(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            batch_confidence, batch_predictions = (
                probabilities.max(
                    dim=1
                )
            )

            targets.extend(
                batch_targets
                .numpy()
                .tolist()
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
                .tolist()
            )

            confidences.extend(
                batch_confidence
                .cpu()
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
        np.asarray(
            confidences,
            dtype=np.float64,
        ),
    )


def select_representative_examples(
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
) -> tuple[int, int]:
    """Select one correct and one failed Stress Increase example.

    The correct example is the correctly predicted Stress Increase
    sample with the highest model confidence.

    The failed example is the misclassified true Stress Increase
    sample with the highest confidence in its wrong prediction.

    This makes the selection deterministic and easy to reproduce.
    """

    increase_class = 2

    correct_mask = (
        (targets == increase_class)
        & (predictions == increase_class)
    )

    incorrect_mask = (
        (targets == increase_class)
        & (predictions != increase_class)
    )

    correct_indices = np.flatnonzero(
        correct_mask
    )

    incorrect_indices = np.flatnonzero(
        incorrect_mask
    )

    if len(correct_indices) == 0:
        raise RuntimeError(
            "No correctly predicted Stress Increase sample exists."
        )

    if len(incorrect_indices) == 0:
        raise RuntimeError(
            "No misclassified Stress Increase sample exists."
        )

    correct_index = int(
        correct_indices[
            np.argmax(
                confidences[
                    correct_indices
                ]
            )
        ]
    )

    incorrect_index = int(
        incorrect_indices[
            np.argmax(
                confidences[
                    incorrect_indices
                ]
            )
        ]
    )

    return (
        correct_index,
        incorrect_index,
    )


def extract_attention(
    model: FinancialStressTransformer,
    sample: np.ndarray,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract final-layer attention for one input window.

    Returns:

    - class probabilities
    - attention matrix averaged across heads
    - mean attention received by each historical timestep
    """

    tensor = torch.tensor(
        sample,
        dtype=torch.float32,
    ).unsqueeze(
        0
    ).to(
        device
    )

    with torch.no_grad():
        logits, attention_maps = (
            model.forward_with_attention(
                tensor
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    if not attention_maps:
        raise RuntimeError(
            "Transformer returned no attention maps."
        )

    final_attention = (
        attention_maps[-1]
        .detach()
        .cpu()
        .squeeze(0)
    )

    if final_attention.ndim != 3:
        raise RuntimeError(
            "Unexpected attention map shape."
        )

    mean_attention_matrix = (
        final_attention
        .mean(
            dim=0
        )
        .numpy()
    )

    attention_received = (
        final_attention
        .mean(
            dim=(
                0,
                1,
            )
        )
        .numpy()
    )

    return (
        probabilities
        .cpu()
        .numpy()[0],
        mean_attention_matrix,
        attention_received,
    )


def relative_day_labels() -> np.ndarray:
    """Return relative positions for the 60-day historical window."""

    return np.arange(
        -(WINDOW_SIZE - 1),
        1,
    )


def top_attention_days(
    attention_received: np.ndarray,
    count: int = 5,
) -> list[dict]:
    """Return the historical positions receiving most attention."""

    relative_days = (
        relative_day_labels()
    )

    indices = np.argsort(
        attention_received
    )[
        ::-1
    ][
        :count
    ]

    return [
        {
            "window_position": int(
                index
            ),
            "relative_trading_day": int(
                relative_days[
                    index
                ]
            ),
            "mean_attention": float(
                attention_received[
                    index
                ]
            ),
        }
        for index in indices
    ]


def save_attention_figure(
    matrix: np.ndarray,
    attention_received: np.ndarray,
    actual_class: int,
    predicted_class: int,
    confidence: float,
    sample_index: int,
    path: Path,
    example_name: str,
) -> None:
    """Create one attention heatmap with temporal attention summary."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_days = (
        relative_day_labels()
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            10,
            10,
        ),
        height_ratios=[
            3,
            1,
        ],
    )

    heatmap_axis = axes[
        0
    ]

    line_axis = axes[
        1
    ]

    image = heatmap_axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )

    figure.colorbar(
        image,
        ax=heatmap_axis,
        label="Mean Attention Weight",
    )

    tick_positions = [
        0,
        9,
        19,
        29,
        39,
        49,
        59,
    ]

    tick_labels = [
        str(
            relative_days[
                position
            ]
        )
        for position in tick_positions
    ]

    heatmap_axis.set_xticks(
        tick_positions
    )

    heatmap_axis.set_xticklabels(
        tick_labels
    )

    heatmap_axis.set_yticks(
        tick_positions
    )

    heatmap_axis.set_yticklabels(
        tick_labels
    )

    heatmap_axis.set_xlabel(
        "Key Position: Relative Trading Day"
    )

    heatmap_axis.set_ylabel(
        "Query Position: Relative Trading Day"
    )

    heatmap_axis.set_title(
        (
            f"{example_name} Transformer Prediction\n"
            f"True: {CLASS_NAMES[actual_class]} | "
            f"Predicted: {CLASS_NAMES[predicted_class]} | "
            f"Confidence: {confidence * 100:.1f}% | "
            f"Test Index: {sample_index}"
        )
    )

    line_axis.plot(
        relative_days,
        attention_received,
    )

    line_axis.set_xlabel(
        "Relative Trading Day"
    )

    line_axis.set_ylabel(
        "Mean Attention"
    )

    line_axis.set_title(
        "Mean Attention Received Across Heads and Query Positions"
    )

    line_axis.grid(
        alpha=0.25
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


def save_error_distribution(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Plot correct and incorrect predictions by true class."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    correct_counts = []

    incorrect_counts = []

    class_results = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        class_mask = (
            targets == class_id
        )

        correct_count = int(
            np.sum(
                class_mask
                & (
                    predictions
                    == class_id
                )
            )
        )

        total_count = int(
            np.sum(
                class_mask
            )
        )

        incorrect_count = (
            total_count
            - correct_count
        )

        correct_counts.append(
            correct_count
        )

        incorrect_counts.append(
            incorrect_count
        )

        class_results[
            class_name
        ] = {
            "total": total_count,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "correct_rate": (
                correct_count
                / total_count
            ),
        }

    positions = np.arange(
        len(
            CLASS_NAMES
        )
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    correct_bars = axis.bar(
        positions
        - width
        / 2,
        correct_counts,
        width=width,
        label="Correct",
    )

    incorrect_bars = axis.bar(
        positions
        + width
        / 2,
        incorrect_counts,
        width=width,
        label="Incorrect",
    )

    axis.set_title(
        "Transformer Correct and Incorrect Predictions by True Class"
    )

    axis.set_xlabel(
        "True Stress Class"
    )

    axis.set_ylabel(
        "Number of Test Samples"
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        CLASS_NAMES
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bars in (
        correct_bars,
        incorrect_bars,
    ):
        for bar in bars:
            value = int(
                bar.get_height()
            )

            axis.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value
                + 5,
                str(
                    value
                ),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.tight_layout()

    figure.savefig(
        ERROR_DISTRIBUTION_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved:",
        ERROR_DISTRIBUTION_PATH,
    )

    return class_results


def confusion_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Return the confusion matrix and most common error transitions."""

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    errors = []

    for actual_class in range(
        len(
            CLASS_NAMES
        )
    ):
        for predicted_class in range(
            len(
                CLASS_NAMES
            )
        ):
            if (
                actual_class
                == predicted_class
            ):
                continue

            errors.append(
                {
                    "actual_class": CLASS_NAMES[
                        actual_class
                    ],
                    "predicted_class": CLASS_NAMES[
                        predicted_class
                    ],
                    "count": int(
                        matrix[
                            actual_class,
                            predicted_class,
                        ]
                    ),
                }
            )

    errors.sort(
        key=lambda item: item[
            "count"
        ],
        reverse=True,
    )

    return {
        "matrix": matrix.tolist(),
        "most_common_errors": errors,
    }


def save_explainability_log(
    checkpoint: dict,
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    correct_index: int,
    incorrect_index: int,
    correct_probabilities: np.ndarray,
    incorrect_probabilities: np.ndarray,
    correct_attention: np.ndarray,
    incorrect_attention: np.ndarray,
    class_results: dict,
    confusion_results: dict,
) -> None:
    """Save explainability and error-analysis results."""

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": (
            "FinancialStressTransformer"
        ),
        "selected_run": checkpoint.get(
            "selected_run"
        ),
        "analysis_split": (
            "held_out_test_set"
        ),
        "test_samples": int(
            len(
                targets
            )
        ),
        "interpretation_note": (
            "Attention weights are used as an interpretability aid. "
            "They do not establish causality and must not be treated "
            "as direct feature importance."
        ),
        "correct_stress_increase_example": {
            "test_index": correct_index,
            "actual_class_id": int(
                targets[
                    correct_index
                ]
            ),
            "actual_class": CLASS_NAMES[
                targets[
                    correct_index
                ]
            ],
            "predicted_class_id": int(
                predictions[
                    correct_index
                ]
            ),
            "predicted_class": CLASS_NAMES[
                predictions[
                    correct_index
                ]
            ],
            "prediction_confidence": float(
                confidences[
                    correct_index
                ]
            ),
            "class_probabilities": {
                class_name: float(
                    correct_probabilities[
                        class_id
                    ]
                )
                for class_id, class_name
                in enumerate(
                    CLASS_NAMES
                )
            },
            "top_attention_days": (
                top_attention_days(
                    correct_attention
                )
            ),
        },
        "misclassified_stress_increase_example": {
            "test_index": incorrect_index,
            "actual_class_id": int(
                targets[
                    incorrect_index
                ]
            ),
            "actual_class": CLASS_NAMES[
                targets[
                    incorrect_index
                ]
            ],
            "predicted_class_id": int(
                predictions[
                    incorrect_index
                ]
            ),
            "predicted_class": CLASS_NAMES[
                predictions[
                    incorrect_index
                ]
            ],
            "prediction_confidence": float(
                confidences[
                    incorrect_index
                ]
            ),
            "class_probabilities": {
                class_name: float(
                    incorrect_probabilities[
                        class_id
                    ]
                )
                for class_id, class_name
                in enumerate(
                    CLASS_NAMES
                )
            },
            "top_attention_days": (
                top_attention_days(
                    incorrect_attention
                )
            ),
        },
        "class_results": class_results,
        "confusion_analysis": confusion_results,
    }

    LOG_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    print(
        "Saved:",
        LOG_PATH,
    )


def print_example(
    title: str,
    index: int,
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray,
    attention_received: np.ndarray,
) -> None:
    """Print one representative prediction example."""

    print()
    print(
        title
    )

    print(
        "Test index:",
        index,
    )

    print(
        "True class:",
        CLASS_NAMES[
            targets[
                index
            ]
        ],
    )

    print(
        "Predicted class:",
        CLASS_NAMES[
            predictions[
                index
            ]
        ],
    )

    print(
        "Confidence:",
        round(
            confidences[
                index
            ],
            4,
        ),
    )

    print(
        "Class probabilities:"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name}: "
            f"{probabilities[class_id]:.4f}"
        )

    print(
        "Top attention positions:"
    )

    for item in top_attention_days(
        attention_received
    ):
        print(
            "  Relative trading day "
            f"{item['relative_trading_day']:>3}: "
            f"{item['mean_attention']:.6f}"
        )


def main() -> None:
    """Run the final Transformer explainability analysis."""

    prepared_data = prepare_ofr_data()

    _, _, test_loader = (
        create_data_loaders(
            prepared_data,
            batch_size=64,
        )
    )

    device = select_device()

    model, checkpoint = load_model(
        device
    )

    (
        targets,
        predictions,
        confidences,
    ) = collect_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    (
        correct_index,
        incorrect_index,
    ) = select_representative_examples(
        targets=targets,
        predictions=predictions,
        confidences=confidences,
    )

    (
        correct_probabilities,
        correct_matrix,
        correct_attention,
    ) = extract_attention(
        model=model,
        sample=prepared_data.X_test[
            correct_index
        ],
        device=device,
    )

    (
        incorrect_probabilities,
        incorrect_matrix,
        incorrect_attention,
    ) = extract_attention(
        model=model,
        sample=prepared_data.X_test[
            incorrect_index
        ],
        device=device,
    )

    print()
    print(
        "=== TRANSFORMER EXPLAINABILITY ANALYSIS ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Selected run:",
        checkpoint.get(
            "selected_run"
        ),
    )

    print(
        "Test samples:",
        len(
            targets
        ),
    )

    print(
        "Attention interpretation:"
        " descriptive, not causal"
    )

    print_example(
        title=(
            "=== CORRECT STRESS INCREASE EXAMPLE ==="
        ),
        index=correct_index,
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        probabilities=correct_probabilities,
        attention_received=correct_attention,
    )

    print_example(
        title=(
            "=== MISCLASSIFIED STRESS INCREASE EXAMPLE ==="
        ),
        index=incorrect_index,
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        probabilities=incorrect_probabilities,
        attention_received=incorrect_attention,
    )

    save_attention_figure(
        matrix=correct_matrix,
        attention_received=correct_attention,
        actual_class=int(
            targets[
                correct_index
            ]
        ),
        predicted_class=int(
            predictions[
                correct_index
            ]
        ),
        confidence=float(
            confidences[
                correct_index
            ]
        ),
        sample_index=correct_index,
        path=CORRECT_ATTENTION_PATH,
        example_name=(
            "Correct Stress Increase"
        ),
    )

    save_attention_figure(
        matrix=incorrect_matrix,
        attention_received=incorrect_attention,
        actual_class=int(
            targets[
                incorrect_index
            ]
        ),
        predicted_class=int(
            predictions[
                incorrect_index
            ]
        ),
        confidence=float(
            confidences[
                incorrect_index
            ]
        ),
        sample_index=incorrect_index,
        path=INCORRECT_ATTENTION_PATH,
        example_name=(
            "Misclassified Stress Increase"
        ),
    )

    class_results = (
        save_error_distribution(
            targets=targets,
            predictions=predictions,
        )
    )

    confusion_results = (
        confusion_summary(
            targets=targets,
            predictions=predictions,
        )
    )

    save_explainability_log(
        checkpoint=checkpoint,
        targets=targets,
        predictions=predictions,
        confidences=confidences,
        correct_index=correct_index,
        incorrect_index=incorrect_index,
        correct_probabilities=correct_probabilities,
        incorrect_probabilities=incorrect_probabilities,
        correct_attention=correct_attention,
        incorrect_attention=incorrect_attention,
        class_results=class_results,
        confusion_results=confusion_results,
    )

    print()
    print(
        "=== CLASS-WISE CORRECT RATES ==="
    )

    for class_name, result in (
        class_results.items()
    ):
        print(
            f"{class_name}: "
            f"{result['correct']} / "
            f"{result['total']} "
            f"({result['correct_rate'] * 100:.2f}%)"
        )

    print()
    print(
        "=== MOST COMMON MISCLASSIFICATIONS ==="
    )

    for error in (
        confusion_results[
            "most_common_errors"
        ]
    ):
        print(
            f"{error['actual_class']} -> "
            f"{error['predicted_class']}: "
            f"{error['count']}"
        )

    print()
    print(
        "Explainability analysis completed."
    )


if __name__ == "__main__":
    main()