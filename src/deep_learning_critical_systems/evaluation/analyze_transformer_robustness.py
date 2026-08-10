"""Robustness analysis for the final frozen Transformer.

The model is not retrained or tuned in this script.

The test investigates how sensitive the final Transformer is to
controlled random noise added to the already standardized input
features.

Because the input features were standardized using training data,
a noise standard deviation of 0.10 corresponds to a perturbation
with a standard deviation equal to 10 percent of one standardized
feature unit.

This is a stress test of model stability. It is not intended to
represent one exact real-world market disturbance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
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

LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "transformer_robustness.json"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "transformer_robustness_noise.png"
)


RANDOM_SEED = 42

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.50,
]


def load_model(
    device: torch.device,
) -> tuple[
    FinancialStressTransformer,
    dict,
]:
    """Load the final tuned Transformer."""

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


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate multi-class evaluation metrics."""

    class_recalls = recall_score(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )

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
        "recall_stress_decrease": float(
            class_recalls[
                0
            ]
        ),
        "recall_stable": float(
            class_recalls[
                1
            ]
        ),
        "recall_stress_increase": float(
            class_recalls[
                2
            ]
        ),
    }


def collect_predictions(
    model: FinancialStressTransformer,
    test_loader,
    device: torch.device,
    noise_standard_deviation: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Collect predictions with one controlled noise level."""

    if noise_standard_deviation < 0:
        raise ValueError(
            "Noise standard deviation must not be negative."
        )

    torch.manual_seed(
        RANDOM_SEED
        + int(
            noise_standard_deviation
            * 1000
        )
    )

    targets = []

    predictions = []

    with torch.no_grad():
        for features, batch_targets in test_loader:
            if noise_standard_deviation > 0:
                noise = (
                    torch.randn_like(
                        features
                    )
                    * noise_standard_deviation
                )

                features = (
                    features
                    + noise
                )

            features = features.to(
                device
            )

            logits = model(
                features
            )

            batch_predictions = (
                logits.argmax(
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


def prediction_agreement(
    reference_predictions: np.ndarray,
    noisy_predictions: np.ndarray,
) -> float:
    """Calculate agreement with clean model predictions."""

    if (
        reference_predictions.shape
        != noisy_predictions.shape
    ):
        raise ValueError(
            "Prediction arrays must have the same shape."
        )

    return float(
        np.mean(
            reference_predictions
            == noisy_predictions
        )
    )


def run_robustness_analysis(
    model: FinancialStressTransformer,
    test_loader,
    device: torch.device,
) -> list[dict]:
    """Evaluate the frozen model under increasing noise."""

    clean_targets, clean_predictions = (
        collect_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            noise_standard_deviation=0.0,
        )
    )

    results = []

    for noise_level in NOISE_LEVELS:
        if noise_level == 0.0:
            targets = clean_targets

            predictions = (
                clean_predictions
            )

        else:
            targets, predictions = (
                collect_predictions(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    noise_standard_deviation=noise_level,
                )
            )

        metrics = calculate_metrics(
            targets,
            predictions,
        )

        agreement = prediction_agreement(
            clean_predictions,
            predictions,
        )

        result = {
            "noise_standard_deviation": (
                noise_level
            ),
            "noise_percentage_of_standardized_unit": (
                noise_level
                * 100
            ),
            **metrics,
            "prediction_agreement_with_clean": (
                agreement
            ),
        }

        results.append(
            result
        )

    return results


def save_plot(
    results: list[dict],
) -> None:
    """Plot robustness metrics across noise levels."""

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    noise_percentages = [
        result[
            "noise_percentage_of_standardized_unit"
        ]
        for result in results
    ]

    accuracy_values = [
        result[
            "accuracy"
        ]
        * 100
        for result in results
    ]

    macro_f1_values = [
        result[
            "macro_f1"
        ]
        * 100
        for result in results
    ]

    increase_recall_values = [
        result[
            "recall_stress_increase"
        ]
        * 100
        for result in results
    ]

    agreement_values = [
        result[
            "prediction_agreement_with_clean"
        ]
        * 100
        for result in results
    ]

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.plot(
        noise_percentages,
        accuracy_values,
        marker="o",
        label="Accuracy",
    )

    axis.plot(
        noise_percentages,
        macro_f1_values,
        marker="o",
        label="Macro-F1",
    )

    axis.plot(
        noise_percentages,
        increase_recall_values,
        marker="o",
        label="Stress Increase Recall",
    )

    axis.plot(
        noise_percentages,
        agreement_values,
        marker="o",
        label="Agreement with Clean Predictions",
    )

    axis.set_title(
        "Transformer Robustness Under Standardized Input Noise"
    )

    axis.set_xlabel(
        "Noise Standard Deviation (% of Standardized Unit)"
    )

    axis.set_ylabel(
        "Score (%)"
    )

    axis.set_xticks(
        noise_percentages
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

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


def save_results(
    checkpoint: dict,
    results: list[dict],
) -> None:
    """Save all robustness results as JSON."""

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
        "random_seed": RANDOM_SEED,
        "method": (
            "Gaussian noise added to standardized input features"
        ),
        "interpretation": (
            "This analysis is a controlled robustness stress test. "
            "It does not model one specific real-world market shock "
            "and is not used for further model tuning."
        ),
        "results": results,
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


def print_results(
    results: list[dict],
) -> None:
    """Print the robustness results."""

    print()
    print(
        "=== TRANSFORMER ROBUSTNESS RESULTS ==="
    )

    print()

    print(
        "Noise | Accuracy | Macro-F1 | "
        "Increase Recall | Clean Agreement"
    )

    print(
        "-" * 70
    )

    for result in results:
        print(
            f"{result['noise_percentage_of_standardized_unit']:>5.0f}% | "
            f"{result['accuracy'] * 100:>7.2f}% | "
            f"{result['macro_f1'] * 100:>7.2f}% | "
            f"{result['recall_stress_increase'] * 100:>14.2f}% | "
            f"{result['prediction_agreement_with_clean'] * 100:>14.2f}%"
        )


def main() -> None:
    """Run robustness analysis for the final Transformer."""

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

    print()
    print(
        "=== TRANSFORMER ROBUSTNESS ANALYSIS ==="
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
            prepared_data.y_test
        ),
    )

    print(
        "Model retraining:"
        " NO"
    )

    print(
        "Hyperparameter tuning:"
        " NO"
    )

    print(
        "Robustness method:"
        " standardized input noise"
    )

    results = (
        run_robustness_analysis(
            model=model,
            test_loader=test_loader,
            device=device,
        )
    )

    print_results(
        results
    )

    save_plot(
        results
    )

    save_results(
        checkpoint=checkpoint,
        results=results,
    )

    print()
    print(
        "Robustness analysis completed."
    )


if __name__ == "__main__":
    main()