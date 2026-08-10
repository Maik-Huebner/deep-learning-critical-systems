"""Validation-based hyperparameter tuning for the Transformer model.

The test set is deliberately not used during hyperparameter tuning.

Within each experiment, early stopping selects the model state with
the best validation loss.

Across experiments, configurations are ranked primarily by validation
Macro-F1 and secondarily by validation loss.

The complete tuning history is saved for later documentation,
visualization and presentation.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

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
    FEATURE_COLUMNS,
    FORECAST_HORIZON,
    WINDOW_SIZE,
    prepare_ofr_data,
)
from deep_learning_critical_systems.models.transformer import (
    DEFAULT_CLASSIFIER_HIDDEN_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_FEED_FORWARD_SIZE,
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    FinancialStressTransformer,
)
from deep_learning_critical_systems.training.trainer import (
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    select_device,
    set_seed,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
)

LOG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
)

BASELINE_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "transformer_model.pt"
)

BASELINE_HISTORY_PATH = (
    LOG_DIR
    / "transformer_training_history.json"
)

TUNED_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "transformer_tuned_model.pt"
)

TUNED_HISTORY_PATH = (
    LOG_DIR
    / "transformer_tuned_training_history.json"
)

TUNING_RESULTS_PATH = (
    LOG_DIR
    / "transformer_tuning_results.json"
)

TUNING_RESULTS_CSV_PATH = (
    LOG_DIR
    / "transformer_tuning_results.csv"
)

TUNING_HISTORIES_PATH = (
    LOG_DIR
    / "transformer_tuning_histories.json"
)


RANDOM_SEED = 42
BATCH_SIZE = 64


BASE_CONFIGURATION = {
    "model_dimension": DEFAULT_MODEL_DIMENSION,
    "num_heads": DEFAULT_NUM_HEADS,
    "feed_forward_size": DEFAULT_FEED_FORWARD_SIZE,
    "num_layers": DEFAULT_NUM_LAYERS,
    "classifier_hidden_size": DEFAULT_CLASSIFIER_HIDDEN_SIZE,
    "dropout": DEFAULT_DROPOUT,
    "learning_rate": 0.001,
}


TUNING_CONFIGURATIONS = [
    {
        "run_id": "T1",
        "description": "Lower learning rate",
        **BASE_CONFIGURATION,
        "learning_rate": 0.0005,
    },
    {
        "run_id": "T2",
        "description": "Larger model dimension",
        **BASE_CONFIGURATION,
        "model_dimension": 128,
    },
    {
        "run_id": "T3",
        "description": "Fewer attention heads",
        **BASE_CONFIGURATION,
        "num_heads": 2,
    },
    {
        "run_id": "T4",
        "description": "Single encoder layer",
        **BASE_CONFIGURATION,
        "num_layers": 1,
    },
    {
        "run_id": "T5",
        "description": "Larger feed-forward network",
        **BASE_CONFIGURATION,
        "feed_forward_size": 256,
    },
    {
        "run_id": "T6",
        "description": "Lower dropout",
        **BASE_CONFIGURATION,
        "dropout": 0.10,
    },
]


def history_to_dict(
    history,
) -> dict:
    """Convert a TrainingHistory object into serializable data."""

    return {
        "train_loss": history.train_loss,
        "validation_loss": history.validation_loss,
        "train_accuracy": history.train_accuracy,
        "validation_accuracy": history.validation_accuracy,
        "best_epoch": history.best_epoch,
        "epochs_trained": history.epochs_trained,
    }


def load_json(
    path: Path,
) -> dict:
    """Load one JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return json.loads(
        path.read_text()
    )


def create_model(
    configuration: dict,
) -> FinancialStressTransformer:
    """Create one Transformer from a tuning configuration."""

    return FinancialStressTransformer(
        feature_count=len(
            FEATURE_COLUMNS
        ),
        model_dimension=configuration[
            "model_dimension"
        ],
        num_heads=configuration[
            "num_heads"
        ],
        feed_forward_size=configuration[
            "feed_forward_size"
        ],
        num_layers=configuration[
            "num_layers"
        ],
        classifier_hidden_size=configuration[
            "classifier_hidden_size"
        ],
        class_count=3,
        dropout=configuration[
            "dropout"
        ],
    )


def evaluate_validation(
    model: FinancialStressTransformer,
    validation_loader,
    device: torch.device,
) -> dict[str, float]:
    """Calculate validation metrics without using the test set."""

    model.eval()

    targets = []
    predictions = []

    with torch.no_grad():
        for features, batch_targets in validation_loader:
            features = features.to(
                device
            )

            logits = model(
                features
            )

            batch_predictions = logits.argmax(
                dim=1
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

    targets_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions_array = np.asarray(
        predictions,
        dtype=np.int64,
    )

    class_recalls = recall_score(
        targets_array,
        predictions_array,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )

    return {
        "validation_accuracy": float(
            accuracy_score(
                targets_array,
                predictions_array,
            )
        ),
        "validation_macro_precision": float(
            precision_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "validation_macro_recall": float(
            recall_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "validation_macro_f1": float(
            f1_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_stress_decrease": float(
            class_recalls[0]
        ),
        "recall_stable": float(
            class_recalls[1]
        ),
        "recall_stress_increase": float(
            class_recalls[2]
        ),
    }


def load_baseline_model(
    device: torch.device,
) -> tuple[
    FinancialStressTransformer,
    dict,
    dict,
]:
    """Load the already trained T0 Transformer."""

    if not BASELINE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Baseline Transformer checkpoint not found: "
            f"{BASELINE_CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT_PATH,
        map_location="cpu",
    )

    configuration = {
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
    }

    model = create_model(
        configuration
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    history = load_json(
        BASELINE_HISTORY_PATH
    )

    return (
        model,
        configuration,
        history,
    )


def build_result(
    run_id: str,
    description: str,
    configuration: dict,
    model: FinancialStressTransformer,
    history: dict,
    validation_metrics: dict[str, float],
) -> dict:
    """Build the complete documentation record for one run."""

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    best_validation_loss = min(
        history[
            "validation_loss"
        ]
    )

    return {
        "run_id": run_id,
        "description": description,

        "model_dimension": configuration[
            "model_dimension"
        ],
        "num_heads": configuration[
            "num_heads"
        ],
        "feed_forward_size": configuration[
            "feed_forward_size"
        ],
        "num_layers": configuration[
            "num_layers"
        ],
        "classifier_hidden_size": configuration[
            "classifier_hidden_size"
        ],
        "dropout": configuration[
            "dropout"
        ],
        "learning_rate": configuration[
            "learning_rate"
        ],
        "batch_size": BATCH_SIZE,

        "parameter_count": parameter_count,

        "best_epoch": history[
            "best_epoch"
        ],
        "epochs_trained": history[
            "epochs_trained"
        ],

        "validation_loss": float(
            best_validation_loss
        ),

        **validation_metrics,
    }


def print_result(
    result: dict,
) -> None:
    """Print one tuning result in a compact form."""

    print()
    print(
        f"{result['run_id']} - "
        f"{result['description']}"
    )

    print(
        "Best epoch:",
        result[
            "best_epoch"
        ],
    )

    print(
        "Validation loss:",
        round(
            result[
                "validation_loss"
            ],
            4,
        ),
    )

    print(
        "Validation accuracy:",
        round(
            result[
                "validation_accuracy"
            ],
            4,
        ),
    )

    print(
        "Validation Macro-F1:",
        round(
            result[
                "validation_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Stress Increase recall:",
        round(
            result[
                "recall_stress_increase"
            ],
            4,
        ),
    )


def save_results(
    results: list[dict],
    histories: dict,
) -> None:
    """Save all tuning results and histories."""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TUNING_RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    TUNING_HISTORIES_PATH.write_text(
        json.dumps(
            histories,
            indent=2,
        )
        + "\n"
    )

    fieldnames = list(
        results[0].keys()
    )

    with TUNING_RESULTS_CSV_PATH.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        "Saved tuning results:"
    )

    print(
        TUNING_RESULTS_PATH
    )

    print(
        TUNING_RESULTS_CSV_PATH
    )

    print(
        TUNING_HISTORIES_PATH
    )


def save_best_model(
    best_result: dict,
    best_state_dict: dict,
    best_history: dict,
    prepared_data,
) -> None:
    """Save the selected validation-best Transformer."""

    checkpoint = {
        "model_state_dict": best_state_dict,
        "model_name": "FinancialStressTransformer",
        "selected_run": best_result[
            "run_id"
        ],
        "selection_metric": "validation_macro_f1",
        "sequence_length": WINDOW_SIZE,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "model_dimension": best_result[
            "model_dimension"
        ],
        "num_heads": best_result[
            "num_heads"
        ],
        "feed_forward_size": best_result[
            "feed_forward_size"
        ],
        "num_layers": best_result[
            "num_layers"
        ],
        "classifier_hidden_size": best_result[
            "classifier_hidden_size"
        ],
        "class_count": 3,
        "dropout": best_result[
            "dropout"
        ],
        "forecast_horizon": FORECAST_HORIZON,
        "low_threshold": prepared_data.low_threshold,
        "high_threshold": prepared_data.high_threshold,
        "feature_names": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": best_result[
            "learning_rate"
        ],
        "maximum_epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "best_epoch": best_result[
            "best_epoch"
        ],
        "validation_loss": best_result[
            "validation_loss"
        ],
        "validation_macro_f1": best_result[
            "validation_macro_f1"
        ],
    }

    torch.save(
        checkpoint,
        TUNED_CHECKPOINT_PATH,
    )

    TUNED_HISTORY_PATH.write_text(
        json.dumps(
            best_history,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved selected Transformer:"
    )

    print(
        TUNED_CHECKPOINT_PATH
    )

    print(
        TUNED_HISTORY_PATH
    )


def main() -> None:
    """Run the controlled Transformer tuning experiment."""

    set_seed(
        RANDOM_SEED
    )

    prepared_data = prepare_ofr_data()

    (
        train_loader,
        validation_loader,
        _,
    ) = create_data_loaders(
        prepared_data,
        batch_size=BATCH_SIZE,
    )

    device = select_device()

    results = []
    histories = {}

    state_dicts = {}

    print()
    print(
        "=== TRANSFORMER HYPERPARAMETER TUNING ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Selection metric: Validation Macro-F1"
    )

    print(
        "Tie-breaker: Validation Loss"
    )

    print(
        "Test set used: NO"
    )

    print()
    print(
        "Loading T0 baseline..."
    )

    (
        baseline_model,
        baseline_configuration,
        baseline_history,
    ) = load_baseline_model(
        device
    )

    baseline_metrics = evaluate_validation(
        baseline_model,
        validation_loader,
        device,
    )

    baseline_result = build_result(
        run_id="T0",
        description="Untuned baseline",
        configuration=baseline_configuration,
        model=baseline_model,
        history=baseline_history,
        validation_metrics=baseline_metrics,
    )

    results.append(
        baseline_result
    )

    histories[
        "T0"
    ] = baseline_history

    state_dicts[
        "T0"
    ] = {
        name: parameter.detach().cpu().clone()
        for name, parameter in baseline_model.state_dict().items()
    }

    print_result(
        baseline_result
    )

    for configuration in TUNING_CONFIGURATIONS:
        run_id = configuration[
            "run_id"
        ]

        description = configuration[
            "description"
        ]

        print()
        print(
            "=" * 70
        )

        print(
            f"Starting {run_id}: {description}"
        )

        print(
            "=" * 70
        )

        # Reset the seed before every run so that comparisons are as
        # reproducible and fair as possible.
        set_seed(
            RANDOM_SEED
        )

        model = create_model(
            configuration
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            epochs=DEFAULT_EPOCHS,
            learning_rate=configuration[
                "learning_rate"
            ],
            patience=DEFAULT_PATIENCE,
        )

        history_data = history_to_dict(
            history
        )

        validation_metrics = evaluate_validation(
            model,
            validation_loader,
            device,
        )

        result = build_result(
            run_id=run_id,
            description=description,
            configuration=configuration,
            model=model,
            history=history_data,
            validation_metrics=validation_metrics,
        )

        results.append(
            result
        )

        histories[
            run_id
        ] = history_data

        state_dicts[
            run_id
        ] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.state_dict().items()
        }

        print_result(
            result
        )

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

    best_result = ranked_results[
        0
    ]

    print()
    print(
        "=" * 70
    )

    print(
        "=== TUNING RANKING ==="
    )

    print()

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['run_id']} | "
            f"Macro-F1: "
            f"{result['validation_macro_f1']:.4f} | "
            f"Val Loss: "
            f"{result['validation_loss']:.4f} | "
            f"Val Acc: "
            f"{result['validation_accuracy']:.4f} | "
            f"Increase Recall: "
            f"{result['recall_stress_increase']:.4f}"
        )

    save_results(
        results,
        histories,
    )

    save_best_model(
        best_result=best_result,
        best_state_dict=deepcopy(
            state_dicts[
                best_result[
                    "run_id"
                ]
            ]
        ),
        best_history=histories[
            best_result[
                "run_id"
            ]
        ],
        prepared_data=prepared_data,
    )

    print()
    print(
        "=== SELECTED CONFIGURATION ==="
    )

    print(
        "Run:",
        best_result[
            "run_id"
        ],
    )

    print(
        "Description:",
        best_result[
            "description"
        ],
    )

    print(
        "Model dimension:",
        best_result[
            "model_dimension"
        ],
    )

    print(
        "Attention heads:",
        best_result[
            "num_heads"
        ],
    )

    print(
        "Encoder layers:",
        best_result[
            "num_layers"
        ],
    )

    print(
        "Feed-forward size:",
        best_result[
            "feed_forward_size"
        ],
    )

    print(
        "Dropout:",
        best_result[
            "dropout"
        ],
    )

    print(
        "Learning rate:",
        best_result[
            "learning_rate"
        ],
    )

    print(
        "Validation Macro-F1:",
        round(
            best_result[
                "validation_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Validation loss:",
        round(
            best_result[
                "validation_loss"
            ],
            4,
        ),
    )

    print()
    print(
        "Transformer tuning completed."
    )


if __name__ == "__main__":
    main()