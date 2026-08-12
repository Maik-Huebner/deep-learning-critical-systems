"""Validation-based hyperparameter tuning for the LSTM model.

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
from deep_learning_critical_systems.models.lstm import (
    DEFAULT_CLASSIFIER_HIDDEN_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_SIZE,
    FinancialStressLSTM,
)
from deep_learning_critical_systems.training.trainer import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
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
    / "lstm_model.pt"
)

BASELINE_HISTORY_PATH = (
    LOG_DIR
    / "lstm_training_history.json"
)

TUNED_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "lstm_tuned_model.pt"
)

TUNED_HISTORY_PATH = (
    LOG_DIR
    / "lstm_tuned_training_history.json"
)

TUNING_RESULTS_PATH = (
    LOG_DIR
    / "lstm_tuning_results.json"
)

TUNING_RESULTS_CSV_PATH = (
    LOG_DIR
    / "lstm_tuning_results.csv"
)

TUNING_HISTORIES_PATH = (
    LOG_DIR
    / "lstm_tuning_histories.json"
)

RANDOM_SEED = 42
BATCH_SIZE = 64

BASE_CONFIGURATION = {
    "hidden_size": DEFAULT_HIDDEN_SIZE,
    "classifier_hidden_size": DEFAULT_CLASSIFIER_HIDDEN_SIZE,
    "dropout": DEFAULT_DROPOUT,
    "learning_rate": DEFAULT_LEARNING_RATE,
}

TUNING_CONFIGURATIONS = [
    {
        "run_id": "L1",
        "description": "Lower learning rate",
        **BASE_CONFIGURATION,
        "learning_rate": 0.0005,
    },
    {
        "run_id": "L2",
        "description": "Smaller LSTM hidden size",
        **BASE_CONFIGURATION,
        "hidden_size": 32,
    },
    {
        "run_id": "L3",
        "description": "Larger LSTM hidden size",
        **BASE_CONFIGURATION,
        "hidden_size": 128,
    },
    {
        "run_id": "L4",
        "description": "Lower classifier dropout",
        **BASE_CONFIGURATION,
        "dropout": 0.10,
    },
    {
        "run_id": "L5",
        "description": "Higher classifier dropout",
        **BASE_CONFIGURATION,
        "dropout": 0.30,
    },
    {
        "run_id": "L6",
        "description": "Larger classifier hidden size",
        **BASE_CONFIGURATION,
        "classifier_hidden_size": 64,
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
) -> FinancialStressLSTM:
    """Create one LSTM from a tuning configuration."""

    return FinancialStressLSTM(
        feature_count=len(
            FEATURE_COLUMNS
        ),
        hidden_size=configuration[
            "hidden_size"
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
    model: FinancialStressLSTM,
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
    FinancialStressLSTM,
    dict,
    dict,
]:
    """Load the already trained L0 LSTM baseline."""

    if not BASELINE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Baseline LSTM checkpoint not found: "
            f"{BASELINE_CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT_PATH,
        map_location="cpu",
    )

    configuration = {
        "hidden_size": checkpoint[
            "hidden_size"
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
    model: FinancialStressLSTM,
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
        "hidden_size": configuration[
            "hidden_size"
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
        + '\n'
    )

    TUNING_HISTORIES_PATH.write_text(
        json.dumps(
            histories,
            indent=2,
        )
        + '\n'
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
    """Save the selected validation-best LSTM."""

    checkpoint = {
        "model_state_dict": best_state_dict,
        "model_name": "FinancialStressLSTM",
        "selected_run": best_result[
            "run_id"
        ],
        "selection_metric": "validation_macro_f1",
        "sequence_length": WINDOW_SIZE,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "hidden_size": best_result[
            "hidden_size"
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
        + '\n'
    )

    print()
    print(
        "Saved selected LSTM:"
    )

    print(
        TUNED_CHECKPOINT_PATH
    )

    print(
        TUNED_HISTORY_PATH
    )


def main() -> None:
    """Run the controlled LSTM tuning experiment."""

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
        "=== LSTM HYPERPARAMETER TUNING ==="
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
        "Loading L0 baseline..."
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
        run_id="L0",
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
        "L0"
    ] = baseline_history

    state_dicts[
        "L0"
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
        "Hidden size:",
        best_result[
            "hidden_size"
        ],
    )

    print(
        "Classifier hidden size:",
        best_result[
            "classifier_hidden_size"
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
        "Trainable parameters:",
        best_result[
            "parameter_count"
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
        "LSTM tuning completed."
    )


if __name__ == "__main__":
    main()
