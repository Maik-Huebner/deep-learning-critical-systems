"""Second-stage validation tuning for the LSTM model.

Stage 1 tested individual hyperparameter changes against the untuned
LSTM baseline.

Stage 2 now investigates the strongest Stage-1 findings in more detail:

- nearby learning rates around the Stage-1 winner
- combinations of the winning learning rate with different dropout
- the winning learning rate combined with a larger LSTM hidden size

The test set remains completely unused.

Model states are selected inside each training run by validation loss.
The final configuration is selected across runs primarily by
validation Macro-F1 and secondarily by validation loss.
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
    FinancialStressLSTM,
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

STAGE_1_RESULTS_PATH = (
    LOG_DIR
    / "lstm_tuning_results.json"
)

STAGE_1_HISTORIES_PATH = (
    LOG_DIR
    / "lstm_tuning_histories.json"
)

FINAL_RESULTS_JSON_PATH = (
    LOG_DIR
    / "lstm_tuning_results_final.json"
)

FINAL_RESULTS_CSV_PATH = (
    LOG_DIR
    / "lstm_tuning_results_final.csv"
)

FINAL_HISTORIES_PATH = (
    LOG_DIR
    / "lstm_tuning_histories_final.json"
)

FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "lstm_tuned_model.pt"
)

FINAL_HISTORY_PATH = (
    LOG_DIR
    / "lstm_tuned_training_history.json"
)

RANDOM_SEED = 42
BATCH_SIZE = 64

BASE_MODEL_CONFIGURATION = {
    "hidden_size": 64,
    "classifier_hidden_size": 32,
    "dropout": 0.20,
    "learning_rate": 0.0005,
}

STAGE_2_CONFIGURATIONS = [
    {
        "run_id": "L7",
        "description": "Lower learning rate around L1",
        **BASE_MODEL_CONFIGURATION,
        "learning_rate": 0.00025,
    },
    {
        "run_id": "L8",
        "description": "Intermediate learning rate around L1",
        **BASE_MODEL_CONFIGURATION,
        "learning_rate": 0.00075,
    },
    {
        "run_id": "L9",
        "description": "L1 learning rate with higher classifier dropout",
        **BASE_MODEL_CONFIGURATION,
        "dropout": 0.30,
    },
    {
        "run_id": "L10",
        "description": "L1 learning rate with lower classifier dropout",
        **BASE_MODEL_CONFIGURATION,
        "dropout": 0.10,
    },
    {
        "run_id": "L11",
        "description": "L1 learning rate with larger LSTM hidden size",
        **BASE_MODEL_CONFIGURATION,
        "hidden_size": 128,
    },
]


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


def history_to_dict(
    history,
) -> dict:
    """Convert TrainingHistory to JSON-compatible data."""

    return {
        "train_loss": history.train_loss,
        "validation_loss": history.validation_loss,
        "train_accuracy": history.train_accuracy,
        "validation_accuracy": history.validation_accuracy,
        "best_epoch": history.best_epoch,
        "epochs_trained": history.epochs_trained,
    }


def create_model(
    configuration: dict,
) -> FinancialStressLSTM:
    """Create an LSTM from one experiment configuration."""

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
    """Calculate validation metrics for one trained model."""

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
        labels=[0, 1, 2],
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


def create_result(
    configuration: dict,
    model: FinancialStressLSTM,
    history: dict,
    metrics: dict[str, float],
) -> dict:
    """Create one complete tuning result record."""

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "run_id": configuration[
            "run_id"
        ],
        "description": configuration[
            "description"
        ],
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
            min(
                history[
                    "validation_loss"
                ]
            )
        ),
        **metrics,
    }


def print_result(
    result: dict,
) -> None:
    """Print the most important values of one experiment."""

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
        "Epochs trained:",
        result[
            "epochs_trained"
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
        "Validation Macro Precision:",
        round(
            result[
                "validation_macro_precision"
            ],
            4,
        ),
    )

    print(
        "Validation Macro Recall:",
        round(
            result[
                "validation_macro_recall"
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
        "Stress Decrease recall:",
        round(
            result[
                "recall_stress_decrease"
            ],
            4,
        ),
    )

    print(
        "Stable recall:",
        round(
            result[
                "recall_stable"
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


def save_combined_results(
    results: list[dict],
    histories: dict,
) -> None:
    """Save Stage-1 and Stage-2 results together."""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_RESULTS_JSON_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    FINAL_HISTORIES_PATH.write_text(
        json.dumps(
            histories,
            indent=2,
        )
        + "\n"
    )

    fieldnames = list(
        results[0].keys()
    )

    with FINAL_RESULTS_CSV_PATH.open(
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
        "Saved final combined tuning results:"
    )
    print(
        FINAL_RESULTS_JSON_PATH
    )
    print(
        FINAL_RESULTS_CSV_PATH
    )
    print(
        FINAL_HISTORIES_PATH
    )


def save_selected_model(
    result: dict,
    state_dict: dict,
    history: dict,
    prepared_data,
) -> None:
    """Save the final LSTM selected using validation results."""

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": state_dict,
        "model_name": "FinancialStressLSTM",
        "selected_run": result[
            "run_id"
        ],
        "selection_metric": (
            "validation_macro_f1"
        ),
        "selection_tie_breaker": (
            "validation_loss"
        ),
        "sequence_length": WINDOW_SIZE,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "hidden_size": result[
            "hidden_size"
        ],
        "classifier_hidden_size": result[
            "classifier_hidden_size"
        ],
        "class_count": 3,
        "dropout": result[
            "dropout"
        ],
        "forecast_horizon": FORECAST_HORIZON,
        "low_threshold": (
            prepared_data.low_threshold
        ),
        "high_threshold": (
            prepared_data.high_threshold
        ),
        "feature_names": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": result[
            "learning_rate"
        ],
        "maximum_epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "best_epoch": result[
            "best_epoch"
        ],
        "validation_loss": result[
            "validation_loss"
        ],
        "validation_accuracy": result[
            "validation_accuracy"
        ],
        "validation_macro_precision": result[
            "validation_macro_precision"
        ],
        "validation_macro_recall": result[
            "validation_macro_recall"
        ],
        "validation_macro_f1": result[
            "validation_macro_f1"
        ],
        "recall_stress_decrease": result[
            "recall_stress_decrease"
        ],
        "recall_stable": result[
            "recall_stable"
        ],
        "recall_stress_increase": result[
            "recall_stress_increase"
        ],
    }

    torch.save(
        checkpoint,
        FINAL_CHECKPOINT_PATH,
    )

    FINAL_HISTORY_PATH.write_text(
        json.dumps(
            history,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved final selected LSTM:"
    )
    print(
        FINAL_CHECKPOINT_PATH
    )
    print(
        FINAL_HISTORY_PATH
    )


def main() -> None:
    """Run Stage 2 of LSTM hyperparameter tuning."""

    stage_1_results = load_json(
        STAGE_1_RESULTS_PATH
    )

    stage_1_histories = load_json(
        STAGE_1_HISTORIES_PATH
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

    print()
    print(
        "=== LSTM HYPERPARAMETER TUNING: STAGE 2 ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Training samples:",
        len(
            prepared_data.y_train
        ),
    )

    print(
        "Validation samples:",
        len(
            prepared_data.y_validation
        ),
    )

    print(
        "Stage-1 experiments loaded:",
        len(
            stage_1_results
        ),
    )

    print(
        "New Stage-2 experiments:",
        len(
            STAGE_2_CONFIGURATIONS
        ),
    )

    print(
        "Primary selection metric:"
        " Validation Macro-F1"
    )

    print(
        "Tie-breaker:"
        " Validation Loss"
    )

    print(
        "Test set used: NO"
    )

    stage_2_results = []
    stage_2_histories = {}
    stage_2_state_dicts = {}

    for configuration in STAGE_2_CONFIGURATIONS:
        print()
        print(
            "=" * 72
        )

        print(
            "Starting "
            f"{configuration['run_id']}: "
            f"{configuration['description']}"
        )

        print(
            "=" * 72
        )

        print(
            "Hidden size:",
            configuration[
                "hidden_size"
            ],
        )

        print(
            "Classifier hidden size:",
            configuration[
                "classifier_hidden_size"
            ],
        )

        print(
            "Dropout:",
            configuration[
                "dropout"
            ],
        )

        print(
            "Learning rate:",
            configuration[
                "learning_rate"
            ],
        )

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

        result = create_result(
            configuration=configuration,
            model=model,
            history=history_data,
            metrics=validation_metrics,
        )

        stage_2_results.append(
            result
        )

        stage_2_histories[
            configuration[
                "run_id"
            ]
        ] = history_data

        stage_2_state_dicts[
            configuration[
                "run_id"
            ]
        ] = {
            name: parameter
            .detach()
            .cpu()
            .clone()
            for name, parameter
            in model.state_dict().items()
        }

        print_result(
            result
        )

    all_results = (
        stage_1_results
        + stage_2_results
    )

    all_histories = {
        **stage_1_histories,
        **stage_2_histories,
    }

    ranked_results = sorted(
        all_results,
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
        "=" * 72
    )
    print(
        "=== FINAL TUNING RANKING ==="
    )
    print(
        "=" * 72
    )
    print()

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{result['run_id']:<3} | "
            f"Macro-F1: "
            f"{result['validation_macro_f1']:.4f} | "
            f"Val Loss: "
            f"{result['validation_loss']:.4f} | "
            f"Val Acc: "
            f"{result['validation_accuracy']:.4f} | "
            f"Increase Recall: "
            f"{result['recall_stress_increase']:.4f}"
        )

    best_result = ranked_results[
        0
    ]

    save_combined_results(
        results=all_results,
        histories=all_histories,
    )

    selected_run = best_result[
        "run_id"
    ]

    if selected_run in stage_2_state_dicts:
        selected_state_dict = deepcopy(
            stage_2_state_dicts[
                selected_run
            ]
        )

    else:
        existing_checkpoint = torch.load(
            FINAL_CHECKPOINT_PATH,
            map_location="cpu",
        )

        if (
            existing_checkpoint.get(
                "selected_run"
            )
            != selected_run
        ):
            raise RuntimeError(
                "The winning Stage-1 model state is not "
                "available in the current checkpoint."
            )

        selected_state_dict = {
            name: parameter
            .detach()
            .cpu()
            .clone()
            for name, parameter
            in existing_checkpoint[
                "model_state_dict"
            ].items()
        }

    selected_history = all_histories[
        selected_run
    ]

    save_selected_model(
        result=best_result,
        state_dict=selected_state_dict,
        history=selected_history,
        prepared_data=prepared_data,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "=== FINAL SELECTED LSTM ==="
    )
    print(
        "=" * 72
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
        "Parameter count:",
        best_result[
            "parameter_count"
        ],
    )

    print(
        "Best epoch:",
        best_result[
            "best_epoch"
        ],
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

    print(
        "Validation accuracy:",
        round(
            best_result[
                "validation_accuracy"
            ],
            4,
        ),
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
        "Stress Increase recall:",
        round(
            best_result[
                "recall_stress_increase"
            ],
            4,
        ),
    )

    print()
    print(
        "Test set used during tuning: NO"
    )

    print()
    print(
        "Stage-2 LSTM tuning completed."
    )


if __name__ == "__main__":
    main()
