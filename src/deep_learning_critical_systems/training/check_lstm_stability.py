"""Small multi-seed stability check for the two strongest LSTM configs.

This analysis compares the two strongest LSTM configurations from the
validation-based hyperparameter tuning:

- L1: learning rate 0.0005, dropout 0.20
- L9: learning rate 0.0005, dropout 0.30

Both configurations are retrained with three predefined random seeds.
The test set is deliberately not used.

The configuration with the highest mean validation Macro-F1 is selected.
Mean validation loss is used as the tie-breaker.

After configuration selection, the canonical seed-42 model state of the
winning configuration is saved as the final tuned LSTM checkpoint.
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

FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "lstm_tuned_model.pt"
)

FINAL_HISTORY_PATH = (
    LOG_DIR
    / "lstm_tuned_training_history.json"
)

STABILITY_RESULTS_JSON_PATH = (
    LOG_DIR
    / "lstm_stability_results.json"
)

STABILITY_RESULTS_CSV_PATH = (
    LOG_DIR
    / "lstm_stability_results.csv"
)

STABILITY_SUMMARY_JSON_PATH = (
    LOG_DIR
    / "lstm_stability_summary.json"
)

STABILITY_SUMMARY_CSV_PATH = (
    LOG_DIR
    / "lstm_stability_summary.csv"
)

STABILITY_HISTORIES_PATH = (
    LOG_DIR
    / "lstm_stability_histories.json"
)

CANONICAL_SEED = 42

STABILITY_SEEDS = [
    42,
    123,
    2026,
]

BATCH_SIZE = 64

FINALIST_CONFIGURATIONS = [
    {
        "run_id": "L1",
        "description": "Lower learning rate",
        "hidden_size": 64,
        "classifier_hidden_size": 32,
        "dropout": 0.20,
        "learning_rate": 0.0005,
    },
    {
        "run_id": "L9",
        "description": (
            "Lower learning rate with higher classifier dropout"
        ),
        "hidden_size": 64,
        "classifier_hidden_size": 32,
        "dropout": 0.30,
        "learning_rate": 0.0005,
    },
]


def history_to_dict(
    history,
) -> dict:
    """Convert TrainingHistory into JSON-compatible data."""

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
    """Create one LSTM finalist configuration."""

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


def build_run_result(
    configuration: dict,
    seed: int,
    model: FinancialStressLSTM,
    history: dict,
    validation_metrics: dict[str, float],
) -> dict:
    """Create one result record for one configuration and one seed."""

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
        "seed": seed,
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
        **validation_metrics,
    }


def summarize_configuration(
    configuration: dict,
    run_results: list[dict],
) -> dict:
    """Summarize stability metrics across the predefined seeds."""

    macro_f1_values = np.asarray(
        [
            result[
                "validation_macro_f1"
            ]
            for result in run_results
        ],
        dtype=np.float64,
    )

    validation_loss_values = np.asarray(
        [
            result[
                "validation_loss"
            ]
            for result in run_results
        ],
        dtype=np.float64,
    )

    accuracy_values = np.asarray(
        [
            result[
                "validation_accuracy"
            ]
            for result in run_results
        ],
        dtype=np.float64,
    )

    increase_recall_values = np.asarray(
        [
            result[
                "recall_stress_increase"
            ]
            for result in run_results
        ],
        dtype=np.float64,
    )

    return {
        "run_id": configuration[
            "run_id"
        ],
        "description": configuration[
            "description"
        ],
        "seed_count": len(
            run_results
        ),
        "seeds": STABILITY_SEEDS,
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
        "mean_validation_macro_f1": float(
            macro_f1_values.mean()
        ),
        "std_validation_macro_f1": float(
            macro_f1_values.std(
                ddof=0
            )
        ),
        "mean_validation_loss": float(
            validation_loss_values.mean()
        ),
        "std_validation_loss": float(
            validation_loss_values.std(
                ddof=0
            )
        ),
        "mean_validation_accuracy": float(
            accuracy_values.mean()
        ),
        "std_validation_accuracy": float(
            accuracy_values.std(
                ddof=0
            )
        ),
        "mean_recall_stress_increase": float(
            increase_recall_values.mean()
        ),
        "std_recall_stress_increase": float(
            increase_recall_values.std(
                ddof=0
            )
        ),
    }


def print_run_result(
    result: dict,
) -> None:
    """Print one seed-specific validation result."""

    print()
    print(
        f"{result['run_id']} | "
        f"Seed {result['seed']}"
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


def print_summary(
    summary: dict,
) -> None:
    """Print the stability summary of one finalist."""

    print()
    print(
        f"{summary['run_id']} - "
        f"{summary['description']}"
    )

    print(
        "Mean Validation Macro-F1:",
        round(
            summary[
                "mean_validation_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Std Validation Macro-F1:",
        round(
            summary[
                "std_validation_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Mean Validation Loss:",
        round(
            summary[
                "mean_validation_loss"
            ],
            4,
        ),
    )

    print(
        "Mean Validation Accuracy:",
        round(
            summary[
                "mean_validation_accuracy"
            ],
            4,
        ),
    )

    print(
        "Mean Stress Increase Recall:",
        round(
            summary[
                "mean_recall_stress_increase"
            ],
            4,
        ),
    )


def save_results(
    run_results: list[dict],
    summaries: list[dict],
    histories: dict,
) -> None:
    """Save seed-level results, summaries and training histories."""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STABILITY_RESULTS_JSON_PATH.write_text(
        json.dumps(
            run_results,
            indent=2,
        )
        + "\n"
    )

    with STABILITY_RESULTS_CSV_PATH.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                run_results[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            run_results
        )

    STABILITY_SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summaries,
            indent=2,
        )
        + "\n"
    )

    summary_rows = []

    for summary in summaries:
        summary_rows.append(
            {
                key: (
                    ",".join(
                        str(seed)
                        for seed in value
                    )
                    if key == "seeds"
                    else value
                )
                for key, value in summary.items()
            }
        )

    with STABILITY_SUMMARY_CSV_PATH.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                summary_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    STABILITY_HISTORIES_PATH.write_text(
        json.dumps(
            histories,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved stability analysis:"
    )
    print(
        STABILITY_RESULTS_JSON_PATH
    )
    print(
        STABILITY_RESULTS_CSV_PATH
    )
    print(
        STABILITY_SUMMARY_JSON_PATH
    )
    print(
        STABILITY_SUMMARY_CSV_PATH
    )
    print(
        STABILITY_HISTORIES_PATH
    )


def save_final_model(
    selected_configuration: dict,
    selected_summary: dict,
    canonical_result: dict,
    canonical_state_dict: dict,
    canonical_history: dict,
    prepared_data,
) -> None:
    """Save the canonical seed-42 state of the selected configuration."""

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": canonical_state_dict,
        "model_name": "FinancialStressLSTM",
        "selected_run": selected_configuration[
            "run_id"
        ],
        "selection_method": (
            "three_seed_validation_stability_check"
        ),
        "selection_metric": (
            "mean_validation_macro_f1"
        ),
        "selection_tie_breaker": (
            "mean_validation_loss"
        ),
        "stability_seeds": STABILITY_SEEDS,
        "canonical_model_seed": CANONICAL_SEED,
        "sequence_length": WINDOW_SIZE,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "hidden_size": selected_configuration[
            "hidden_size"
        ],
        "classifier_hidden_size": selected_configuration[
            "classifier_hidden_size"
        ],
        "class_count": 3,
        "dropout": selected_configuration[
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
        "batch_size": BATCH_SIZE,
        "learning_rate": selected_configuration[
            "learning_rate"
        ],
        "maximum_epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "best_epoch": canonical_result[
            "best_epoch"
        ],
        "validation_loss": canonical_result[
            "validation_loss"
        ],
        "validation_accuracy": canonical_result[
            "validation_accuracy"
        ],
        "validation_macro_precision": canonical_result[
            "validation_macro_precision"
        ],
        "validation_macro_recall": canonical_result[
            "validation_macro_recall"
        ],
        "validation_macro_f1": canonical_result[
            "validation_macro_f1"
        ],
        "recall_stress_decrease": canonical_result[
            "recall_stress_decrease"
        ],
        "recall_stable": canonical_result[
            "recall_stable"
        ],
        "recall_stress_increase": canonical_result[
            "recall_stress_increase"
        ],
        "mean_validation_macro_f1": selected_summary[
            "mean_validation_macro_f1"
        ],
        "std_validation_macro_f1": selected_summary[
            "std_validation_macro_f1"
        ],
        "mean_validation_loss": selected_summary[
            "mean_validation_loss"
        ],
        "mean_validation_accuracy": selected_summary[
            "mean_validation_accuracy"
        ],
        "mean_recall_stress_increase": selected_summary[
            "mean_recall_stress_increase"
        ],
    }

    torch.save(
        checkpoint,
        FINAL_CHECKPOINT_PATH,
    )

    FINAL_HISTORY_PATH.write_text(
        json.dumps(
            canonical_history,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved final stability-selected LSTM:"
    )
    print(
        FINAL_CHECKPOINT_PATH
    )
    print(
        FINAL_HISTORY_PATH
    )


def main() -> None:
    """Run the predefined three-seed stability comparison."""

    prepared_data = prepare_ofr_data()

    device = select_device()

    print()
    print(
        "=== LSTM MULTI-SEED STABILITY CHECK ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Finalists:",
        ", ".join(
            configuration[
                "run_id"
            ]
            for configuration
            in FINALIST_CONFIGURATIONS
        ),
    )

    print(
        "Seeds:",
        STABILITY_SEEDS,
    )

    print(
        "Primary selection metric:"
        " Mean Validation Macro-F1"
    )

    print(
        "Tie-breaker:"
        " Mean Validation Loss"
    )

    print(
        "Test set used: NO"
    )

    run_results = []
    histories = {}

    canonical_state_dicts = {}
    canonical_histories = {}
    canonical_results = {}

    for configuration in FINALIST_CONFIGURATIONS:
        print()
        print(
            "=" * 72
        )
        print(
            f"Configuration {configuration['run_id']}: "
            f"{configuration['description']}"
        )
        print(
            "=" * 72
        )

        for seed in STABILITY_SEEDS:
            print()
            print(
                "-" * 72
            )
            print(
                f"Training {configuration['run_id']} "
                f"with seed {seed}"
            )
            print(
                "-" * 72
            )

            set_seed(
                seed
            )

            (
                train_loader,
                validation_loader,
                _,
            ) = create_data_loaders(
                prepared_data,
                batch_size=BATCH_SIZE,
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

            result = build_run_result(
                configuration=configuration,
                seed=seed,
                model=model,
                history=history_data,
                validation_metrics=validation_metrics,
            )

            run_results.append(
                result
            )

            history_key = (
                f"{configuration['run_id']}"
                f"_seed_{seed}"
            )

            histories[
                history_key
            ] = history_data

            if seed == CANONICAL_SEED:
                canonical_state_dicts[
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

                canonical_histories[
                    configuration[
                        "run_id"
                    ]
                ] = history_data

                canonical_results[
                    configuration[
                        "run_id"
                    ]
                ] = result

            print_run_result(
                result
            )

    summaries = []

    for configuration in FINALIST_CONFIGURATIONS:
        configuration_results = [
            result
            for result in run_results
            if result[
                "run_id"
            ]
            == configuration[
                "run_id"
            ]
        ]

        summary = summarize_configuration(
            configuration,
            configuration_results,
        )

        summaries.append(
            summary
        )

    ranked_summaries = sorted(
        summaries,
        key=lambda summary: (
            -summary[
                "mean_validation_macro_f1"
            ],
            summary[
                "mean_validation_loss"
            ],
        ),
    )

    print()
    print(
        "=" * 72
    )
    print(
        "=== STABILITY SUMMARY ==="
    )
    print(
        "=" * 72
    )

    for summary in ranked_summaries:
        print_summary(
            summary
        )

    selected_summary = ranked_summaries[
        0
    ]

    selected_run = selected_summary[
        "run_id"
    ]

    selected_configuration = next(
        configuration
        for configuration in FINALIST_CONFIGURATIONS
        if configuration[
            "run_id"
        ]
        == selected_run
    )

    save_results(
        run_results=run_results,
        summaries=ranked_summaries,
        histories=histories,
    )

    save_final_model(
        selected_configuration=selected_configuration,
        selected_summary=selected_summary,
        canonical_result=canonical_results[
            selected_run
        ],
        canonical_state_dict=deepcopy(
            canonical_state_dicts[
                selected_run
            ]
        ),
        canonical_history=canonical_histories[
            selected_run
        ],
        prepared_data=prepared_data,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "=== FINAL STABILITY-SELECTED LSTM ==="
    )
    print(
        "=" * 72
    )

    print(
        "Run:",
        selected_run,
    )

    print(
        "Description:",
        selected_configuration[
            "description"
        ],
    )

    print(
        "Hidden size:",
        selected_configuration[
            "hidden_size"
        ],
    )

    print(
        "Classifier hidden size:",
        selected_configuration[
            "classifier_hidden_size"
        ],
    )

    print(
        "Dropout:",
        selected_configuration[
            "dropout"
        ],
    )

    print(
        "Learning rate:",
        selected_configuration[
            "learning_rate"
        ],
    )

    print(
        "Mean Validation Macro-F1:",
        round(
            selected_summary[
                "mean_validation_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Std Validation Macro-F1:",
        round(
            selected_summary[
                "std_validation_macro_f1"
            ],
            4,
        ),
    )

    print(
        "Mean Validation Loss:",
        round(
            selected_summary[
                "mean_validation_loss"
            ],
            4,
        ),
    )

    print(
        "Mean Stress Increase Recall:",
        round(
            selected_summary[
                "mean_recall_stress_increase"
            ],
            4,
        ),
    )

    print(
        "Canonical saved model seed:",
        CANONICAL_SEED,
    )

    print()
    print(
        "Test set used during stability check: NO"
    )

    print()
    print(
        "LSTM multi-seed stability check completed."
    )


if __name__ == "__main__":
    main()
