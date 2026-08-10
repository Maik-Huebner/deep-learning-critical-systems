"""Train the LSTM model on the OFR financial-stress dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch

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

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "lstm_model.pt"
)

HISTORY_PATH = (
    LOG_DIR
    / "lstm_training_history.json"
)

RANDOM_SEED = 42
BATCH_SIZE = 64


def save_training_artifacts(
    model: FinancialStressLSTM,
    history,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Save the trained LSTM and its complete training history."""

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": cpu_state_dict,
        "model_name": "FinancialStressLSTM",
        "sequence_length": WINDOW_SIZE,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "classifier_hidden_size": DEFAULT_CLASSIFIER_HIDDEN_SIZE,
        "class_count": 3,
        "dropout": DEFAULT_DROPOUT,
        "forecast_horizon": FORECAST_HORIZON,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "feature_names": FEATURE_COLUMNS,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "maximum_epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "best_epoch": history.best_epoch,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    history_data = {
        "train_loss": history.train_loss,
        "validation_loss": history.validation_loss,
        "train_accuracy": history.train_accuracy,
        "validation_accuracy": history.validation_accuracy,
        "best_epoch": history.best_epoch,
        "epochs_trained": history.epochs_trained,
    }

    HISTORY_PATH.write_text(
        json.dumps(
            history_data,
            indent=2,
        )
        + "\n"
    )

    print()
    print("Saved checkpoint:")
    print(CHECKPOINT_PATH)

    print()
    print("Saved training history:")
    print(HISTORY_PATH)


def main() -> None:
    """Train and save the LSTM financial-stress model."""

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

    model = FinancialStressLSTM(
        feature_count=len(
            FEATURE_COLUMNS
        ),
    )

    device = select_device()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("=== LSTM EXPERIMENT ===")

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
        "Window size:",
        WINDOW_SIZE,
    )

    print(
        "Feature count:",
        len(
            FEATURE_COLUMNS
        ),
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        patience=DEFAULT_PATIENCE,
    )

    save_training_artifacts(
        model=model,
        history=history,
        low_threshold=prepared_data.low_threshold,
        high_threshold=prepared_data.high_threshold,
    )

    print()
    print("=== LSTM TRAINING COMPLETE ===")

    print(
        "Best epoch:",
        history.best_epoch,
    )

    print(
        "Epochs trained:",
        history.epochs_trained,
    )

    print(
        "Best validation loss:",
        round(
            min(
                history.validation_loss
            ),
            4,
        ),
    )

    print(
        "Validation accuracy at best epoch:",
        round(
            history.validation_accuracy[
                history.best_epoch - 1
            ],
            4,
        ),
    )


if __name__ == "__main__":
    main()
