"""Reusable PyTorch training utilities for financial-stress models."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_PATIENCE = 7
DEFAULT_MIN_DELTA = 0.0001


@dataclass
class TrainingHistory:
    """Store training and validation results across epochs."""

    train_loss: list[float]
    validation_loss: list[float]

    train_accuracy: list[float]
    validation_accuracy: list[float]

    best_epoch: int
    epochs_trained: int


def set_seed(
    seed: int = 42,
) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    """Select the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[
    float,
    float,
]:
    """Train the model for one complete epoch."""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, targets in data_loader:
        features = features.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            features
        )

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = targets.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions
            == targets
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


def run_validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    float,
]:
    """Evaluate validation loss and accuracy without updating weights."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(
                device
            )

            targets = targets.to(
                device
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                targets,
            )

            batch_size = targets.size(
                0
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions
                == targets
            ).sum().item()

            total_samples += batch_size

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> TrainingHistory:
    """Train a classification model with validation and early stopping.

    Early stopping monitors validation loss. Whenever validation loss
    improves, the model state is saved in memory.

    If validation loss does not improve for ``patience`` consecutive
    epochs, training stops and the best model state is restored.
    """

    if epochs <= 0:
        raise ValueError(
            "Epoch count must be greater than zero."
        )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if patience <= 0:
        raise ValueError(
            "Patience must be greater than zero."
        )

    if min_delta < 0:
        raise ValueError(
            "Minimum improvement must not be negative."
        )

    model.to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []

    best_validation_loss = float(
        "inf"
    )

    best_model_state = deepcopy(
        model.state_dict()
    )

    best_epoch = 0
    epochs_without_improvement = 0

    print()
    print(
        "=== TRAINING ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Maximum epochs:",
        epochs,
    )

    print(
        "Learning rate:",
        learning_rate,
    )

    print(
        "Early stopping patience:",
        patience,
    )

    print()

    for epoch in range(
        1,
        epochs + 1,
    ):
        (
            train_loss,
            train_accuracy,
        ) = run_training_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            validation_loss,
            validation_accuracy,
        ) = run_validation_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        validation_accuracies.append(
            validation_accuracy
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        improvement = (
            best_validation_loss
            - validation_loss
        )

        if improvement > min_delta:
            best_validation_loss = (
                validation_loss
            )

            best_model_state = deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print()
            print(
                "Early stopping triggered."
            )

            break

    model.load_state_dict(
        best_model_state
    )

    epochs_trained = len(
        train_losses
    )

    print()
    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        round(
            best_validation_loss,
            4,
        ),
    )

    print(
        "Epochs trained:",
        epochs_trained,
    )

    return TrainingHistory(
        train_loss=train_losses,
        validation_loss=validation_losses,

        train_accuracy=train_accuracies,
        validation_accuracy=validation_accuracies,

        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
    )
