"""Tests for the reusable PyTorch training utilities."""

import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_learning_critical_systems.training.trainer import (
    run_training_epoch,
    run_validation_epoch,
    select_device,
    set_seed,
    train_model,
)


def create_classification_loader(
    sample_count: int = 30,
    batch_size: int = 10,
) -> DataLoader:
    """Create a small deterministic classification data loader."""

    features = torch.zeros(
        sample_count,
        4,
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            index % 3
            for index in range(sample_count)
        ],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        features,
        targets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


class TinyClassifier(nn.Module):
    """Small model used only for trainer unit tests."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                4,
                8,
            ),
            nn.ReLU(),
            nn.Linear(
                8,
                3,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return three class logits."""

        return self.network(
            inputs
        )


class ConstantClassifier(nn.Module):
    """Model whose output remains constant during training."""

    def __init__(self) -> None:
        super().__init__()

        self.dummy_parameter = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return equal logits for all three classes."""

        batch_size = inputs.shape[0]

        logits = torch.zeros(
            batch_size,
            3,
            device=inputs.device,
        )

        return (
            logits
            + self.dummy_parameter * 0.0
        )


def test_set_seed_makes_random_numbers_reproducible():
    """Setting the same seed should reproduce random values."""

    set_seed(
        42
    )

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(
        1
    )

    set_seed(
        42
    )

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(
        1
    )

    assert python_value_1 == python_value_2

    assert numpy_value_1 == numpy_value_2

    assert torch.equal(
        torch_value_1,
        torch_value_2,
    )


def test_select_device_returns_supported_device():
    """Device selection should return a valid PyTorch device."""

    device = select_device()

    assert isinstance(
        device,
        torch.device,
    )

    assert device.type in {
        "cpu",
        "cuda",
        "mps",
    }


def test_training_epoch_returns_valid_metrics():
    """One training epoch should return finite loss and accuracy."""

    set_seed(
        42
    )

    model = TinyClassifier()

    data_loader = (
        create_classification_loader()
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss, accuracy = run_training_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(
            "cpu"
        ),
    )

    assert np.isfinite(
        loss
    )

    assert loss > 0.0

    assert 0.0 <= accuracy <= 1.0


def test_training_epoch_updates_model_parameters():
    """Training should change at least one trainable parameter."""

    set_seed(
        42
    )

    model = TinyClassifier()

    data_loader = (
        create_classification_loader()
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    run_training_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(
            "cpu"
        ),
    )

    parameters_after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    parameter_changed = any(
        not torch.equal(
            before,
            after,
        )
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    )

    assert parameter_changed


def test_validation_epoch_does_not_update_parameters():
    """Validation must never modify model weights."""

    set_seed(
        42
    )

    model = TinyClassifier()

    data_loader = (
        create_classification_loader()
    )

    criterion = nn.CrossEntropyLoss()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss, accuracy = run_validation_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=torch.device(
            "cpu"
        ),
    )

    parameters_after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert np.isfinite(
        loss
    )

    assert 0.0 <= accuracy <= 1.0

    for before, after in zip(
        parameters_before,
        parameters_after,
    ):
        assert torch.equal(
            before,
            after,
        )


def test_train_model_returns_complete_history():
    """The training function should record metrics for every epoch."""

    set_seed(
        42
    )

    model = TinyClassifier()

    train_loader = (
        create_classification_loader()
    )

    validation_loader = (
        create_classification_loader()
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=torch.device(
            "cpu"
        ),
        epochs=2,
        learning_rate=0.001,
        patience=2,
    )

    assert history.epochs_trained == 2

    assert len(
        history.train_loss
    ) == 2

    assert len(
        history.validation_loss
    ) == 2

    assert len(
        history.train_accuracy
    ) == 2

    assert len(
        history.validation_accuracy
    ) == 2

    assert 1 <= history.best_epoch <= 2


def test_early_stopping_stops_constant_model():
    """Training should stop when validation loss no longer improves."""

    model = ConstantClassifier()

    train_loader = (
        create_classification_loader()
    )

    validation_loader = (
        create_classification_loader()
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=torch.device(
            "cpu"
        ),
        epochs=10,
        learning_rate=0.001,
        patience=2,
        min_delta=0.0,
    )

    # Epoch 1 establishes the initial best value.
    # Epochs 2 and 3 do not improve it.
    assert history.best_epoch == 1
    assert history.epochs_trained == 3


@pytest.mark.parametrize(
    (
        "parameter_name",
        "parameter_value",
        "error_message",
    ),
    [
        (
            "epochs",
            0,
            "Epoch count",
        ),
        (
            "learning_rate",
            0.0,
            "Learning rate",
        ),
        (
            "patience",
            0,
            "Patience",
        ),
        (
            "min_delta",
            -0.1,
            "Minimum improvement",
        ),
    ],
)
def test_train_model_rejects_invalid_configuration(
    parameter_name: str,
    parameter_value: int | float,
    error_message: str,
):
    """Invalid training settings should fail immediately."""

    model = TinyClassifier()

    train_loader = (
        create_classification_loader()
    )

    validation_loader = (
        create_classification_loader()
    )

    arguments = {
        "model": model,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "device": torch.device(
            "cpu"
        ),
        "epochs": 2,
        "learning_rate": 0.001,
        "patience": 2,
        "min_delta": 0.0001,
    }

    arguments[
        parameter_name
    ] = parameter_value

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        train_model(
            **arguments
        )