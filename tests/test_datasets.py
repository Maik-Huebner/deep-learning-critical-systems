"""Tests for the PyTorch OFR datasets and data loaders."""

import numpy as np
import pytest
import torch

from deep_learning_critical_systems.data.datasets import (
    OFRSequenceDataset,
    create_data_loaders,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    PreparedOFRData,
)


def create_prepared_test_data() -> PreparedOFRData:
    """Create small deterministic prepared arrays for dataset tests."""

    X_train = np.zeros(
        (10, 60, 9),
        dtype=np.float32,
    )

    y_train = np.array(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        dtype=np.int64,
    )

    X_validation = np.ones(
        (6, 60, 9),
        dtype=np.float32,
    )

    y_validation = np.array(
        [0, 1, 2, 0, 1, 2],
        dtype=np.int64,
    )

    X_test = np.full(
        (8, 60, 9),
        2.0,
        dtype=np.float32,
    )

    y_test = np.array(
        [0, 1, 2, 0, 1, 2, 0, 1],
        dtype=np.int64,
    )

    dates_train = np.arange(
        len(X_train)
    )

    dates_validation = np.arange(
        len(X_validation)
    )

    dates_test = np.arange(
        len(X_test)
    )

    return PreparedOFRData(
        X_train=X_train,
        y_train=y_train,
        dates_train=dates_train,

        X_validation=X_validation,
        y_validation=y_validation,
        dates_validation=dates_validation,

        X_test=X_test,
        y_test=y_test,
        dates_test=dates_test,

        feature_names=[
            f"feature_{index}"
            for index in range(9)
        ],

        low_threshold=-0.1,
        high_threshold=0.1,

        scaler=None,
    )


def test_dataset_returns_correct_tensor_shapes():
    """Dataset samples should contain one sequence and one class label."""

    features = np.zeros(
        (5, 60, 9),
        dtype=np.float32,
    )

    targets = np.array(
        [0, 1, 2, 0, 1],
        dtype=np.int64,
    )

    dataset = OFRSequenceDataset(
        features,
        targets,
    )

    sequence, target = dataset[0]

    assert len(dataset) == 5

    assert sequence.shape == (
        60,
        9,
    )

    assert target.shape == torch.Size([])


def test_dataset_uses_correct_tensor_types():
    """Features and targets must have model-compatible PyTorch dtypes."""

    features = np.zeros(
        (5, 60, 9),
        dtype=np.float64,
    )

    targets = np.array(
        [0, 1, 2, 0, 1],
        dtype=np.int32,
    )

    dataset = OFRSequenceDataset(
        features,
        targets,
    )

    sequence, target = dataset[0]

    assert sequence.dtype == torch.float32
    assert target.dtype == torch.int64


def test_dataset_rejects_different_sample_counts():
    """Features and targets must contain the same number of samples."""

    features = np.zeros(
        (5, 60, 9),
        dtype=np.float32,
    )

    targets = np.array(
        [0, 1, 2, 0],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        OFRSequenceDataset(
            features,
            targets,
        )


def test_dataset_rejects_invalid_feature_shape():
    """Input features must be three-dimensional sequences."""

    features = np.zeros(
        (5, 9),
        dtype=np.float32,
    )

    targets = np.array(
        [0, 1, 2, 0, 1],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="samples, timesteps, features",
    ):
        OFRSequenceDataset(
            features,
            targets,
        )


def test_dataset_rejects_invalid_target_shape():
    """Targets must be a one-dimensional class array."""

    features = np.zeros(
        (5, 60, 9),
        dtype=np.float32,
    )

    targets = np.zeros(
        (5, 1),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="Targets must have shape",
    ):
        OFRSequenceDataset(
            features,
            targets,
        )


def test_data_loaders_create_expected_batches():
    """Data loaders should create correct batch shapes."""

    prepared_data = create_prepared_test_data()

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(
        prepared_data,
        batch_size=4,
    )

    X_batch, y_batch = next(
        iter(train_loader)
    )

    assert X_batch.shape == (
        4,
        60,
        9,
    )

    assert y_batch.shape == (
        4,
    )

    assert len(train_loader) == 3
    assert len(validation_loader) == 2
    assert len(test_loader) == 2


def test_data_loader_preserves_sample_order():
    """Time-series samples must remain in chronological dataset order."""

    prepared_data = create_prepared_test_data()

    for sample_index in range(
        len(prepared_data.X_train)
    ):
        prepared_data.X_train[
            sample_index
        ] = sample_index

    train_loader, _, _ = create_data_loaders(
        prepared_data,
        batch_size=4,
    )

    first_batch, _ = next(
        iter(train_loader)
    )

    assert torch.all(
        first_batch[0] == 0
    )

    assert torch.all(
        first_batch[1] == 1
    )

    assert torch.all(
        first_batch[2] == 2
    )

    assert torch.all(
        first_batch[3] == 3
    )


def test_data_loader_rejects_invalid_batch_size():
    """Batch size must be greater than zero."""

    prepared_data = create_prepared_test_data()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        create_data_loaders(
            prepared_data,
            batch_size=0,
        )
