"""PyTorch datasets and data loaders for OFR time-series sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    PreparedOFRData,
)


DEFAULT_BATCH_SIZE = 64


class OFRSequenceDataset(Dataset):
    """PyTorch dataset for OFR financial-stress sequences.

    Each sample contains:

    - a sequence with shape (timesteps, features)
    - one integer target class

    Target classes:

    0 = Stress Decrease
    1 = Stable
    2 = Stress Increase
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Store NumPy arrays as PyTorch tensors."""

        if len(features) != len(targets):
            raise ValueError(
                "Features and targets must contain "
                "the same number of samples."
            )

        if features.ndim != 3:
            raise ValueError(
                "Features must have shape "
                "(samples, timesteps, features)."
            )

        if targets.ndim != 1:
            raise ValueError(
                "Targets must have shape (samples,)."
            )

        self.features = torch.tensor(
            features,
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        """Return the number of available sequences."""

        return len(self.features)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return one sequence and its target class."""

        return (
            self.features[index],
            self.targets[index],
        )


def create_datasets(
    prepared_data: PreparedOFRData,
) -> tuple[
    OFRSequenceDataset,
    OFRSequenceDataset,
    OFRSequenceDataset,
]:
    """Create train, validation and test datasets."""

    train_dataset = OFRSequenceDataset(
        prepared_data.X_train,
        prepared_data.y_train,
    )

    validation_dataset = OFRSequenceDataset(
        prepared_data.X_validation,
        prepared_data.y_validation,
    )

    test_dataset = OFRSequenceDataset(
        prepared_data.X_test,
        prepared_data.y_test,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def create_data_loaders(
    prepared_data: PreparedOFRData,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """Create chronological PyTorch data loaders.

    The data loaders do not shuffle observations because this project
    works with time-series data and preserves chronological order.
    """

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(
        prepared_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )
