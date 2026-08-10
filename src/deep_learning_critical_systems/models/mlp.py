"""MLP baseline model for financial-stress classification."""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_FEATURE_COUNT = 9
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_SECOND_HIDDEN_SIZE = 64
DEFAULT_CLASS_COUNT = 3
DEFAULT_DROPOUT = 0.20


class FinancialStressMLP(nn.Module):
    """Feed-forward baseline for OFR financial-stress sequences.

    The model receives a complete time-series window with shape:

        (batch_size, sequence_length, feature_count)

    The sequence is flattened before being passed through fully
    connected layers.

    This makes the MLP a useful baseline because it receives the same
    information as the later sequence models, but it does not contain
    recurrent layers or an attention mechanism.
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        feature_count: int = DEFAULT_FEATURE_COUNT,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        second_hidden_size: int = DEFAULT_SECOND_HIDDEN_SIZE,
        class_count: int = DEFAULT_CLASS_COUNT,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Initialize the MLP baseline."""

        super().__init__()

        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be greater than zero."
            )

        if feature_count <= 0:
            raise ValueError(
                "Feature count must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "Hidden size must be greater than zero."
            )

        if second_hidden_size <= 0:
            raise ValueError(
                "Second hidden size must be greater than zero."
            )

        if class_count <= 1:
            raise ValueError(
                "Class count must be greater than one."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Dropout must be between 0.0 and 1.0."
            )

        self.sequence_length = sequence_length
        self.feature_count = feature_count

        input_size = (
            sequence_length
            * feature_count
        )

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                second_hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                second_hidden_size,
                class_count,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return class logits for one batch of sequences."""

        if inputs.ndim != 3:
            raise ValueError(
                "Model input must have shape "
                "(batch_size, sequence_length, feature_count)."
            )

        if inputs.shape[1] != self.sequence_length:
            raise ValueError(
                "Input sequence length does not match "
                "the configured model sequence length."
            )

        if inputs.shape[2] != self.feature_count:
            raise ValueError(
                "Input feature count does not match "
                "the configured model feature count."
            )

        return self.network(
            inputs
        )