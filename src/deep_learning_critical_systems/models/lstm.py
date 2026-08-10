"""LSTM model for financial-stress classification."""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_FEATURE_COUNT = 9
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_CLASSIFIER_HIDDEN_SIZE = 32
DEFAULT_CLASS_COUNT = 3
DEFAULT_DROPOUT = 0.20


class FinancialStressLSTM(nn.Module):
    """LSTM classifier for OFR financial-stress sequences.

    The model receives time-series windows with shape:

        (batch_size, sequence_length, feature_count)

    Unlike the MLP baseline, the LSTM processes the observations in
    chronological order and can learn temporal relationships within
    the 60-day input window.

    The final LSTM output is passed to a small feed-forward
    classification head that produces one logit for each target class.
    """

    def __init__(
        self,
        feature_count: int = DEFAULT_FEATURE_COUNT,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        classifier_hidden_size: int = DEFAULT_CLASSIFIER_HIDDEN_SIZE,
        class_count: int = DEFAULT_CLASS_COUNT,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Initialize the LSTM classifier."""

        super().__init__()

        if feature_count <= 0:
            raise ValueError(
                "Feature count must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "Hidden size must be greater than zero."
            )

        if classifier_hidden_size <= 0:
            raise ValueError(
                "Classifier hidden size must be greater than zero."
            )

        if class_count <= 1:
            raise ValueError(
                "Class count must be greater than one."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Dropout must be between 0.0 and 1.0."
            )

        self.feature_count = feature_count
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size,
                classifier_hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                classifier_hidden_size,
                class_count,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return class logits for one batch of time-series windows."""

        if inputs.ndim != 3:
            raise ValueError(
                "Model input must have shape "
                "(batch_size, sequence_length, feature_count)."
            )

        if inputs.shape[2] != self.feature_count:
            raise ValueError(
                "Input feature count does not match "
                "the configured model feature count."
            )

        sequence_output, _ = self.lstm(
            inputs
        )

        final_output = sequence_output[
            :,
            -1,
            :,
        ]

        logits = self.classifier(
            final_output
        )

        return logits