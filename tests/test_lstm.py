"""Tests for the LSTM financial-stress model."""

import pytest
import torch

from deep_learning_critical_systems.models.lstm import (
    FinancialStressLSTM,
)


def test_lstm_returns_correct_output_shape():
    """The model should return one logit per class and sample."""

    model = FinancialStressLSTM()

    inputs = torch.zeros(
        16,
        60,
        9,
        dtype=torch.float32,
    )

    outputs = model(
        inputs
    )

    assert outputs.shape == (
        16,
        3,
    )


def test_lstm_output_is_float_tensor():
    """Model logits should be floating-point tensors."""

    model = FinancialStressLSTM()

    inputs = torch.zeros(
        8,
        60,
        9,
        dtype=torch.float32,
    )

    outputs = model(
        inputs
    )

    assert outputs.dtype == torch.float32


def test_lstm_supports_different_batch_sizes():
    """The batch dimension should remain flexible."""

    model = FinancialStressLSTM()

    for batch_size in [
        1,
        7,
        32,
    ]:
        inputs = torch.zeros(
            batch_size,
            60,
            9,
            dtype=torch.float32,
        )

        outputs = model(
            inputs
        )

        assert outputs.shape == (
            batch_size,
            3,
        )


def test_lstm_supports_different_sequence_lengths():
    """LSTM should process different temporal window lengths."""

    model = FinancialStressLSTM()

    for sequence_length in [
        10,
        30,
        60,
    ]:
        inputs = torch.zeros(
            4,
            sequence_length,
            9,
            dtype=torch.float32,
        )

        outputs = model(
            inputs
        )

        assert outputs.shape == (
            4,
            3,
        )


def test_lstm_rejects_two_dimensional_input():
    """Input must contain batch, time and feature dimensions."""

    model = FinancialStressLSTM()

    invalid_inputs = torch.zeros(
        16,
        540,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="batch_size, sequence_length, feature_count",
    ):
        model(
            invalid_inputs
        )


def test_lstm_rejects_wrong_feature_count():
    """The feature dimension must match the model configuration."""

    model = FinancialStressLSTM(
        feature_count=9,
    )

    invalid_inputs = torch.zeros(
        8,
        60,
        5,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="feature count",
    ):
        model(
            invalid_inputs
        )


def test_lstm_rejects_invalid_configuration():
    """Invalid architecture parameters should fail immediately."""

    with pytest.raises(
        ValueError,
        match="Feature count",
    ):
        FinancialStressLSTM(
            feature_count=0,
        )

    with pytest.raises(
        ValueError,
        match="Hidden size",
    ):
        FinancialStressLSTM(
            hidden_size=0,
        )

    with pytest.raises(
        ValueError,
        match="Classifier hidden size",
    ):
        FinancialStressLSTM(
            classifier_hidden_size=0,
        )

    with pytest.raises(
        ValueError,
        match="Class count",
    ):
        FinancialStressLSTM(
            class_count=1,
        )

    with pytest.raises(
        ValueError,
        match="Dropout",
    ):
        FinancialStressLSTM(
            dropout=1.0,
        )


def test_lstm_has_expected_trainable_parameter_count():
    """The default LSTM architecture should remain reproducible."""

    model = FinancialStressLSTM()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count == 21379