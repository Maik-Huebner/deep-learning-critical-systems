"""Tests for the MLP financial-stress baseline."""

import pytest
import torch

from deep_learning_critical_systems.models.mlp import (
    FinancialStressMLP,
)


def test_mlp_returns_correct_output_shape():
    """The model should return one logit per class and sample."""

    model = FinancialStressMLP()

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


def test_mlp_output_is_float_tensor():
    """Model logits should be floating-point tensors."""

    model = FinancialStressMLP()

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


def test_mlp_supports_different_batch_sizes():
    """The batch dimension should remain flexible."""

    model = FinancialStressMLP()

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


def test_mlp_rejects_two_dimensional_input():
    """Input must contain batch, time and feature dimensions."""

    model = FinancialStressMLP()

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


def test_mlp_rejects_wrong_sequence_length():
    """The sequence length must match the model configuration."""

    model = FinancialStressMLP(
        sequence_length=60,
    )

    invalid_inputs = torch.zeros(
        8,
        30,
        9,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="sequence length",
    ):
        model(
            invalid_inputs
        )


def test_mlp_rejects_wrong_feature_count():
    """The feature dimension must match the model configuration."""

    model = FinancialStressMLP(
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


def test_mlp_rejects_invalid_configuration():
    """Invalid architecture parameters should fail immediately."""

    with pytest.raises(
        ValueError,
        match="Sequence length",
    ):
        FinancialStressMLP(
            sequence_length=0,
        )

    with pytest.raises(
        ValueError,
        match="Feature count",
    ):
        FinancialStressMLP(
            feature_count=0,
        )

    with pytest.raises(
        ValueError,
        match="Hidden size",
    ):
        FinancialStressMLP(
            hidden_size=0,
        )

    with pytest.raises(
        ValueError,
        match="Second hidden size",
    ):
        FinancialStressMLP(
            second_hidden_size=0,
        )

    with pytest.raises(
        ValueError,
        match="Class count",
    ):
        FinancialStressMLP(
            class_count=1,
        )

    with pytest.raises(
        ValueError,
        match="Dropout",
    ):
        FinancialStressMLP(
            dropout=1.0,
        )


def test_mlp_has_trainable_parameters():
    """The baseline must contain trainable neural-network parameters."""

    model = FinancialStressMLP()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    assert parameter_count == 77699