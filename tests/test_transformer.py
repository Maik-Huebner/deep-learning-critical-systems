"""Tests for the financial-stress Transformer model."""

import pytest
import torch

from deep_learning_critical_systems.models.transformer import (
    FinancialStressTransformer,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)


def test_positional_encoding_preserves_shape():
    """Positional encoding must not change the tensor dimensions."""

    encoding = SinusoidalPositionalEncoding(
        model_dimension=64,
        max_sequence_length=100,
    )

    inputs = torch.zeros(
        8,
        60,
        64,
        dtype=torch.float32,
    )

    outputs = encoding(
        inputs
    )

    assert outputs.shape == (
        8,
        60,
        64,
    )


def test_positional_encoding_changes_zero_input():
    """Different positions should receive positional information."""

    encoding = SinusoidalPositionalEncoding(
        model_dimension=64,
        max_sequence_length=100,
    )

    inputs = torch.zeros(
        1,
        60,
        64,
        dtype=torch.float32,
    )

    outputs = encoding(
        inputs
    )

    assert not torch.all(
        outputs == 0
    )

    assert not torch.equal(
        outputs[:, 0, :],
        outputs[:, 1, :],
    )


def test_positional_encoding_rejects_long_sequence():
    """Sequences longer than the configured maximum must fail."""

    encoding = SinusoidalPositionalEncoding(
        model_dimension=64,
        max_sequence_length=20,
    )

    inputs = torch.zeros(
        2,
        21,
        64,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="longer",
    ):
        encoding(
            inputs
        )


def test_encoder_block_preserves_shape():
    """An encoder block must preserve sequence and embedding dimensions."""

    block = TransformerEncoderBlock(
        model_dimension=64,
        num_heads=4,
        feed_forward_size=128,
        dropout=0.2,
    )

    inputs = torch.zeros(
        8,
        60,
        64,
        dtype=torch.float32,
    )

    outputs, attention = block(
        inputs,
        return_attention=False,
    )

    assert outputs.shape == (
        8,
        60,
        64,
    )

    assert attention is None


def test_encoder_block_returns_attention_weights():
    """Attention weights should be available for interpretation."""

    block = TransformerEncoderBlock(
        model_dimension=64,
        num_heads=4,
        feed_forward_size=128,
        dropout=0.0,
    )

    inputs = torch.zeros(
        2,
        60,
        64,
        dtype=torch.float32,
    )

    outputs, attention = block(
        inputs,
        return_attention=True,
    )

    assert outputs.shape == (
        2,
        60,
        64,
    )

    assert attention is not None

    assert attention.shape == (
        2,
        4,
        60,
        60,
    )


def test_transformer_returns_correct_output_shape():
    """The Transformer should produce three logits per sample."""

    model = FinancialStressTransformer()

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


def test_transformer_supports_different_batch_sizes():
    """The batch dimension should remain flexible."""

    model = FinancialStressTransformer()

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


def test_transformer_supports_different_sequence_lengths():
    """The Transformer should support windows within its maximum length."""

    model = FinancialStressTransformer()

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


def test_transformer_returns_attention_maps():
    """The model should expose one attention map per encoder layer."""

    model = FinancialStressTransformer(
        num_layers=2,
        num_heads=4,
    )

    inputs = torch.zeros(
        2,
        60,
        9,
        dtype=torch.float32,
    )

    logits, attention_maps = (
        model.forward_with_attention(
            inputs
        )
    )

    assert logits.shape == (
        2,
        3,
    )

    assert len(
        attention_maps
    ) == 2

    for attention_map in attention_maps:
        assert attention_map.shape == (
            2,
            4,
            60,
            60,
        )


def test_transformer_rejects_two_dimensional_input():
    """Input must contain batch, time and feature dimensions."""

    model = FinancialStressTransformer()

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


def test_transformer_rejects_wrong_feature_count():
    """Input feature dimension must match the configuration."""

    model = FinancialStressTransformer(
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


def test_transformer_requires_compatible_attention_heads():
    """Model dimension must be divisible by the head count."""

    with pytest.raises(
        ValueError,
        match="divisible",
    ):
        FinancialStressTransformer(
            model_dimension=64,
            num_heads=3,
        )


def test_transformer_rejects_invalid_configuration():
    """Invalid architecture parameters should fail immediately."""

    with pytest.raises(
        ValueError,
        match="Feature count",
    ):
        FinancialStressTransformer(
            feature_count=0,
        )

    with pytest.raises(
        ValueError,
        match="Model dimension",
    ):
        FinancialStressTransformer(
            model_dimension=0,
        )

    with pytest.raises(
        ValueError,
        match="Number of attention heads",
    ):
        FinancialStressTransformer(
            num_heads=0,
        )

    with pytest.raises(
        ValueError,
        match="Feed-forward size",
    ):
        FinancialStressTransformer(
            feed_forward_size=0,
        )

    with pytest.raises(
        ValueError,
        match="Number of encoder layers",
    ):
        FinancialStressTransformer(
            num_layers=0,
        )

    with pytest.raises(
        ValueError,
        match="Classifier hidden size",
    ):
        FinancialStressTransformer(
            classifier_hidden_size=0,
        )

    with pytest.raises(
        ValueError,
        match="Class count",
    ):
        FinancialStressTransformer(
            class_count=1,
        )

    with pytest.raises(
        ValueError,
        match="Dropout",
    ):
        FinancialStressTransformer(
            dropout=1.0,
        )


def test_transformer_has_expected_parameter_count():
    """The default Transformer architecture should remain reproducible."""

    model = FinancialStressTransformer()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count == 69763