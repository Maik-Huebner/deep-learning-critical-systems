"""Transformer model for financial-stress classification."""

from __future__ import annotations

import math

import torch
from torch import nn


DEFAULT_FEATURE_COUNT = 9
DEFAULT_MODEL_DIMENSION = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_FEED_FORWARD_SIZE = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_CLASSIFIER_HIDDEN_SIZE = 32
DEFAULT_CLASS_COUNT = 3
DEFAULT_DROPOUT = 0.20
DEFAULT_MAX_SEQUENCE_LENGTH = 512


class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal position information to sequence embeddings.

    Self-attention does not know the chronological position of an
    observation by itself. Positional encoding therefore adds a unique
    position-dependent signal to every timestep.
    """

    def __init__(
        self,
        model_dimension: int,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> None:
        """Create sinusoidal positional encodings."""

        super().__init__()

        if model_dimension <= 0:
            raise ValueError(
                "Model dimension must be greater than zero."
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "Maximum sequence length must be greater than zero."
            )

        position = torch.arange(
            max_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        divisor = torch.exp(
            torch.arange(
                0,
                model_dimension,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(
                    10000.0
                )
                / model_dimension
            )
        )

        encoding = torch.zeros(
            max_sequence_length,
            model_dimension,
            dtype=torch.float32,
        )

        encoding[
            :,
            0::2,
        ] = torch.sin(
            position
            * divisor
        )

        if model_dimension > 1:
            encoding[
                :,
                1::2,
            ] = torch.cos(
                position
                * divisor[
                    :encoding[
                        :,
                        1::2,
                    ].shape[1]
                ]
            )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(
                0
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Add positional information to one batch of embeddings."""

        if inputs.ndim != 3:
            raise ValueError(
                "Positional encoding input must have shape "
                "(batch_size, sequence_length, model_dimension)."
            )

        sequence_length = inputs.shape[
            1
        ]

        if sequence_length > self.encoding.shape[
            1
        ]:
            raise ValueError(
                "Input sequence is longer than the configured "
                "maximum sequence length."
            )

        return (
            inputs
            + self.encoding[
                :,
                :sequence_length,
                :,
            ]
        )


class TransformerEncoderBlock(nn.Module):
    """One Transformer encoder block.

    Structure:

        Self-Attention
            ↓
        Residual connection
            ↓
        Layer Normalization
            ↓
        Feed-Forward Network
            ↓
        Residual connection
            ↓
        Layer Normalization
    """

    def __init__(
        self,
        model_dimension: int,
        num_heads: int,
        feed_forward_size: int,
        dropout: float,
    ) -> None:
        """Initialize one encoder block."""

        super().__init__()

        if model_dimension <= 0:
            raise ValueError(
                "Model dimension must be greater than zero."
            )

        if num_heads <= 0:
            raise ValueError(
                "Number of attention heads must be greater than zero."
            )

        if model_dimension % num_heads != 0:
            raise ValueError(
                "Model dimension must be divisible "
                "by the number of attention heads."
            )

        if feed_forward_size <= 0:
            raise ValueError(
                "Feed-forward size must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Dropout must be between 0.0 and 1.0."
            )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dimension,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.attention_norm = nn.LayerNorm(
            model_dimension
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                model_dimension,
                feed_forward_size,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                feed_forward_size,
                model_dimension,
            ),
        )

        self.feed_forward_dropout = nn.Dropout(
            dropout
        )

        self.feed_forward_norm = nn.LayerNorm(
            model_dimension
        )

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Process one sequence through attention and feed-forward layers."""

        (
            attention_output,
            attention_weights,
        ) = self.self_attention(
            inputs,
            inputs,
            inputs,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        attention_output = (
            self.attention_dropout(
                attention_output
            )
        )

        hidden = self.attention_norm(
            inputs
            + attention_output
        )

        feed_forward_output = (
            self.feed_forward(
                hidden
            )
        )

        feed_forward_output = (
            self.feed_forward_dropout(
                feed_forward_output
            )
        )

        output = self.feed_forward_norm(
            hidden
            + feed_forward_output
        )

        return (
            output,
            attention_weights,
        )


class FinancialStressTransformer(nn.Module):
    """Transformer classifier for OFR financial-stress sequences.

    Input shape:

        (batch_size, sequence_length, feature_count)

    The nine raw OFR features are first projected into a larger model
    dimension. Positional encoding provides chronological information.

    Transformer encoder blocks then model relationships between all
    observations inside the historical window.

    The final sequence representation is mean-pooled before the
    classification head produces three class logits.
    """

    def __init__(
        self,
        feature_count: int = DEFAULT_FEATURE_COUNT,
        model_dimension: int = DEFAULT_MODEL_DIMENSION,
        num_heads: int = DEFAULT_NUM_HEADS,
        feed_forward_size: int = DEFAULT_FEED_FORWARD_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        classifier_hidden_size: int = DEFAULT_CLASSIFIER_HIDDEN_SIZE,
        class_count: int = DEFAULT_CLASS_COUNT,
        dropout: float = DEFAULT_DROPOUT,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> None:
        """Initialize the financial-stress Transformer."""

        super().__init__()

        if feature_count <= 0:
            raise ValueError(
                "Feature count must be greater than zero."
            )

        if model_dimension <= 0:
            raise ValueError(
                "Model dimension must be greater than zero."
            )

        if num_heads <= 0:
            raise ValueError(
                "Number of attention heads must be greater than zero."
            )

        if model_dimension % num_heads != 0:
            raise ValueError(
                "Model dimension must be divisible "
                "by the number of attention heads."
            )

        if feed_forward_size <= 0:
            raise ValueError(
                "Feed-forward size must be greater than zero."
            )

        if num_layers <= 0:
            raise ValueError(
                "Number of encoder layers must be greater than zero."
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
        self.model_dimension = model_dimension

        self.input_projection = nn.Linear(
            feature_count,
            model_dimension,
        )

        self.positional_encoding = (
            SinusoidalPositionalEncoding(
                model_dimension=model_dimension,
                max_sequence_length=max_sequence_length,
            )
        )

        self.input_dropout = nn.Dropout(
            dropout
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    model_dimension=model_dimension,
                    num_heads=num_heads,
                    feed_forward_size=feed_forward_size,
                    dropout=dropout,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                model_dimension,
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

    def encode(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """Encode a time-series window and optionally return attention."""

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

        hidden = self.input_projection(
            inputs
        )

        hidden = self.positional_encoding(
            hidden
        )

        hidden = self.input_dropout(
            hidden
        )

        attention_maps = []

        for encoder_block in self.encoder_blocks:
            (
                hidden,
                attention_weights,
            ) = encoder_block(
                hidden,
                return_attention=return_attention,
            )

            if (
                return_attention
                and attention_weights is not None
            ):
                attention_maps.append(
                    attention_weights
                )

        return (
            hidden,
            attention_maps,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return class logits for one batch."""

        hidden, _ = self.encode(
            inputs,
            return_attention=False,
        )

        pooled = hidden.mean(
            dim=1
        )

        logits = self.classifier(
            pooled
        )

        return logits

    def forward_with_attention(
        self,
        inputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """Return class logits together with attention maps."""

        (
            hidden,
            attention_maps,
        ) = self.encode(
            inputs,
            return_attention=True,
        )

        pooled = hidden.mean(
            dim=1
        )

        logits = self.classifier(
            pooled
        )

        return (
            logits,
            attention_maps,
        )