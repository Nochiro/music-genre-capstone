"""LSTM sequence model for music genre classification (Expt 6, adapted).

Processes sequences of spectrogram frames across time.
"""

import torch
import torch.nn as nn


class GenreLSTM(nn.Module):
    """LSTM-based model for genre classification from frame-level feature sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the LSTM model.

        Args:
            input_dim: Dimensionality of each frame's feature vector.
            hidden_dim: Number of hidden units in the LSTM.
            num_layers: Number of stacked LSTM layers.
            num_classes: Number of genre classes.
            bidirectional: If True, use a bidirectional LSTM.
            dropout: Dropout probability between LSTM layers.
        """
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        ...
