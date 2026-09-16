"""CNN for music genre classification from mel-spectrograms (Expt 3).

Handles class imbalance via weighting and SpecAugment-style augmentation.
"""

import torch
import torch.nn as nn


class GenreCNN(nn.Module):
    """Convolutional neural network for genre classification from spectrograms."""

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        """Initialize the CNN.

        Args:
            num_classes: Number of genre classes.
            in_channels: Number of input channels (1 for single-channel spectrograms).
        """
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, freq_bins, time_frames).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        ...
