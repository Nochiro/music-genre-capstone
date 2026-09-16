"""Baseline ANN for music genre classification (Expt 2).

Simple feedforward network on aggregated MFCC features, logged with MLflow.
"""

from typing import List

import torch
import torch.nn as nn


class BaselineANN(nn.Module):
    """Simple feedforward ANN for genre classification from aggregated features."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None) -> None:
        """Initialize the baseline ANN.

        Args:
            input_dim: Dimensionality of the input feature vector.
            num_classes: Number of genre classes.
            hidden_dims: List of hidden layer sizes.
        """
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        ...
