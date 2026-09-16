"""Transfer learning model for music genre classification (Expt 4).

Wraps pretrained ResNet/VGG for fine-tuning vs. feature extraction comparison.
"""

import torch
import torch.nn as nn


class TransferModel(nn.Module):
    """Transfer learning wrapper around pretrained CNN backbones."""

    def __init__(self, num_classes: int, backbone: str = "resnet18", freeze_backbone: bool = False) -> None:
        """Initialize the transfer learning model.

        Args:
            num_classes: Number of genre classes.
            backbone: Pretrained model name ('resnet18', 'resnet50', 'vgg16', etc.).
            freeze_backbone: If True, freeze backbone weights (feature extraction mode);
                if False, fine-tune the entire network.
        """
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W) — backbone expects 3-channel input.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        ...
