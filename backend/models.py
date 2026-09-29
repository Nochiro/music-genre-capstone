"""
Model definitions for music genre classification.
Exact architectures from training notebooks.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Tuple


GENRE_NAMES = [
    "Blues",
    "Classical",
    "Country",
    "Easy Listening",
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Jazz",
    "Old-Time / Historic",
    "Pop",
    "Rock",
    "Soul-RnB",
    "Spoken"
]

NUM_CLASSES = 16
assert len(GENRE_NAMES) == NUM_CLASSES


class GenreCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

class GenreLSTM(nn.Module):
    """
    LSTM for music genre classification from mel spectrograms.
    Architecture from 05_lstm_sequence.ipynb training notebooks.

    Input: (batch, seq_len, features) = (batch, 322, 128)
    Output: (batch, 16) — logits for 16 genres
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        batch_first: bool = True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len=322, features=128)
        Returns:
            (batch, 16) logits
        """
        # LSTM forward: output shape (batch, seq_len, hidden_size)
        output, (hidden, cell) = self.lstm(x)

        # Take last time step
        x = output[:, -1, :]  # (batch, hidden_size)

        # Classifier
        x = self.classifier(x)
        return x


class GenreResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        from torchvision.models import resnet18

        model = resnet18(weights=None)

        # Copy the ResNet18 structure
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def create_cnn(num_classes: int = NUM_CLASSES) -> GenreCNN:
    """Create CNN model."""
    return GenreCNN(num_classes=num_classes)


def create_lstm(num_classes: int = NUM_CLASSES) -> GenreLSTM:
    """Create LSTM model."""
    return GenreLSTM(
        input_size=128,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        batch_first=True
    )


def create_resnet(num_classes: int = NUM_CLASSES) -> GenreResNet:
    """Create ResNet18 model."""
    return GenreResNet(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
