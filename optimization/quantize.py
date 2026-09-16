"""Post-training quantization for model optimization (Expt 7).

Applies dynamic and static quantization to trained models and compares
the resulting model sizes against the original.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Apply dynamic quantization to the model.

    Args:
        model: Trained PyTorch model.
        dtype: Target quantized dtype (default: torch.qint8).

    Returns:
        Dynamically quantized model.
    """
    ...


def quantize_static(model: nn.Module, calibration_loader: DataLoader) -> nn.Module:
    """Apply static quantization with a calibration dataset.

    Args:
        model: Trained PyTorch model (must define quant/dequant stubs).
        calibration_loader: DataLoader used for calibration.

    Returns:
        Statically quantized model.
    """
    ...


def compare_model_sizes(original: nn.Module, quantized: nn.Module) -> Dict[str, float]:
    """Compare file sizes of original and quantized models.

    Args:
        original: Original (unquantized) model.
        quantized: Quantized model.

    Returns:
        Dict with original_size_mb, quantized_size_mb, and compression_ratio.
    """
    ...
