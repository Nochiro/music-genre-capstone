"""Benchmarking utilities to measure latency vs. accuracy tradeoff (Expt 7).

Provides functions to time inference, evaluate accuracy on a test set,
and generate a comparison report across model variants (original,
quantized, pruned).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure average inference latency over multiple runs.

    Args:
        model: PyTorch model to benchmark.
        input_shape: Shape of a single input tensor (e.g. (1, 1, 128, 128)).
        num_runs: Number of forward passes to average over.
        device: Device to run inference on.

    Returns:
        Dict with mean_ms, std_ms, and median_ms.
    """
    ...


def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model accuracy on a test set.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for the test split.
        device: Device to run evaluation on.

    Returns:
        Dict with accuracy, top3_accuracy, and per-class metrics.
    """
    ...


def generate_report(
    results_dict: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> str:
    """Generate a latency-vs-accuracy comparison report.

    Args:
        results_dict: Mapping of model variant name to its benchmark
            results (latency + accuracy dicts merged).
        output_path: If provided, write the report to this file.

    Returns:
        Formatted report string.
    """
    ...
