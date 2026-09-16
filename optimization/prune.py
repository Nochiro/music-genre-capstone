"""Structured and unstructured pruning for model optimization (Expt 7).

Provides utilities to prune trained models and make pruning permanent
for deployment.
"""

import torch.nn as nn


def prune_unstructured(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply global unstructured pruning to the model.

    Args:
        model: Trained PyTorch model.
        amount: Fraction of parameters to prune (0.0–1.0).

    Returns:
        Pruned model (with pruning reparametrization hooks).
    """
    ...


def prune_structured(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply structured (channel-level) pruning to the model.

    Args:
        model: Trained PyTorch model.
        amount: Fraction of channels to prune (0.0–1.0).

    Returns:
        Pruned model (with pruning reparametrization hooks).
    """
    ...


def remove_pruning_reparametrization(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing reparametrization hooks.

    Converts the pruned mask into the actual weight tensor so the model
    can be saved and deployed without the pruning forward hooks.

    Args:
        model: Pruned model with active reparametrization.

    Returns:
        Model with pruning baked in (no hooks).
    """
    ...
