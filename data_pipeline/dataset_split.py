"""Create stratified train / validation / test splits and compute class weights.

Loads cached features produced by :mod:`data_pipeline.feature_extraction`,
splits them with stratification to preserve genre proportions, and computes
class weights for handling imbalanced genre distributions.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def load_cached_features(feature_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously extracted and cached features from Google Drive.

    Args:
        feature_dir: Directory containing cached feature files.

    Returns:
        Tuple of ``(features, labels)`` as NumPy arrays.
    """
    ...


def create_splits(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Create stratified train / validation / test splits.

    Args:
        features: Feature matrix.
        labels: Corresponding label array.
        test_size: Fraction of data reserved for the test set.
        val_size: Fraction of data reserved for the validation set.
        random_state: Random seed for reproducibility.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``
    """
    ...


def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute per-class weights inversely proportional to class frequency.

    Useful for the class-weighted loss in the CNN stage.

    Args:
        labels: Array of integer class labels.

    Returns:
        Dictionary mapping each class index to its weight.
    """
    ...
