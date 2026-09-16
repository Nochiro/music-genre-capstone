"""Extract and cache audio features for downstream models.

Supports three feature representations used by different model stages:

* **MFCC** (aggregated) — for the baseline ANN.
* **Mel-spectrogram** (2-D image) — for CNN and transfer-learning models.
* **Frame-level sequences** — for the LSTM sequence model.

All outputs are saved to Google Drive so feature extraction only needs to
run once (see ``batch_extract_features``).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np


def extract_mfcc(
    audio_path: str,
    sr: int = 22050,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Extract MFCC features from a single audio file.

    Args:
        audio_path: Path to the audio file (MP3/WAV).
        sr: Target sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        NumPy array of aggregated MFCC features.
    """
    ...


def extract_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
    n_mels: int = 128,
) -> np.ndarray:
    """Extract a mel-spectrogram from a single audio file.

    Args:
        audio_path: Path to the audio file.
        sr: Target sample rate.
        n_mels: Number of mel-frequency bins.

    Returns:
        2-D NumPy array (n_mels × time_frames).
    """
    ...


def extract_frame_sequences(
    audio_path: str,
    sr: int = 22050,
) -> np.ndarray:
    """Extract frame-level feature sequences for LSTM input.

    Args:
        audio_path: Path to the audio file.
        sr: Target sample rate.

    Returns:
        2-D NumPy array (time_steps × feature_dim).
    """
    ...


def batch_extract_features(
    file_list: List[str],
    output_dir: str,
    feature_type: str = "mfcc",
) -> None:
    """Batch-extract features for a list of audio files and cache to disk.

    Args:
        file_list: Paths to audio files.
        output_dir: Directory on Google Drive to save extracted features.
        feature_type: One of ``"mfcc"``, ``"mel_spectrogram"``, or
                      ``"frame_sequences"``.
    """
    ...
