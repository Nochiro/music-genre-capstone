"""
Audio preprocessing for music genre classification.
Reproduces exact pipeline from training notebooks.
"""

import numpy as np
import librosa
import torch
from typing import Tuple


def load_audio(audio_path: str, sr: int = 22050, duration: float = 30.0) -> np.ndarray:
    """
    Load audio file and enforce 30-second duration.

    Args:
        audio_path: Path to audio file
        sr: Sample rate (Hz)
        duration: Target duration in seconds

    Returns:
        Audio time series (mono), shape (num_samples,)

    Raises:
        Exception: If audio cannot be loaded
    """
    try:
        # Load as mono
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

        # Pad or truncate to exactly 30 seconds
        target_samples = int(sr * duration)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode='constant', constant_values=0)
        elif len(y) > target_samples:
            y = y[:target_samples]

        return y

    except Exception as e:
        raise Exception(f"Failed to load audio from {audio_path}: {str(e)}")


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract mel spectrogram from audio and convert to dB scale.

    Matches training pipeline exactly:
    - librosa.feature.melspectrogram with specified parameters
    - Convert to dB scale (librosa.power_to_db)
    - Range: approximately [-80, 0] dB

    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Number of samples between frames

    Returns:
        Mel spectrogram in dB, shape (n_mels, num_frames)
    """
    # Compute mel spectrogram (power scale)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to dB scale (ref=np.max by default)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_db  # shape: (128, ~1292) for 30-second audio


def preprocess_for_cnn(mel_db):
    mel = torch.tensor(mel_db, dtype=torch.float32)
    mel = mel.unsqueeze(0)
    return mel


def preprocess_for_lstm(mel_db: np.ndarray) -> torch.Tensor:
    """
    Preprocess mel spectrogram for LSTM inference.

    Pipeline from 05_lstm_sequence.ipynb training:
    1. Convert float16 -> float32
    2. Normalize dB range [-80, 0] to [0, 1]
    3. Clamp to [0, 1]
    4. Temporal downsampling: 1292 -> 1288 -> 322 frames
    5. Transpose to (seq_len=322, features=128)

    Args:
        mel_db: Mel spectrogram in dB, shape (128, num_frames)

    Returns:
        Preprocessed tensor for LSTM, shape (322, 128) [unbatched]
    """
    # Convert to float32
    mel = torch.tensor(mel_db, dtype=torch.float32)

    # Normalize [-80, 0] -> [0, 1]
    mel = (mel + 80.0) / 80.0

    # Clamp to [0, 1]
    mel = torch.clamp(mel, 0.0, 1.0)

    # Temporal downsampling: 1292 -> 322 frames
    # Original notebook: truncate to 1288, reshape (128, 1288) -> (128, 322, 4), mean over last dim
    mel = mel[:, :1288]  # Truncate to 1288 frames
    mel = mel.reshape(128, 322, 4).mean(dim=2)  # (128, 322)

    # Transpose to (seq_len, features) for LSTM
    mel = mel.transpose(0, 1)  # (322, 128)

    return mel


def preprocess_for_resnet(mel_db: np.ndarray) -> torch.Tensor:
    """
    Preprocess mel spectrogram for ResNet18 inference.

    Pipeline from 06_optimization.ipynb training:
    1. Convert float16 -> float32
    2. Normalize dB range [-80, 0] to [0, 1]
    3. Clamp to [0, 1]
    4. Add channel dimension: (128, 1292) -> (1, 128, 1292)

    Args:
        mel_db: Mel spectrogram in dB, shape (128, num_frames)

    Returns:
        Preprocessed tensor for ResNet, shape (1, 128, num_frames)
    """
    # Convert to float32
    mel = torch.tensor(mel_db, dtype=torch.float32)

    # Normalize [-80, 0] -> [0, 1]
    mel = (mel + 80.0) / 80.0

    # Clamp to [0, 1]
    mel = torch.clamp(mel, 0.0, 1.0)

    # Add channel dimension
    mel = mel.unsqueeze(0)  # (1, 128, num_frames)

    return mel
