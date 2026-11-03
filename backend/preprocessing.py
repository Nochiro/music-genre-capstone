"""
Audio preprocessing for music genre classification.
Reproduces exact pipeline from training notebooks.
"""

import numpy as np
import librosa
import torch
from typing import Tuple


AUDIO_FEATURE_NAMES = (
    "tempo",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "zero_crossing_rate_mean",
    *[f"mfcc_{index}_mean" for index in range(1, 14)],
    *[f"mfcc_{index}_std" for index in range(1, 14)],
)


def load_audio(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Load the complete audio file as a mono waveform.

    Args:
        audio_path: Path to audio file
        sr: Sample rate (Hz)
    Returns:
        Complete audio time series (mono), shape (num_samples,)

    Raises:
        Exception: If audio cannot be loaded
    """
    try:
        # Load as mono
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        return y

    except Exception as e:
        raise Exception(f"Failed to load audio from {audio_path}: {str(e)}")


def create_audio_windows(
    audio: np.ndarray,
    sr: int = 22050,
    window_seconds: float = 30.0,
    stride_seconds: float = 15.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Split complete audio into overlapping, model-sized windows.

    Windows use a 15-second stride by default, so adjacent full windows
    overlap by 50%. A short song produces one padded window. For a longer
    song, a final partial stride window is added when needed so no trailing
    audio is discarded; only that final window may contain zero padding.

    Returns tuples of ``(start_seconds, end_seconds, audio_window)``. The end
    timestamp describes the original audio and therefore excludes any padding.
    """
    if audio.ndim != 1:
        raise ValueError("Audio must be a one-dimensional mono waveform.")
    if sr <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("Sample rate, window size, and stride must be positive.")

    window_samples = int(round(sr * window_seconds))
    stride_samples = int(round(sr * stride_seconds))
    audio_length = len(audio)

    if audio_length <= window_samples:
        starts = [0]
    else:
        starts = list(range(0, audio_length - window_samples + 1, stride_samples))
        last_full_end = starts[-1] + window_samples
        next_start = starts[-1] + stride_samples
        if last_full_end < audio_length:
            starts.append(next_start)

    windows = []
    for start_sample in starts:
        end_sample = min(start_sample + window_samples, audio_length)
        window = audio[start_sample:end_sample]
        if len(window) < window_samples:
            window = np.pad(
                window,
                (0, window_samples - len(window)),
                mode="constant",
                constant_values=0,
            )

        start_seconds = start_sample / sr
        end_seconds = end_sample / sr
        windows.append((start_seconds, end_seconds, window))

    return windows


def _finite_mean(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.mean(finite_values))


def extract_audio_features(
    audio: np.ndarray,
    sr: int = 22050,
) -> dict[str, float]:
    """Extract an aggregated full-song audio descriptor.

    Each overlapping Phase 1 window is analyzed independently. The final
    partial window is trimmed back to its original audio before feature
    extraction, so zero padding cannot become a similarity signal. Features
    are averaged across windows and are intended for relative retrieval, not
    as ground-truth labels.
    """
    window_features = []
    for start_seconds, end_seconds, audio_window in create_audio_windows(audio, sr=sr):
        actual_samples = max(1, int(round((end_seconds - start_seconds) * sr)))
        window_features.append(_extract_window_features(audio_window[:actual_samples], sr))

    if not window_features:
        raise ValueError("Audio did not produce any feature windows.")

    return {
        name: float(np.mean([features[name] for features in window_features]))
        for name in AUDIO_FEATURE_NAMES
    }


def _extract_window_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    if audio.size == 0:
        raise ValueError("Audio window is empty.")

    try:
        tempo = librosa.feature.tempo(y=audio, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    except Exception as error:
        raise ValueError(f"Audio feature extraction failed: {error}") from error

    values = {
        "tempo": _finite_mean(np.asarray(tempo)),
        "spectral_centroid_mean": _finite_mean(spectral_centroid),
        "spectral_bandwidth_mean": _finite_mean(spectral_bandwidth),
        "spectral_rolloff_mean": _finite_mean(spectral_rolloff),
        "zero_crossing_rate_mean": _finite_mean(zero_crossing_rate),
    }
    values.update({
        f"mfcc_{index}_mean": _finite_mean(mfcc[index - 1])
        for index in range(1, 14)
    })
    values.update({
        f"mfcc_{index}_std": float(np.std(mfcc[index - 1]))
        if np.isfinite(mfcc[index - 1]).any() else 0.0
        for index in range(1, 14)
    })
    return values


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
