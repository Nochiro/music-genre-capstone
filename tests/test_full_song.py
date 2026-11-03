import numpy as np
import pytest

from backend.main import GENRE_NAMES, get_full_song_prediction
from backend.preprocessing import create_audio_windows


SR = 10
WINDOW_SAMPLES = 30 * SR


def test_short_audio_creates_one_padded_window():
    windows = create_audio_windows(np.ones(5 * SR), sr=SR)

    assert len(windows) == 1
    assert windows[0][0] == 0
    assert windows[0][1] == 5
    assert len(windows[0][2]) == WINDOW_SAMPLES
    assert np.all(windows[0][2][5 * SR:] == 0)


def test_long_audio_uses_overlapping_windows_and_covers_end():
    windows = create_audio_windows(np.ones(46 * SR), sr=SR)

    assert [(start, end) for start, end, _ in windows] == [
        (0, 30),
        (15, 45),
        (30, 46),
    ]
    assert all(len(window) == WINDOW_SAMPLES for _, _, window in windows)
    assert np.all(windows[-1][2][:16 * SR] == 1)
    assert np.all(windows[-1][2][16 * SR:] == 0)


def test_full_song_prediction_averages_window_probabilities(monkeypatch):
    calls = []

    def fake_prediction(model_name, audio):
        calls.append(audio)
        dominant_index = 0 if audio[0] == 0 else 1
        probabilities = [0.0] * len(GENRE_NAMES)
        probabilities[dominant_index] = 0.8
        probabilities[1 - dominant_index] = 0.2
        return {
            "model": model_name,
            "genre": GENRE_NAMES[dominant_index],
            "confidence": 0.8,
            "probabilities": {
                genre: probabilities[index]
                for index, genre in enumerate(GENRE_NAMES)
            },
        }

    monkeypatch.setattr("backend.main.get_prediction", fake_prediction)

    audio = np.arange(46 * SR, dtype=float)
    result = get_full_song_prediction("resnet", audio, sr=SR)

    assert len(calls) == 3
    assert len(result["windows"]) == 3
    assert result["genre"] == GENRE_NAMES[1]
    assert result["probabilities"][GENRE_NAMES[0]] == pytest.approx((0.8 + 0.2 + 0.2) / 3)
    assert result["probabilities"][GENRE_NAMES[1]] == pytest.approx((0.2 + 0.8 + 0.8) / 3)
    assert result["windows"][-1]["start_seconds"] == 30
    assert result["windows"][-1]["end_seconds"] == 46
