import math

from backend.models import GENRE_NAMES
from backend.recommendations import (
    AUDIO_WEIGHT,
    GENRE_WEIGHT,
    audio_similarity,
    combined_score,
    genre_similarity,
    rank_recommendations,
)
from backend.preprocessing import AUDIO_FEATURE_NAMES


def make_features(value: float) -> dict[str, float]:
    return {name: value for name in AUDIO_FEATURE_NAMES}


def make_probabilities(genre: str, value: float = 1.0) -> dict[str, float]:
    return {name: value if name == genre else 0.0 for name in GENRE_NAMES}


def song(song_id: str, value: float, genre: str = "Rock") -> dict:
    return {
        "id": song_id,
        "original_filename": f"{song_id}.wav",
        "predicted_genre": genre,
        "prediction_confidence": 0.8,
        "probabilities": make_probabilities(genre),
        "audio_features": make_features(value),
        "feature_status": "ready",
    }


def test_similarity_components_are_bounded_and_malformed_vectors_are_safe():
    first = song("first", 1.0)
    second = song("second", 1.0)

    assert audio_similarity(first["audio_features"], second["audio_features"], [first["audio_features"], second["audio_features"]]) == 1.0
    assert genre_similarity(first["probabilities"], second["probabilities"]) == 1.0
    assert genre_similarity({"Rock": "invalid"}, second["probabilities"]) is None
    extra_probability = dict(second["probabilities"])
    extra_probability["unexpected"] = 0.0
    assert genre_similarity(extra_probability, second["probabilities"]) is None
    assert genre_similarity({"Rock": 1.0}, second["probabilities"]) is None
    assert genre_similarity({genre: 0.0 for genre in GENRE_NAMES}, second["probabilities"]) is None
    assert genre_similarity({genre: float("nan") for genre in GENRE_NAMES}, second["probabilities"]) is None
    assert combined_score(None, None) == (0.0, 0.0, 0.0)
    assert combined_score(float("nan"), float("inf")) == (0.0, 0.0, 0.0)
    assert combined_score("invalid", "invalid") == (0.0, 0.0, 0.0)


def test_ranking_is_descending_and_excludes_source():
    source = song("source", 1.0)
    close = song("close", 1.0)
    far = song("far", 10.0, genre="Jazz")

    ranked = rank_recommendations(source, [source, far, close])

    assert [item["song_id"] for item in ranked] == ["close", "far"]
    assert all(0.0 <= item["score"] <= 1.0 for item in ranked)
    assert ranked[0]["score"] >= ranked[1]["score"]
    assert math.isclose(AUDIO_WEIGHT + GENRE_WEIGHT, 1.0)


def test_audio_similarity_decreases_with_feature_distance():
    source = make_features(0.0)
    close = make_features(1.0)
    far = make_features(10.0)
    references = [source, close, far]

    assert audio_similarity(source, source, references) == 1.0
    assert audio_similarity(source, close, references) > audio_similarity(source, far, references)
