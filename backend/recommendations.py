"""Similarity calculations for library recommendations."""

import math
from typing import Any

from backend.models import GENRE_NAMES
from backend.preprocessing import AUDIO_FEATURE_NAMES

AUDIO_WEIGHT = 0.7
GENRE_WEIGHT = 0.3


def _finite_vector(values: dict[str, Any], names: tuple[str, ...]) -> list[float] | None:
    try:
        vector = [float(values[name]) for name in names]
    except (KeyError, TypeError, ValueError):
        return None
    if not vector or not all(math.isfinite(value) for value in vector):
        return None
    return vector


def normalized_probability_vector(values: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(values, dict):
        return None
    if set(values) != set(GENRE_NAMES):
        return None
    vector = _finite_vector(values, tuple(GENRE_NAMES))
    if vector is None or any(value < 0 for value in vector):
        return None
    total = sum(vector)
    if not math.isfinite(total) or total <= 0:
        return None
    return [value / total for value in vector]


def genre_similarity(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> float | None:
    first_vector = normalized_probability_vector(first)
    second_vector = normalized_probability_vector(second)
    if first_vector is None or second_vector is None:
        return None
    numerator = sum(left * right for left, right in zip(first_vector, second_vector))
    first_norm = math.sqrt(sum(value * value for value in first_vector))
    second_norm = math.sqrt(sum(value * value for value in second_vector))
    if first_norm == 0 or second_norm == 0:
        return None
    return max(0.0, min(1.0, numerator / (first_norm * second_norm)))


def audio_similarity(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
    reference_features: list[dict[str, Any]],
) -> float | None:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    first_vector = _finite_vector(first, AUDIO_FEATURE_NAMES)
    second_vector = _finite_vector(second, AUDIO_FEATURE_NAMES)
    reference_vectors = [
        _finite_vector(features, AUDIO_FEATURE_NAMES)
        for features in reference_features
        if isinstance(features, dict)
    ]
    reference_vectors = [vector for vector in reference_vectors if vector is not None]
    if first_vector is None or second_vector is None or not reference_vectors:
        return None

    means = [
        sum(vector[index] for vector in reference_vectors) / len(reference_vectors)
        for index in range(len(AUDIO_FEATURE_NAMES))
    ]
    standard_deviations = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in reference_vectors) / len(reference_vectors)
        standard_deviations.append(math.sqrt(variance) or 1.0)

    distance = math.sqrt(sum(
        ((left - mean) / deviation - (right - mean) / deviation) ** 2
        for left, right, mean, deviation in zip(
            first_vector, second_vector, means, standard_deviations
        )
    ) / len(AUDIO_FEATURE_NAMES))
    return max(0.0, min(1.0, 1.0 / (1.0 + distance)))


def combined_score(audio_score: float | None, genre_score: float | None) -> tuple[float, float, float]:
    try:
        usable_audio = float(audio_score) if audio_score is not None else 0.0
    except (TypeError, ValueError):
        usable_audio = 0.0
    try:
        usable_genre = float(genre_score) if genre_score is not None else 0.0
    except (TypeError, ValueError):
        usable_genre = 0.0
    if not math.isfinite(usable_audio):
        usable_audio = 0.0
    if not math.isfinite(usable_genre):
        usable_genre = 0.0
    score = AUDIO_WEIGHT * usable_audio + GENRE_WEIGHT * usable_genre
    return (
        max(0.0, min(1.0, score)),
        max(0.0, min(1.0, usable_audio)),
        max(0.0, min(1.0, usable_genre)),
    )


def rank_recommendations(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_features = [
        song["audio_features"]
        for song in [source, *candidates]
        if song.get("feature_status") == "ready" and song.get("audio_features")
    ]
    ranked = []
    for candidate in candidates:
        if candidate.get("id") == source.get("id"):
            continue
        audio_score = audio_similarity(
            source.get("audio_features"),
            candidate.get("audio_features"),
            reference_features,
        )
        genre_score = genre_similarity(
            source.get("probabilities"),
            candidate.get("probabilities"),
        )
        score, bounded_audio, bounded_genre = combined_score(audio_score, genre_score)
        components = []
        if audio_score is not None:
            components.append(f"audio profile ({bounded_audio:.2f})")
        if genre_score is not None:
            components.append(f"genre distribution ({bounded_genre:.2f})")
        ranked.append({
            "song_id": candidate["id"],
            "filename": candidate["original_filename"],
            "predicted_genre": candidate["predicted_genre"],
            "confidence": candidate["prediction_confidence"],
            "score": score,
            "audio_similarity": bounded_audio,
            "genre_similarity": bounded_genre,
            "explanation": "Recommended using " + " and ".join(components)
            if components else "Insufficient similarity components; score is conservative.",
        })
    return sorted(ranked, key=lambda recommendation: recommendation["score"], reverse=True)
