import io
import sqlite3
import shutil
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.library import LibraryStore
from backend.models import GENRE_NAMES


REAL_FULL_SONG_PREDICTION = main.get_full_song_prediction


def make_wav(duration_seconds: float = 1.0) -> bytes:
    sample_rate = 22050
    samples = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * samples)
    return buffer.getvalue()


def fake_analysis(model_name, audio):
    return {
        "model": model_name,
        "genre": "Rock",
        "confidence": 0.75,
        "probabilities": {
            genre: (0.75 if genre == "Rock" else 0.25 / (len(GENRE_NAMES) - 1))
            for genre in GENRE_NAMES
        },
        "windows": [
            {
                "start_seconds": 0.0,
                "end_seconds": 30.0,
                "genre": "Rock",
                "confidence": 0.75,
                "probabilities": {
                    genre: (0.75 if genre == "Rock" else 0.25 / (len(GENRE_NAMES) - 1))
                    for genre in GENRE_NAMES
                },
            }
        ],
    }


@pytest.fixture
def library_environment(tmp_path, monkeypatch):
    store = LibraryStore(tmp_path / "library.db", tmp_path / "audio")
    monkeypatch.setattr(main, "library_store", store)
    monkeypatch.setattr(main, "get_full_song_prediction", fake_analysis)
    monkeypatch.setattr(main, "MAX_LIBRARY_UPLOAD_BYTES", 5 * 1024 * 1024)
    return store


def test_database_persists_after_reinitialization(tmp_path):
    database_path = tmp_path / "library.db"
    audio_directory = tmp_path / "audio"
    store = LibraryStore(database_path, audio_directory)
    store.create_song(
        {
            "id": "song-1",
            "original_filename": "persisted.wav",
            "stored_audio_path": "song-1.wav",
            "uploaded_at": store.timestamp(),
            "predicted_genre": "Rock",
            "prediction_confidence": 0.8,
            "probabilities": {"Rock": 0.8},
            "analysis_status": "completed",
        },
        [],
    )

    reopened = LibraryStore(database_path, audio_directory)

    assert reopened.get_song("song-1")["original_filename"] == "persisted.wav"


def test_database_connections_close_cleanly(tmp_path):
    database_path = tmp_path / "library.db"
    store = LibraryStore(database_path, tmp_path / "audio")
    store.list_songs()

    shutil.rmtree(tmp_path)
    assert not database_path.exists()


def test_existing_phase_two_schema_migrates_without_data_loss(tmp_path):
    database_path = tmp_path / "library.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE songs (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            stored_audio_path TEXT NOT NULL UNIQUE,
            uploaded_at TEXT NOT NULL,
            predicted_genre TEXT,
            prediction_confidence REAL,
            probabilities_json TEXT NOT NULL,
            analysis_status TEXT NOT NULL
        );
        CREATE TABLE song_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id TEXT NOT NULL,
            window_index INTEGER NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            genre TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities_json TEXT NOT NULL
        );
        INSERT INTO songs VALUES ('old', 'old.wav', 'old.wav', 'now', 'Rock', 0.8, '{"Rock": 0.8}', 'completed');
        """
    )
    connection.commit()
    connection.close()

    store = LibraryStore(database_path, tmp_path / "audio")

    old_song = store.get_song("old")
    assert old_song["original_filename"] == "old.wav"
    assert old_song["feature_status"] == "missing"
    assert old_song["audio_features"] is None


def test_malformed_persisted_json_is_returned_as_missing_data(tmp_path):
    store = LibraryStore(tmp_path / "library.db", tmp_path / "audio")
    with store._connect() as connection:
        connection.execute(
            "UPDATE songs SET probabilities_json = ? WHERE id = ?",
            ("not-json", "missing"),
        )

    store.create_song(
        {
            "id": "missing",
            "original_filename": "missing.wav",
            "stored_audio_path": "missing.wav",
            "uploaded_at": store.timestamp(),
            "predicted_genre": "Rock",
            "prediction_confidence": 0.5,
            "probabilities": {"Rock": 0.5},
            "analysis_status": "completed",
        },
        [],
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE songs SET probabilities_json = ? WHERE id = ?",
            ("not-json", "missing"),
        )

    assert store.get_song("missing")["probabilities"] is None


def test_mixed_uploads_continue_and_save_only_successful_files(library_environment):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files=[
            ("files", ("valid.wav", make_wav(), "audio/wav")),
            ("files", ("invalid.txt", b"not audio", "text/plain")),
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 1
    assert data["failed"] == 1
    assert len(library_environment.list_songs()) == 1
    assert len(list(library_environment.audio_directory.glob("*"))) == 1


def test_failed_processing_cleans_file_and_database(library_environment, monkeypatch):
    def fail_analysis(model_name, audio):
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(main, "get_full_song_prediction", fail_analysis)
    client = TestClient(main.app)

    response = client.post(
        "/library/upload",
        files={"files": ("failed.wav", make_wav(), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["failed"] == 1
    assert library_environment.list_songs() == []
    assert list(library_environment.audio_directory.iterdir()) == []


def test_corrupt_audio_cleans_staged_file(library_environment):
    client = TestClient(main.app)

    response = client.post(
        "/library/upload",
        files={"files": ("corrupt.wav", b"not a wav file", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["failed"] == 1
    assert library_environment.list_songs() == []
    assert list(library_environment.audio_directory.iterdir()) == []


def test_long_upload_uses_full_song_inference_pipeline(library_environment, monkeypatch):
    monkeypatch.setattr(main, "get_full_song_prediction", REAL_FULL_SONG_PREDICTION)
    client = TestClient(main.app)

    response = client.post(
        "/library/upload",
        files={"files": ("long.wav", make_wav(31.0), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["successful"] == 1
    song_id = response.json()["results"][0]["song_id"]
    assert len(library_environment.get_song(song_id)["windows"]) == 2


def test_library_search_filter_detail_and_delete(library_environment):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files=[
            ("files", ("rock-song.wav", make_wav(), "audio/wav")),
            ("files", ("second-song.wav", make_wav(), "audio/wav")),
        ],
    )
    song_id = response.json()["results"][0]["song_id"]

    listed = client.get("/library", params={"search": "rock"})
    assert [song["original_filename"] for song in listed.json()["songs"]] == ["rock-song.wav"]
    filtered = client.get("/library", params={"genre": "Rock"})
    assert len(filtered.json()["songs"]) == 2

    detail = client.get(f"/library/{song_id}")
    assert detail.status_code == 200
    assert detail.json()["windows"][0]["genre"] == "Rock"
    stored_path = library_environment.audio_directory / detail.json()["stored_audio_path"]

    deleted = client.delete(f"/library/{song_id}")
    assert deleted.status_code == 200
    assert not stored_path.exists()
    assert client.get(f"/library/{song_id}").status_code == 404


def test_recommendations_and_feature_rebuild(library_environment):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files=[
            ("files", ("source.wav", make_wav(), "audio/wav")),
            ("files", ("candidate.wav", make_wav(), "audio/wav")),
        ],
    )
    source_id = response.json()["results"][0]["song_id"]
    candidate_id = response.json()["results"][1]["song_id"]

    source = library_environment.get_song(source_id)
    assert source["feature_status"] == "ready"
    assert source["audio_features"]

    recommendations = client.get(f"/recommendations/{source_id}")
    assert recommendations.status_code == 200
    payload = recommendations.json()
    assert payload["source"]["song_id"] == source_id
    assert payload["recommendations"][0]["song_id"] == candidate_id
    assert 0.0 <= payload["recommendations"][0]["score"] <= 1.0
    assert payload["formula"] == "0.7 * audio_similarity + 0.3 * genre_similarity"

    rebuild = client.post("/library/rebuild-features")
    assert rebuild.status_code == 200
    assert all(item["status"] == "already_ready" for item in rebuild.json()["results"])

    candidate_path = library_environment.audio_directory / library_environment.get_song(candidate_id)["stored_audio_path"]
    candidate_path.unlink()
    missing_file_response = client.get(f"/recommendations/{source_id}")
    assert missing_file_response.status_code == 200
    assert missing_file_response.json()["recommendations"] == []
    assert missing_file_response.json()["feature_errors"][0]["song_id"] == candidate_id


def test_missing_source_features_are_rebuilt_for_recommendations(library_environment):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files=[
            ("files", ("source.wav", make_wav(), "audio/wav")),
            ("files", ("candidate.wav", make_wav(), "audio/wav")),
        ],
    )
    source_id = response.json()["results"][0]["song_id"]
    library_environment.update_audio_features(source_id, None, "missing")

    result = client.get(f"/recommendations/{source_id}")

    assert result.status_code == 200
    assert result.json()["recommendations"]
    assert library_environment.get_song(source_id)["feature_status"] == "ready"


def test_force_rebuild_reprocesses_ready_features(library_environment, monkeypatch):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files={"files": ("ready.wav", make_wav(), "audio/wav")},
    )
    song_id = response.json()["results"][0]["song_id"]
    original_extract = main.extract_audio_features
    calls = []

    def counted_extract(audio):
        calls.append(song_id)
        return original_extract(audio)

    monkeypatch.setattr(main, "extract_audio_features", counted_extract)

    skipped = client.post("/library/rebuild-features")
    forced = client.post("/library/rebuild-features?force=true")

    assert skipped.json()["results"][0]["status"] == "already_ready"
    assert forced.json()["results"][0]["status"] == "ready"
    assert calls == [song_id]


def test_recommendations_with_one_song_are_empty(library_environment):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files={"files": ("only.wav", make_wav(), "audio/wav")},
    )
    song_id = response.json()["results"][0]["song_id"]

    result = client.get(f"/recommendations/{song_id}")

    assert result.status_code == 200
    assert result.json()["recommendations"] == []


def test_recommendation_limit_validation(library_environment):
    client = TestClient(main.app)
    response = client.post(
        "/library/upload",
        files={"files": ("only.wav", make_wav(), "audio/wav")},
    )
    song_id = response.json()["results"][0]["song_id"]

    assert client.get(f"/recommendations/{song_id}?limit=0").status_code == 422
    assert client.get(f"/recommendations/{song_id}?limit=-1").status_code == 422
    assert client.get(f"/recommendations/{song_id}?limit=not-a-number").status_code == 422
    assert client.get(f"/recommendations/{song_id}?limit=51").status_code == 422


def test_existing_predict_and_compare_routes_remain_compatible(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "load_audio", lambda path: np.zeros(22050))
    monkeypatch.setattr(main, "get_full_song_prediction", fake_analysis)

    prediction = client.post(
        "/predict",
        files={"audio": ("track.wav", make_wav(), "audio/wav")},
        data={"model": "resnet"},
    )
    assert prediction.status_code == 200
    assert prediction.json()["genre"] == "Rock"
    assert "probabilities" in prediction.json()

    comparison = client.post(
        "/compare",
        files={"audio": ("track.wav", make_wav(), "audio/wav")},
    )
    assert comparison.status_code == 200
    assert comparison.json()["consensus_genre"] == "Rock"
    assert len(comparison.json()["predictions"]) == 3
