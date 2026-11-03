from fastapi.testclient import TestClient

from backend import main
from backend.library import LibraryStore


def add_song(store, song_id: str, stored_path: str = "song.mp3") -> None:
    store.create_song(
        {
            "id": song_id,
            "original_filename": stored_path,
            "stored_audio_path": stored_path,
            "uploaded_at": store.timestamp(),
            "predicted_genre": "Rock",
            "prediction_confidence": 0.8,
            "probabilities": {"Rock": 1.0},
            "analysis_status": "completed",
        },
        [],
    )


def test_audio_stream_full_and_range_responses(tmp_path, monkeypatch):
    store = LibraryStore(tmp_path / "library.db", tmp_path / "audio")
    add_song(store, "song-1")
    payload = bytes(range(32))
    (store.audio_directory / "song.mp3").write_bytes(payload)
    monkeypatch.setattr(main, "library_store", store)
    client = TestClient(main.app)

    full = client.get("/library/song-1/audio")
    assert full.status_code == 200
    assert full.content == payload
    assert full.headers["content-type"] == "audio/mpeg"
    assert full.headers["content-length"] == "32"
    assert full.headers["accept-ranges"] == "bytes"

    explicit = client.get("/library/song-1/audio", headers={"Range": "bytes=2-5"})
    assert explicit.status_code == 206
    assert explicit.content == payload[2:6]
    assert explicit.headers["content-range"] == "bytes 2-5/32"
    assert explicit.headers["content-length"] == "4"

    open_ended = client.get("/library/song-1/audio", headers={"Range": "bytes=8-"})
    assert open_ended.status_code == 206
    assert open_ended.content == payload[8:]

    suffix = client.get("/library/song-1/audio", headers={"Range": "bytes=-4"})
    assert suffix.status_code == 206
    assert suffix.content == payload[-4:]


def test_audio_stream_rejects_invalid_ranges_and_missing_files(tmp_path, monkeypatch):
    store = LibraryStore(tmp_path / "library.db", tmp_path / "audio")
    add_song(store, "song-1")
    (store.audio_directory / "song.mp3").write_bytes(b"abc")
    add_song(store, "missing", "missing.mp3")
    add_song(store, "traversal", "../outside.mp3")
    monkeypatch.setattr(main, "library_store", store)
    client = TestClient(main.app)

    for value in ("bytes=99-100", "bytes=4-2", "bytes=0-1,2-3", "items=0-1", "bytes=-0"):
        response = client.get("/library/song-1/audio", headers={"Range": value})
        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */3"

    assert client.get("/library/unknown/audio").status_code == 404
    assert client.get("/library/missing/audio").status_code == 404
    assert client.get("/library/traversal/audio").status_code == 404
