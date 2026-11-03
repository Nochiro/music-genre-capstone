import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.library import LibraryStore


def add_song(store, song_id: str) -> None:
    store.create_song(
        {
            "id": song_id,
            "original_filename": f"{song_id}.wav",
            "stored_audio_path": f"{song_id}.wav",
            "uploaded_at": store.timestamp(),
            "predicted_genre": "Rock",
            "prediction_confidence": 0.8,
            "probabilities": {"Rock": 1.0},
            "analysis_status": "completed",
        },
        [],
    )


@pytest.fixture
def playlist_client(tmp_path, monkeypatch):
    store = LibraryStore(tmp_path / "library.db", tmp_path / "audio")
    add_song(store, "song-a")
    add_song(store, "song-b")
    add_song(store, "song-c")
    monkeypatch.setattr(main, "library_store", store)
    return TestClient(main.app), store


def test_playlist_crud_ordering_and_duplicates(playlist_client):
    client, store = playlist_client
    created = client.post("/playlists", json={"name": "Road songs", "description": "Drive"})
    assert created.status_code == 201
    playlist = created.json()
    playlist_id = playlist["id"]

    listed = client.get("/playlists")
    assert listed.status_code == 200
    assert listed.json()["playlists"][0]["song_count"] == 0

    added = client.post(f"/playlists/{playlist_id}/songs", json={"song_ids": ["song-a", "song-b"]})
    assert [song["id"] for song in added.json()["playlist"]["songs"]] == ["song-a", "song-b"]
    duplicate = client.post(f"/playlists/{playlist_id}/songs", json={"song_ids": ["song-a"]})
    assert duplicate.status_code == 200
    assert duplicate.json()["added"] == 0

    reordered = client.patch(
        f"/playlists/{playlist_id}/songs/reorder",
        json={"song_ids": ["song-b", "song-a"]},
    )
    assert [song["id"] for song in reordered.json()["songs"]] == ["song-b", "song-a"]

    updated = client.patch(f"/playlists/{playlist_id}", json={"name": "Road songs 2", "description": None})
    assert updated.status_code == 200
    assert updated.json()["name"] == "Road songs 2"
    assert updated.json()["description"] is None

    removed = client.delete(f"/playlists/{playlist_id}/songs/song-a")
    assert removed.status_code == 200
    assert [song["id"] for song in removed.json()["songs"]] == ["song-b"]
    assert store.get_song("song-a") is not None


def test_playlist_validation_unknown_ids_and_reorder_rollback(playlist_client):
    client, store = playlist_client
    playlist_id = client.post("/playlists", json={"name": "Valid"}).json()["id"]

    assert client.post(f"/playlists/{playlist_id}/songs", json={"song_ids": ["song-a", "missing"]}).status_code == 404
    assert client.get(f"/playlists/{playlist_id}").json()["songs"] == []
    assert client.patch(f"/playlists/{playlist_id}/songs/reorder", json={"song_ids": ["missing"]}).status_code == 422
    assert client.patch(f"/playlists/{playlist_id}/songs/reorder", json={"song_ids": ["song-a", "song-a"]}).status_code == 422
    assert client.post(f"/playlists/{playlist_id}/songs", json={"song_ids": [" "]}).status_code == 422
    assert client.post("/playlists", json={"name": "   "}).status_code == 422
    assert client.patch(f"/playlists/{playlist_id}", json={}).status_code == 422
    assert client.patch(f"/playlists/{playlist_id}", json={"name": None}).status_code == 422
    assert client.get("/playlists/unknown").status_code == 404
    assert client.delete("/playlists/unknown").status_code == 404

    # Direct store validation is transactional even when called outside HTTP.
    with pytest.raises(Exception):
        store.add_songs_to_playlist(playlist_id, ["song-a", "missing"], store.timestamp())
    assert store.get_playlist(playlist_id)["songs"] == []


def test_shared_songs_library_deletion_and_playlist_deletion(playlist_client):
    client, store = playlist_client
    first = client.post("/playlists", json={"name": "First"}).json()["id"]
    second = client.post("/playlists", json={"name": "Second"}).json()["id"]
    client.post(f"/playlists/{first}/songs", json={"song_ids": ["song-a", "song-b"]})
    client.post(f"/playlists/{second}/songs", json={"song_ids": ["song-a"]})

    assert client.delete(f"/playlists/{first}").status_code == 200
    assert store.get_song("song-a") is not None
    assert client.get(f"/playlists/{second}").json()["songs"][0]["id"] == "song-a"

    assert store.delete_song("song-a") is True
    assert client.get(f"/playlists/{second}").json()["songs"] == []
    assert store.get_playlist(second)["song_count"] == 0


def test_playlist_persistence_and_migration(tmp_path):
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
        INSERT INTO songs VALUES ('legacy', 'legacy.wav', 'legacy.wav', 'now', 'Rock', 0.5, '{"Rock": 1}', 'completed');
        """
    )
    connection.commit()
    connection.close()

    store = LibraryStore(database_path, tmp_path / "audio")
    playlist = store.create_playlist("playlist-1", "Legacy", None, store.timestamp())
    assert playlist["song_count"] == 0
    store.add_songs_to_playlist("playlist-1", ["legacy"], store.timestamp())

    reopened = LibraryStore(database_path, tmp_path / "audio")
    persisted = reopened.get_playlist("playlist-1")
    assert [song["id"] for song in persisted["songs"]] == ["legacy"]
    assert reopened.get_song("legacy") is not None


def test_recommendation_import_endpoint(playlist_client, monkeypatch):
    client, _ = playlist_client
    playlist_id = client.post("/playlists", json={"name": "Suggested"}).json()["id"]
    response = client.post(
        f"/playlists/{playlist_id}/recommendations",
        json={"song_ids": ["song-a", "song-c"]},
    )
    assert response.status_code == 200
    assert [song["id"] for song in response.json()["playlist"]["songs"]] == ["song-a", "song-c"]
    assert client.post(
        f"/playlists/{playlist_id}/recommendations",
        json={"song_ids": ["song-a"]},
    ).json()["added"] == 0
