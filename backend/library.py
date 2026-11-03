"""SQLite persistence for the MusicIQ audio library."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = BASE_DIR / "data" / "music_library.db"
DEFAULT_AUDIO_DIRECTORY = BASE_DIR / "data" / "audio"


class PlaylistNotFoundError(LookupError):
    pass


class SongNotFoundError(LookupError):
    pass


class PlaylistValidationError(ValueError):
    pass


class LibraryStore:
    """Small SQLite repository with one connection per operation."""

    def __init__(self, database_path: str | Path, audio_directory: str | Path):
        self.database_path = Path(database_path)
        self.audio_directory = Path(audio_directory)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_directory.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @classmethod
    def from_environment(cls) -> "LibraryStore":
        database_path = os.getenv("MUSIQ_DATABASE_PATH", str(DEFAULT_DATABASE_PATH))
        audio_directory = os.getenv("MUSIQ_AUDIO_DIRECTORY", str(DEFAULT_AUDIO_DIRECTORY))
        return cls(database_path, audio_directory)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_audio_path TEXT NOT NULL UNIQUE,
                    uploaded_at TEXT NOT NULL,
                    predicted_genre TEXT,
                    prediction_confidence REAL,
                    probabilities_json TEXT NOT NULL,
                    analysis_status TEXT NOT NULL,
                    audio_features_json TEXT,
                    feature_status TEXT NOT NULL DEFAULT 'missing',
                    feature_error TEXT
                );

                CREATE TABLE IF NOT EXISTS song_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    window_index INTEGER NOT NULL,
                    start_seconds REAL NOT NULL,
                    end_seconds REAL NOT NULL,
                    genre TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    UNIQUE(song_id, window_index)
                );

                CREATE INDEX IF NOT EXISTS idx_songs_filename
                    ON songs(original_filename);
                CREATE INDEX IF NOT EXISTS idx_songs_genre
                    ON songs(predicted_genre);

                CREATE TABLE IF NOT EXISTS playlists (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS playlist_songs (
                    playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
                    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (playlist_id, song_id),
                    UNIQUE (playlist_id, position)
                );

                CREATE INDEX IF NOT EXISTS idx_playlist_songs_order
                    ON playlist_songs(playlist_id, position);
                """
            )
            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(songs)")
            }
            migrations = {
                "audio_features_json": "ALTER TABLE songs ADD COLUMN audio_features_json TEXT",
                "feature_status": "ALTER TABLE songs ADD COLUMN feature_status TEXT NOT NULL DEFAULT 'missing'",
                "feature_error": "ALTER TABLE songs ADD COLUMN feature_error TEXT",
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    connection.execute(statement)

    @staticmethod
    def _load_json(value: str | None) -> Any:
        if not value:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _song_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "original_filename": row["original_filename"],
            "stored_audio_path": row["stored_audio_path"],
            "uploaded_at": row["uploaded_at"],
            "predicted_genre": row["predicted_genre"],
            "prediction_confidence": row["prediction_confidence"],
            "probabilities": LibraryStore._load_json(row["probabilities_json"]),
            "analysis_status": row["analysis_status"],
            "audio_features": LibraryStore._load_json(row["audio_features_json"]),
            "feature_status": row["feature_status"],
            "feature_error": row["feature_error"],
        }

    def create_song(
        self,
        song: dict[str, Any],
        windows: list[dict[str, Any]],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO songs (
                    id, original_filename, stored_audio_path, uploaded_at,
                    predicted_genre, prediction_confidence,
                    probabilities_json, analysis_status,
                    audio_features_json, feature_status, feature_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    song["id"],
                    song["original_filename"],
                    song["stored_audio_path"],
                    song["uploaded_at"],
                    song["predicted_genre"],
                    song["prediction_confidence"],
                    json.dumps(song["probabilities"]),
                    song["analysis_status"],
                    json.dumps(song["audio_features"])
                    if song.get("audio_features") is not None else None,
                    song.get("feature_status", "missing"),
                    song.get("feature_error"),
                ),
            )
            connection.executemany(
                """
                INSERT INTO song_windows (
                    song_id, window_index, start_seconds, end_seconds,
                    genre, confidence, probabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        song["id"],
                        index,
                        window["start_seconds"],
                        window["end_seconds"],
                        window["genre"],
                        window["confidence"],
                        json.dumps(window["probabilities"]),
                    )
                    for index, window in enumerate(windows)
                ],
            )

    def list_songs(
        self,
        search: str | None = None,
        genre: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        if search:
            clauses.append("original_filename LIKE ?")
            parameters.append(f"%{search}%")
        if genre:
            clauses.append("predicted_genre = ?")
            parameters.append(genre)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM songs
                {where}
                ORDER BY uploaded_at DESC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        return [self._song_from_row(row) for row in rows]

    def all_songs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM songs ORDER BY uploaded_at DESC"
            ).fetchall()
        return [self._song_from_row(row) for row in rows]

    def update_audio_features(
        self,
        song_id: str,
        features: dict[str, float] | None,
        status: str,
        error: str | None = None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE songs
                SET audio_features_json = ?, feature_status = ?, feature_error = ?
                WHERE id = ?
                """,
                (
                    json.dumps(features) if features is not None else None,
                    status,
                    error,
                    song_id,
                ),
            )
        return cursor.rowcount > 0

    def get_song(self, song_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM songs WHERE id = ?",
                (song_id,),
            ).fetchone()
            if row is None:
                return None
            song = self._song_from_row(row)
            windows = connection.execute(
                """
                SELECT window_index, start_seconds, end_seconds,
                       genre, confidence, probabilities_json
                FROM song_windows
                WHERE song_id = ?
                ORDER BY window_index
                """,
                (song_id,),
            ).fetchall()
        song["windows"] = [
            {
                "window_index": window["window_index"],
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "genre": window["genre"],
                "confidence": window["confidence"],
                "probabilities": self._load_json(window["probabilities_json"]),
            }
            for window in windows
        ]
        return song

    def delete_song(self, song_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM songs WHERE id = ?",
                (song_id,),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _playlist_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "song_count": row["song_count"] if "song_count" in row.keys() else 0,
        }

    def create_playlist(
        self,
        playlist_id: str,
        name: str,
        description: str | None,
        timestamp: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO playlists (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (playlist_id, name, description, timestamp, timestamp),
            )
        return self.get_playlist(playlist_id)

    def list_playlists(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*, COUNT(ps.song_id) AS song_count
                FROM playlists AS p
                LEFT JOIN playlist_songs AS ps ON ps.playlist_id = p.id
                GROUP BY p.id
                ORDER BY p.updated_at DESC, p.created_at DESC
                """
            ).fetchall()
        return [self._playlist_from_row(row) for row in rows]

    def get_playlist(self, playlist_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.*, COUNT(ps.song_id) AS song_count
                FROM playlists AS p
                LEFT JOIN playlist_songs AS ps ON ps.playlist_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (playlist_id,),
            ).fetchone()
            if row is None:
                return None
            playlist = self._playlist_from_row(row)
            songs = connection.execute(
                """
                SELECT s.*, ps.position
                FROM playlist_songs AS ps
                JOIN songs AS s ON s.id = ps.song_id
                WHERE ps.playlist_id = ?
                ORDER BY ps.position
                """,
                (playlist_id,),
            ).fetchall()
        playlist["songs"] = [
            {**self._song_from_row(song), "position": song["position"]}
            for song in songs
        ]
        return playlist

    def update_playlist(
        self,
        playlist_id: str,
        name: str | None,
        description: str | None,
        update_name: bool,
        update_description: bool,
        timestamp: str,
    ) -> dict[str, Any] | None:
        updates = []
        parameters: list[Any] = []
        if update_name:
            updates.append("name = ?")
            parameters.append(name)
        if update_description:
            updates.append("description = ?")
            parameters.append(description)
        if updates:
            updates.append("updated_at = ?")
            parameters.append(timestamp)
            parameters.append(playlist_id)
            with self._connect() as connection:
                cursor = connection.execute(
                    f"UPDATE playlists SET {', '.join(updates)} WHERE id = ?",
                    parameters,
                )
                if cursor.rowcount == 0:
                    return None
        return self.get_playlist(playlist_id)

    def delete_playlist(self, playlist_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM playlists WHERE id = ?",
                (playlist_id,),
            )
        return cursor.rowcount > 0

    def add_songs_to_playlist(
        self,
        playlist_id: str,
        song_ids: list[str],
        timestamp: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            playlist = connection.execute(
                "SELECT id FROM playlists WHERE id = ?",
                (playlist_id,),
            ).fetchone()
            if playlist is None:
                raise PlaylistNotFoundError(playlist_id)
            if len(song_ids) != len(set(song_ids)):
                raise PlaylistValidationError("Duplicate song IDs are not allowed.")
            for song_id in song_ids:
                if connection.execute(
                    "SELECT 1 FROM songs WHERE id = ?", (song_id,)
                ).fetchone() is None:
                    raise SongNotFoundError(song_id)
            existing = {
                row["song_id"]
                for row in connection.execute(
                    "SELECT song_id FROM playlist_songs WHERE playlist_id = ?",
                    (playlist_id,),
                )
            }
            position_row = connection.execute(
                "SELECT COALESCE(MAX(position), -1) AS position FROM playlist_songs WHERE playlist_id = ?",
                (playlist_id,),
            ).fetchone()
            next_position = position_row["position"] + 1
            added = 0
            for song_id in song_ids:
                if song_id in existing:
                    continue
                connection.execute(
                    "INSERT INTO playlist_songs (playlist_id, song_id, position) VALUES (?, ?, ?)",
                    (playlist_id, song_id, next_position),
                )
                next_position += 1
                added += 1
            if song_ids:
                connection.execute(
                    "UPDATE playlists SET updated_at = ? WHERE id = ?",
                    (timestamp, playlist_id),
                )
        return {"added": added, "duplicate": added != len(song_ids), "playlist": self.get_playlist(playlist_id)}

    def remove_song_from_playlist(self, playlist_id: str, song_id: str, timestamp: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM playlist_songs WHERE playlist_id = ? AND song_id = ?",
                (playlist_id, song_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE playlists SET updated_at = ? WHERE id = ?",
                    (timestamp, playlist_id),
                )
                rows = connection.execute(
                    "SELECT song_id FROM playlist_songs WHERE playlist_id = ? ORDER BY position",
                    (playlist_id,),
                ).fetchall()
                for position, row in enumerate(rows):
                    connection.execute(
                        "UPDATE playlist_songs SET position = ? WHERE playlist_id = ? AND song_id = ?",
                        (position, playlist_id, row["song_id"]),
                    )
        return cursor.rowcount > 0

    def reorder_playlist(self, playlist_id: str, song_ids: list[str], timestamp: str) -> dict[str, Any]:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM playlists WHERE id = ?", (playlist_id,)).fetchone() is None:
                raise PlaylistNotFoundError(playlist_id)
            current = [
                row["song_id"]
                for row in connection.execute(
                    "SELECT song_id FROM playlist_songs WHERE playlist_id = ? ORDER BY position",
                    (playlist_id,),
                )
            ]
            if len(song_ids) != len(set(song_ids)) or set(song_ids) != set(current):
                raise PlaylistValidationError("Reorder must contain every playlist song exactly once.")
            connection.execute(
                "UPDATE playlists SET updated_at = ? WHERE id = ?",
                (timestamp, playlist_id),
            )
            for position, song_id in enumerate(song_ids):
                connection.execute(
                    "UPDATE playlist_songs SET position = ? WHERE playlist_id = ? AND song_id = ?",
                    (position + len(song_ids), playlist_id, song_id),
                )
            for position, song_id in enumerate(song_ids):
                connection.execute(
                    "UPDATE playlist_songs SET position = ? WHERE playlist_id = ? AND song_id = ?",
                    (position, playlist_id, song_id),
                )
        return self.get_playlist(playlist_id)

    @staticmethod
    def timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
