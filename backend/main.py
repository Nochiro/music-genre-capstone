from pathlib import Path
from collections import Counter
import gc
import mimetypes
import os
import tempfile
from uuid import uuid4

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.models import GenreCNN, GenreLSTM, GenreResNet, GENRE_NAMES
from backend.library import LibraryStore
from backend.library import (
    PlaylistNotFoundError,
    PlaylistValidationError,
    SongNotFoundError,
)
from backend.schemas import (
    PlaylistCreateRequest,
    PlaylistReorderRequest,
    PlaylistSongsRequest,
    PlaylistUpdateRequest,
)
from backend.preprocessing import (
    create_audio_windows,
    extract_audio_features,
    load_audio,
    extract_mel_spectrogram,
    preprocess_for_cnn,
    preprocess_for_lstm,
    preprocess_for_resnet,
)
from backend.recommendations import rank_recommendations


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_artifacts"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 16
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
MAX_LIBRARY_UPLOAD_BYTES = int(
    os.getenv("MUSIQ_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))
)


def remove_staged_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        # Some Windows audio decoders release file handles during collection.
        gc.collect()
        path.unlink(missing_ok=True)


MODEL_PATHS = {
    "cnn": MODEL_DIR / "cnn" / "best_cnn_model.pth",
    "lstm": MODEL_DIR / "lstm" / "best_lstm_model.pth",
    "resnet": MODEL_DIR / "resnet" / "resnet18_fp32.pth",
}


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="GenreLab API",
    description="Music Genre Classification using CNN, LSTM and ResNet18",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Model loading
# ============================================================

models = {}
library_store = LibraryStore.from_environment()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def load_models():
    print(f"Using device: {DEVICE}")

    # CNN
    cnn = GenreCNN(num_classes=NUM_CLASSES)
    cnn = load_checkpoint(cnn, MODEL_PATHS["cnn"])
    models["cnn"] = cnn

    print(
        f"cnn loaded: "
        f"{sum(p.numel() for p in cnn.parameters()):,} parameters"
    )

    # LSTM
    lstm = GenreLSTM(num_classes=NUM_CLASSES)
    lstm = load_checkpoint(lstm, MODEL_PATHS["lstm"])
    models["lstm"] = lstm

    print(
        f"lstm loaded: "
        f"{sum(p.numel() for p in lstm.parameters()):,} parameters"
    )

    # ResNet18
    resnet = GenreResNet(num_classes=NUM_CLASSES)
    resnet = load_checkpoint(resnet, MODEL_PATHS["resnet"])
    models["resnet"] = resnet

    print(
        f"resnet loaded: "
        f"{sum(p.numel() for p in resnet.parameters()):,} parameters"
    )

    print("Startup complete")


# ============================================================
# Prediction helpers
# ============================================================

def _probability_dict(probabilities):
    return {
        GENRE_NAMES[i]: probabilities[i]
        for i in range(len(GENRE_NAMES))
    }


def _stored_audio_path(song: dict) -> Path:
    audio_root = library_store.audio_directory.resolve()
    stored_path = (audio_root / song["stored_audio_path"]).resolve()
    if audio_root not in stored_path.parents:
        raise ValueError("Stored audio path is outside the configured audio directory.")
    return stored_path


def _parse_audio_range(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    if not range_header:
        return None
    if not range_header.startswith("bytes=") or "," in range_header:
        raise ValueError("Invalid range")
    value = range_header[6:].strip()
    if "-" not in value:
        raise ValueError("Invalid range")
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError("Invalid range")
            start = max(file_size - suffix_length, 0)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
            if start < 0 or end < start:
                raise ValueError("Invalid range")
            end = min(end, file_size - 1)
        if start >= file_size or end < 0:
            raise ValueError("Invalid range")
    except (TypeError, ValueError):
        raise ValueError("Invalid range") from None
    return start, end


def _stream_file(path: Path, start: int, end: int):
    remaining = end - start + 1
    with path.open("rb") as audio_file:
        audio_file.seek(start)
        while remaining > 0:
            chunk = audio_file.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def ensure_song_features(song: dict, force: bool = False) -> bool:
    try:
        stored_path = _stored_audio_path(song)
        if not stored_path.is_file():
            raise FileNotFoundError("Stored audio file is missing.")
        if song.get("feature_status") == "ready" and not force:
            return True
        audio = load_audio(str(stored_path))
        features = extract_audio_features(audio)
        library_store.update_audio_features(song["id"], features, "ready")
        song["audio_features"] = features
        song["feature_status"] = "ready"
        song["feature_error"] = None
        return True
    except Exception as error:
        message = str(error)
        library_store.update_audio_features(song["id"], None, "failed", message)
        song["audio_features"] = None
        song["feature_status"] = "failed"
        song["feature_error"] = message
        return False


def get_prediction(model_name: str, audio):
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    mel_db = extract_mel_spectrogram(audio)

    if model_name == "cnn":
        x = preprocess_for_cnn(mel_db)
        x = x.unsqueeze(0)

    elif model_name == "lstm":
        x = preprocess_for_lstm(mel_db)
        x = x.unsqueeze(0)

    elif model_name == "resnet":
        x = preprocess_for_resnet(mel_db)
        x = x.unsqueeze(0)

    x = x.to(DEVICE)

    model = models[model_name]

    with torch.no_grad():
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    probability_values = probabilities[0].cpu().tolist()

    return {
        "model": model_name,
        "genre": GENRE_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": _probability_dict(probability_values),
    }


def get_full_song_prediction(model_name: str, audio, sr: int = 22050):
    """Predict a song by averaging per-window probabilities.

    Equal-weight arithmetic averaging prevents songs with more windows from
    implicitly changing the class scale. The argmax of the averaged
    probabilities is the overall prediction and confidence.
    """
    windows = create_audio_windows(audio, sr=sr)
    window_results = []
    probability_vectors = []

    for start_seconds, end_seconds, audio_window in windows:
        result = get_prediction(model_name, audio_window)
        probability_vectors.append(
            [result["probabilities"][genre] for genre in GENRE_NAMES]
        )
        window_results.append({
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "genre": result["genre"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        })

    overall_probabilities = torch.tensor(probability_vectors).mean(dim=0).tolist()
    predicted_index = int(torch.tensor(overall_probabilities).argmax().item())

    return {
        "model": model_name,
        "genre": GENRE_NAMES[predicted_index],
        "confidence": overall_probabilities[predicted_index],
        "probabilities": _probability_dict(overall_probabilities),
        "windows": window_results,
    }


async def save_upload_to_temp(audio: UploadFile):
    suffix = Path(audio.filename or "").suffix

    if not suffix:
        suffix = ".audio"

    contents = await audio.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty.",
        )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_path = Path(temp_file.name)

    try:
        temp_file.write(contents)
        temp_file.close()
        return temp_path

    except Exception:
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise


async def stage_library_upload(audio: UploadFile) -> tuple[Path, str, str]:
    """Stream one library upload to a temporary file under audio storage."""
    original_filename = Path(audio.filename or "").name
    extension = Path(original_filename).suffix.lower()
    if not original_filename or extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("Unsupported audio format.")

    temporary_path = library_store.audio_directory / f".upload-{uuid4().hex}.part"
    total_bytes = 0
    try:
        with temporary_path.open("wb") as output:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_LIBRARY_UPLOAD_BYTES:
                    raise ValueError(
                        f"File exceeds the {MAX_LIBRARY_UPLOAD_BYTES} byte limit."
                    )
                output.write(chunk)
        if total_bytes == 0:
            raise ValueError("Uploaded audio file is empty.")
        return temporary_path, original_filename, extension
    except Exception:
        remove_staged_file(temporary_path)
        raise


async def process_library_upload(audio: UploadFile) -> dict:
    temporary_path = None
    stored_path = None
    song_id = str(uuid4())
    try:
        temporary_path, original_filename, extension = await stage_library_upload(audio)
        audio_data = load_audio(str(temporary_path))
        analysis = get_full_song_prediction("resnet", audio_data)
        feature_status = "ready"
        feature_error = None
        try:
            audio_features = extract_audio_features(audio_data)
        except Exception as error:
            audio_features = None
            feature_status = "failed"
            feature_error = str(error)

        stored_filename = f"{song_id}{extension}"
        stored_path = library_store.audio_directory / stored_filename
        temporary_path.replace(stored_path)
        library_store.create_song(
            {
                "id": song_id,
                "original_filename": original_filename,
                "stored_audio_path": stored_filename,
                "uploaded_at": library_store.timestamp(),
                "predicted_genre": analysis["genre"],
                "prediction_confidence": analysis["confidence"],
                "probabilities": analysis["probabilities"],
                "analysis_status": "completed",
                "audio_features": audio_features,
                "feature_status": feature_status,
                "feature_error": feature_error,
            },
            analysis["windows"],
        )
        return {
            "success": True,
            "song_id": song_id,
            "filename": original_filename,
            "genre": analysis["genre"],
            "confidence": analysis["confidence"],
            "feature_status": feature_status,
            "feature_error": feature_error,
        }
    except Exception:
        remove_staged_file(temporary_path)
        remove_staged_file(stored_path)
        raise


# ============================================================
# Startup
# ============================================================

load_models()


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "name": "GenreLab API",
        "version": "1.0.0",
        "status": "running",
        "models": ["cnn", "lstm", "resnet"],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "compare": "/compare",
            "library": "/library",
            "recommendations": "/recommendations/{song_id}",
            "playlists": "/playlists",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "cnn": "cnn" in models,
            "lstm": "lstm" in models,
            "resnet": "resnet" in models,
        },
    }


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    model: str = Form("resnet"),
):
    if model not in ["cnn", "lstm", "resnet"]:
        raise HTTPException(
            status_code=400,
            detail="Model must be cnn, lstm, or resnet.",
        )

    temp_path = await save_upload_to_temp(audio)

    try:
        audio_data = load_audio(str(temp_path))

        result = get_full_song_prediction(
            model,
            audio_data,
        )

        return {
            "filename": audio.filename,
            **result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/compare")
async def compare(
    audio: UploadFile = File(...),
):
    temp_path = await save_upload_to_temp(audio)

    try:
        audio_data = load_audio(str(temp_path))

        cnn_result = get_full_song_prediction(
            "cnn",
            audio_data,
        )

        lstm_result = get_full_song_prediction(
            "lstm",
            audio_data,
        )

        resnet_result = get_full_song_prediction(
            "resnet",
            audio_data,
        )

        predictions = [
            cnn_result,
            lstm_result,
            resnet_result,
        ]

        genres = [
            prediction["genre"]
            for prediction in predictions
        ]

        vote_counts = Counter(genres)

        consensus_genre, votes = vote_counts.most_common(1)[0]

        return {
            "filename": audio.filename,
            "predictions": predictions,
            "consensus_genre": consensus_genre,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}",
        )

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/library/upload")
async def upload_to_library(files: list[UploadFile] = File(...)):
    results = []
    for audio in files:
        filename = audio.filename or ""
        try:
            results.append(await process_library_upload(audio))
        except Exception as error:
            results.append({
                "success": False,
                "filename": filename,
                "error": str(error),
            })
    return {
        "results": results,
        "successful": sum(result["success"] for result in results),
        "failed": sum(not result["success"] for result in results),
    }


@app.get("/library")
def list_library(
    search: str | None = None,
    genre: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return {
        "songs": library_store.list_songs(search, genre, limit, offset),
        "limit": limit,
        "offset": offset,
    }


@app.post("/library/rebuild-features")
def rebuild_library_features(force: bool = Query(False)):
    results = []
    for song in library_store.all_songs():
        if song["analysis_status"] != "completed":
            results.append({
                "song_id": song["id"],
                "filename": song["original_filename"],
                "success": False,
                "status": "skipped",
                "error": "Song analysis did not complete.",
            })
            continue
        if song.get("feature_status") == "ready" and not force:
            results.append({
                "song_id": song["id"],
                "filename": song["original_filename"],
                "success": True,
                "status": "already_ready",
            })
            continue
        success = ensure_song_features(song, force=force)
        results.append({
            "song_id": song["id"],
            "filename": song["original_filename"],
            "success": success,
            "status": song["feature_status"],
            "error": song.get("feature_error"),
        })
    return {"results": results}


@app.get("/library/{song_id}")
def get_library_song(song_id: str):
    song = library_store.get_song(song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found.")
    return song


@app.get("/library/{song_id}/audio")
def stream_library_audio(song_id: str, request: Request):
    song = library_store.get_song(song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found.")
    try:
        audio_path = _stored_audio_path(song)
    except ValueError:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    file_size = audio_path.stat().st_size
    media_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": media_type,
    }
    if file_size == 0:
        headers["Content-Length"] = "0"
        return StreamingResponse(iter(()), status_code=200, headers=headers, media_type=media_type)

    try:
        byte_range = _parse_audio_range(request.headers.get("range"), file_size)
    except ValueError:
        return StreamingResponse(
            iter(()),
            status_code=416,
            headers={**headers, "Content-Range": f"bytes */{file_size}", "Content-Length": "0"},
            media_type=media_type,
        )

    if byte_range is None:
        start, end = 0, file_size - 1
        status_code = 200
    else:
        start, end = byte_range
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _stream_file(audio_path, start, end),
        status_code=status_code,
        headers=headers,
        media_type=media_type,
    )


@app.get("/recommendations/{song_id}")
def recommendations(
    song_id: str,
    limit: int = Query(10, ge=1, le=50),
):
    source = library_store.get_song(song_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Song not found.")
    if source["analysis_status"] != "completed":
        raise HTTPException(status_code=409, detail="Song analysis is not complete.")

    source_ready = ensure_song_features(source)
    candidates = []
    feature_errors = []
    for song in library_store.all_songs():
        if song["id"] == song_id or song["analysis_status"] != "completed":
            continue
        if not ensure_song_features(song):
            feature_errors.append({
                "song_id": song["id"],
                "filename": song["original_filename"],
                "error": song.get("feature_error"),
            })
            continue
        candidates.append(song)

    ranked = rank_recommendations(source, candidates)[:limit] if source_ready else []
    return {
        "source": {
            "song_id": source["id"],
            "filename": source["original_filename"],
            "predicted_genre": source["predicted_genre"],
            "confidence": source["prediction_confidence"],
        },
        "recommendations": ranked,
        "formula": "0.7 * audio_similarity + 0.3 * genre_similarity",
        "message": (
            "Ranked by normalized audio features and the complete genre probability vector."
            if ranked else "No comparable analyzed songs are currently available."
        ),
        "feature_errors": feature_errors,
    }


def _playlist_or_404(playlist_id: str):
    playlist = library_store.get_playlist(playlist_id)
    if playlist is None:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    return playlist


def _clean_playlist_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Playlist name cannot be blank.")
    return cleaned


def _validate_playlist_song_ids(song_ids: list[str]) -> None:
    if any(not song_id.strip() for song_id in song_ids):
        raise HTTPException(status_code=422, detail="Song IDs cannot be blank.")


@app.post("/playlists", status_code=201)
def create_playlist(request: PlaylistCreateRequest):
    name = _clean_playlist_name(request.name)
    description = request.description.strip() if request.description else None
    return library_store.create_playlist(
        str(uuid4()),
        name,
        description or None,
        library_store.timestamp(),
    )


@app.get("/playlists")
def list_playlists():
    return {"playlists": library_store.list_playlists()}


@app.get("/playlists/{playlist_id}")
def get_playlist(playlist_id: str):
    return _playlist_or_404(playlist_id)


@app.patch("/playlists/{playlist_id}")
def update_playlist(playlist_id: str, request: PlaylistUpdateRequest):
    _playlist_or_404(playlist_id)
    fields = request.model_fields_set
    if not fields:
        raise HTTPException(status_code=422, detail="No playlist fields were supplied.")
    if "name" in fields and request.name is None:
        raise HTTPException(status_code=422, detail="Playlist name cannot be null.")
    name = _clean_playlist_name(request.name) if "name" in fields and request.name is not None else request.name
    description = request.description.strip() if "description" in fields and request.description else request.description
    return library_store.update_playlist(
        playlist_id,
        name,
        description or None,
        "name" in fields,
        "description" in fields,
        library_store.timestamp(),
    )


@app.delete("/playlists/{playlist_id}")
def delete_playlist(playlist_id: str):
    _playlist_or_404(playlist_id)
    library_store.delete_playlist(playlist_id)
    return {"deleted": True, "playlist_id": playlist_id}


@app.post("/playlists/{playlist_id}/songs")
def add_playlist_songs(playlist_id: str, request: PlaylistSongsRequest):
    _validate_playlist_song_ids(request.song_ids)
    try:
        return library_store.add_songs_to_playlist(
            playlist_id,
            request.song_ids,
            library_store.timestamp(),
        )
    except PlaylistNotFoundError:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    except SongNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Song not found: {error}")
    except PlaylistValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.delete("/playlists/{playlist_id}/songs/{song_id}")
def remove_playlist_song(playlist_id: str, song_id: str):
    _playlist_or_404(playlist_id)
    if library_store.get_song(song_id) is None:
        raise HTTPException(status_code=404, detail="Song not found.")
    if not library_store.remove_song_from_playlist(playlist_id, song_id, library_store.timestamp()):
        raise HTTPException(status_code=404, detail="Song is not in this playlist.")
    return library_store.get_playlist(playlist_id)


@app.patch("/playlists/{playlist_id}/songs/reorder")
def reorder_playlist(playlist_id: str, request: PlaylistReorderRequest):
    _validate_playlist_song_ids(request.song_ids)
    try:
        return library_store.reorder_playlist(
            playlist_id,
            request.song_ids,
            library_store.timestamp(),
        )
    except PlaylistNotFoundError:
        raise HTTPException(status_code=404, detail="Playlist not found.")
    except PlaylistValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/playlists/{playlist_id}/recommendations")
def add_playlist_recommendations(playlist_id: str, request: PlaylistSongsRequest):
    return add_playlist_songs(playlist_id, request)


@app.delete("/library/{song_id}")
def delete_library_song(song_id: str):
    song = library_store.get_song(song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found.")

    audio_root = library_store.audio_directory.resolve()
    stored_path = (audio_root / song["stored_audio_path"]).resolve()
    if audio_root not in stored_path.parents:
        raise HTTPException(status_code=500, detail="Stored audio path is invalid.")

    tombstone_path = audio_root / f".delete-{uuid4().hex}.part"
    try:
        if stored_path.exists():
            stored_path.replace(tombstone_path)
        if not library_store.delete_song(song_id):
            if tombstone_path.exists():
                tombstone_path.replace(stored_path)
            raise HTTPException(status_code=404, detail="Song not found.")
        tombstone_path.unlink(missing_ok=True)
    except HTTPException:
        if tombstone_path.exists() and not stored_path.exists():
            tombstone_path.replace(stored_path)
        raise
    except Exception as error:
        if tombstone_path.exists() and not stored_path.exists():
            tombstone_path.replace(stored_path)
        raise HTTPException(status_code=500, detail=f"Could not delete song: {error}")

    return {"deleted": True, "song_id": song_id}