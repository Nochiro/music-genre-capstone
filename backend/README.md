# Music Genre Classification API

FastAPI backend for real-time music genre classification using three deep learning models: CNN, LSTM, and ResNet18.

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Verify Models

Ensure the serving checkpoints exist at:
- `model_artifacts/cnn/best_cnn_model.pth`
- `model_artifacts/lstm/best_lstm_model.pth`
- `model_artifacts/resnet/resnet18_fp32.pth`

### 3. Start the API

```bash
# From the repository root
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will start on `http://localhost:8000`

### Persistent Music Library

The library uses SQLite and stores audio files separately from the database. Configure the
locations with:

```bash
MUSIQ_DATABASE_PATH=./data/music_library.db
MUSIQ_AUDIO_DIRECTORY=./data/audio
MUSIQ_MAX_UPLOAD_BYTES=104857600
```

The upload limit defaults to 100 MiB. In Docker, the image defaults to `/data/music_library.db`
and `/data/audio`; mount one persistent volume at `/data` so records and audio survive restarts:

```bash
docker run -p 8000:8000 -v musiciq-data:/data musiq-backend
```

## API Endpoints

### Health Check
```
GET /health
```
Returns API and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "cnn": true,
    "lstm": true,
    "resnet": true
  }
}
```

### Single Model Prediction
```
POST /predict
```
Classify audio using a specified model.

**Parameters:**
- `audio` (file): Audio file to classify (MP3, WAV, FLAC, etc.)
- `model` (string): Model to use (`cnn`, `lstm`, or `resnet`)

**Response:**
```json
{
  "model": "cnn",
  "genre": "Rock",
  "confidence": 0.892,
  "probabilities": {
    "Blues": 0.001,
    "Classical": 0.002,
    "Country": 0.003,
    "Electronic": 0.015,
    "Rock": 0.892,
    ...
  }
}
```

### Multi-Model Comparison
```
POST /compare
```
Classify audio using all three models and compare predictions.

**Parameters:**
- `audio` (file): Audio file to classify

**Response:**
```json
{
  "predictions": [
    {
      "model": "cnn",
      "genre": "Rock",
      "confidence": 0.892,
      "probabilities": { ... }
    },
    {
      "model": "lstm",
      "genre": "Rock",
      "confidence": 0.756,
      "probabilities": { ... }
    },
    {
      "model": "resnet",
      "genre": "Electronic",
      "confidence": 0.634,
      "probabilities": { ... }
    }
  ],
  "consensus_genre": "Rock"
}
```

### Persistent Library

```
POST /library/upload
GET /library?search=track&genre=Rock&limit=50&offset=0
GET /library/{song_id}
DELETE /library/{song_id}
```

`POST /library/upload` accepts repeated `files` fields, analyzes each file with the full-song
overlapping-window pipeline, and returns an individual success or failure result for every file.

Stored audio can be streamed without re-encoding:

```text
GET /library/{song_id}/audio
```

The endpoint supports browser HTTP byte ranges (`206 Partial Content` and `416 Range Not
Satisfiable`), seeking, `Accept-Ranges: bytes`, and the original file MIME type. Paths are
resolved inside `MUSIQ_AUDIO_DIRECTORY`; client requests never provide filesystem paths.

### Playlists

Playlists persist ordered references to library songs. Deleting a playlist never deletes songs;
deleting a library song removes its playlist membership through the foreign-key cascade.

```text
POST /playlists
GET /playlists
GET /playlists/{playlist_id}
PATCH /playlists/{playlist_id}
DELETE /playlists/{playlist_id}
POST /playlists/{playlist_id}/songs
DELETE /playlists/{playlist_id}/songs/{song_id}
PATCH /playlists/{playlist_id}/songs/reorder
POST /playlists/{playlist_id}/recommendations
```

Create a playlist:

```json
{
  "name": "Road songs",
  "description": "Tracks for the drive"
}
```

Add songs or selected recommendation results:

```json
{
  "song_ids": ["song-id-1", "song-id-2"]
}
```

Names are required and limited to 120 characters; descriptions are optional and limited to
1,000 characters. Song IDs must be known library songs. Duplicate additions are idempotent.
Reorder requests must contain every current playlist song exactly once. Playlist tables are
created with `CREATE TABLE IF NOT EXISTS`, and existing library rows are preserved during startup.

### Recommendations

Audio recommendations persist a full-song descriptor containing tempo, spectral centroid,
spectral bandwidth, spectral rolloff, zero-crossing rate, and 13 MFCC means and standard
deviations. Features are extracted over the existing overlapping windows, excluding zero
padding from a final partial window.

```text
GET /recommendations/{song_id}?limit=10
POST /library/rebuild-features
```

The ranking formula is:

```text
score = 0.7 * audio_similarity + 0.3 * genre_similarity
```

Audio similarity uses normalized Euclidean distance and genre similarity uses cosine similarity
over the complete 16-class probability vector. Scores are bounded to `[0, 1]`. These weights are
initial transparent defaults, not an evaluated claim of recommendation quality.

## Interactive Documentation

Once running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Model Details

### CNN
- Architecture: 3 convolutional blocks with BatchNorm and MaxPool
- Input: 1-channel mel spectrogram (128 × ~1292)
- Test Accuracy: ~58%

### LSTM
- Architecture: 2-layer LSTM + 2-layer classifier
- Input: Temporally downsampled mel spectrogram (322 × 128)
- Test Accuracy: ~56%

### ResNet18
- Architecture: ResNet18 with 1-channel conv1
- Input: 1-channel mel spectrogram (128 × ~1292)
- Test Accuracy: ~64% (best performer)

## Audio Preprocessing

All models use identical preprocessing:
1. Load audio at 22050 Hz (mono)
2. Enforce 30-second duration (pad/truncate as needed)
3. Extract mel spectrogram: 128 bins, FFT 2048, hop 512
4. Convert to dB scale: range [-80, 0] dB
5. Normalize: (mel + 80) / 80 → [0, 1]
6. Model-specific adjustments (channel dims, temporal downsampling for LSTM)

## Supported Genres (16 classes)

Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time / Historic, Pop, Rock, Soul-RnB, Spoken

## Performance

- **ResNet18** (best): 63.69% test accuracy, 3.66 ms/sample (GPU)
- **CNN**: ~58% test accuracy
- **LSTM**: 55.87% test accuracy

## Example Usage

### Using curl

```bash
# Single model prediction
curl -X POST "http://localhost:8000/predict" \
  -F "audio=@sample.mp3" \
  -F "model=resnet"

# Multi-model comparison
curl -X POST "http://localhost:8000/compare" \
  -F "audio=@sample.mp3"

# Health check
curl "http://localhost:8000/health"
```

### Using Python requests

```python
import requests

# Predict with ResNet
with open("sample.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"audio": f},
        data={"model": "resnet"}
    )
    print(response.json())

# Compare all models
with open("sample.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/compare",
        files={"audio": f}
    )
    print(response.json())
```

## Device Support

- **GPU (CUDA):** Automatically detected and used if available
- **CPU:** Falls back to CPU if CUDA not available
- Check device on startup in logs

## Troubleshooting

**Models not loading:**
- Verify checkpoint paths exist
- Check file permissions
- Ensure PyTorch and torchvision versions match requirements.txt

**Audio processing errors:**
- Ensure librosa can decode the audio format
- Try with WAV or MP3 first
- Check audio file is not corrupted

**Out of memory errors:**
- Reduce batch sizes (currently single samples)
- Use CPU instead of GPU
- Close other applications

## Architecture Notes

- Models are loaded once at startup and cached in memory
- All inference uses `torch.inference_mode()` for efficiency
- Audio preprocessing is deterministic and matches training exactly
- No data augmentation applied at inference time
