# Music Genre Classification API

FastAPI backend for real-time music genre classification using three deep learning models: CNN, LSTM, and ResNet18.

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Verify Models

Ensure trained model checkpoints exist at:
- `music_genre_models/cnn/best_cnn_model.pth`
- `music_genre_models/lstm/best_lstm_model.pth`
- `music_genre_models/transfer_learning/best_resnet18_finetuned_extended.pth`

### 3. Start the API

```bash
# From the backend directory
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will start on `http://localhost:8000`

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
