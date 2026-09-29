from pathlib import Path
from collections import Counter
import tempfile

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.models import GenreCNN, GenreLSTM, GenreResNet, GENRE_NAMES
from backend.preprocessing import (
    load_audio,
    extract_mel_spectrogram,
    preprocess_for_cnn,
    preprocess_for_lstm,
    preprocess_for_resnet,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_artifacts"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 16


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

    probability_dict = {
        GENRE_NAMES[i]: probability_values[i]
        for i in range(len(GENRE_NAMES))
    }

    return {
        "model": model_name,
        "genre": GENRE_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probability_dict,
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

        result = get_prediction(
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

        cnn_result = get_prediction(
            "cnn",
            audio_data,
        )

        lstm_result = get_prediction(
            "lstm",
            audio_data,
        )

        resnet_result = get_prediction(
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