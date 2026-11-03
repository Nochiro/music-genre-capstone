"""
FastAPI request/response schemas for genre classification API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionResponse(BaseModel):
    """Single model prediction response."""
    model: str = Field(..., description="Model name (cnn, lstm, resnet)")
    genre: str = Field(..., description="Predicted genre")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Genre probabilities")


class ComparisonResponse(BaseModel):
    """Comparison of all three models on same audio."""
    predictions: List[PredictionResponse] = Field(..., description="Predictions from all models")
    consensus_genre: Optional[str] = Field(None, description="Most common predicted genre")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")


class PlaylistCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=1000)


class PlaylistUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=1000)


class PlaylistSongsRequest(BaseModel):
    song_ids: List[str] = Field(..., min_length=1, max_length=1000)


class PlaylistReorderRequest(BaseModel):
    song_ids: List[str] = Field(..., max_length=1000)
