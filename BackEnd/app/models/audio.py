# app/models/audio.py
"""
Audio data models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class AudioAnalysis(BaseModel):
    """
    Audio analysis results
    """
    duration: float = Field(..., description="Audio duration in seconds")
    sample_count: int = Field(..., description="Number of audio samples")
    sample_rate: int = Field(..., description="Audio sample rate")
    energy: float = Field(..., description="Audio energy")
    is_silent: bool = Field(..., description="Whether audio is silent")
    zero_crossing_rate: float = Field(..., description="Zero crossing rate")
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio in dB")
    max_amplitude: float = Field(..., description="Maximum amplitude")
    mean_amplitude: float = Field(..., description="Mean amplitude")


class VoiceSample(BaseModel):
    """
    Voice sample model
    """
    id: str = Field(..., description="Sample ID")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    duration: float = Field(..., description="Sample duration")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    created_at: datetime = Field(..., description="Creation timestamp")
    embedding_id: Optional[str] = Field(None, description="Embedding ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")