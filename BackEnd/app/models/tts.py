# app/models/tts.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

class VoiceType(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"

class EngineType(str, Enum):
    GTTS = "gtts"  # Google Text-to-Speech
    PYTTSX3 = "pyttsx3"  # Offline engine
    SYSTEM = "system"  # System default

class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"

class TTSPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech")
    voice: VoiceType = Field(default=VoiceType.NEUTRAL, description="Voice type")
    engine: EngineType = Field(default=EngineType.GTTS, description="TTS engine to use")
    language: str = Field(default="en", description="Language code (en, es, fr, etc.)")
    slow: bool = Field(default=False, description="Slow down speech")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    format: AudioFormat = Field(default=AudioFormat.MP3, description="Output audio format")
    
    @field_validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        return v.strip()

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_url: Optional[str] = None
    file_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None