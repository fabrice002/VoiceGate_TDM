# app/models/transcription.py
"""
Transcription data models
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class TranscriptionSegment(BaseModel):
    """
    Transcription segment model
    """
    id: int = Field(..., description="Segment ID")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    words: Optional[List[Dict[str, Any]]] = Field(None, description="Word-level timestamps")
    
    model_config = ConfigDict(from_attributes=True)


class TranscriptionRequest(BaseModel):
    """
    Request for audio transcription
    """
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format")
    language: Optional[str] = Field(None, description="Language code")
    task: str = Field(default="transcribe", description="Task type: transcribe or translate")
    user_id: Optional[str] = Field(None, description="User ID for conversation history")
    
    model_config = ConfigDict(from_attributes=True)


class TranscriptionResponse(BaseModel):
    """
    Response for audio transcription
    """
    success: bool = Field(..., description="Transcription success")
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected language code")
    duration: float = Field(..., description="Audio duration in seconds")
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="Segments")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    word_count: int = Field(..., description="Number of words in transcription")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")
    audio_analysis: Optional[Dict[str, Any]] = Field(None, description="Audio analysis results")
    is_mock: bool = Field(default=False, description="Whether mock transcription was used")
    original_format: str = Field(default="unknown", description="Original audio format")
    converted_file: Optional[str] = Field(None, description="Path to converted WAV file (if kept)")
    
    model_config = ConfigDict(from_attributes=True)


class ConversationEntry(BaseModel):
    """
    Conversation entry model
    """
    id: str = Field(..., description="Entry ID")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    audio_duration: float = Field(..., description="Audio duration")
    transcription: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Language code")
    timestamp: datetime = Field(..., description="Timestamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Transcription confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(from_attributes=True)


class VoiceCommand(BaseModel):
    """
    Voice command model
    """
    command_id: str = Field(..., description="Command ID")
    user_id: str = Field(..., description="User ID")
    transcription: str = Field(..., description="Transcribed command")
    intent: Optional[str] = Field(None, description="Detected intent")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    timestamp: datetime = Field(..., description="Timestamp")
    response: Optional[str] = Field(None, description="System response")
    executed: bool = Field(default=False, description="Whether command was executed")
    
    model_config = ConfigDict(from_attributes=True)