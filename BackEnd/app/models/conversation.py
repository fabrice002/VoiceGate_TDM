# app/models/conversation.py
"""
Enhanced conversation models with audio transcription support
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageSource(str, Enum):
    """Source of the message"""
    VOICE = "voice"
    TEXT = "text"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Single message in a conversation with audio context"""
    # FIX: Use alias for MongoDB compatibility
    id: str = Field(..., alias="_id", description="Message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content (transcribed text)")
    source: MessageSource = Field(default=MessageSource.TEXT, description="How message was created")
    audio_duration: Optional[float] = Field(None, description="Original audio duration in seconds")
    audio_file_path: Optional[str] = Field(None, description="Path to original audio file")
    transcription_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    language: Optional[str] = Field(None, description="Language of the message")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # FIX: Add proper configuration for MongoDB
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        populate_by_name=True,  # This allows using aliases
        arbitrary_types_allowed=True
    )
    
    @field_validator('id', mode='before')
    @classmethod
    def validate_id(cls, v):
        """Convert _id to id if needed"""
        if v is None:
            return v
        if isinstance(v, dict) and '_id' in v:
            return str(v['_id'])
        return str(v)


class Conversation(BaseModel):
    """Conversation between user and AI with voice context"""
    # FIX: Use alias for MongoDB compatibility
    id: str = Field(..., alias="_id", description="Conversation ID")
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    title: Optional[str] = Field(None, description="Conversation title")
    voice_session: bool = Field(default=False, description="Is this a voice conversation?")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    message_count: int = Field(default=0, description="Number of messages")
    total_audio_duration: float = Field(default=0.0, description="Total audio duration")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # FIX: Add proper configuration for MongoDB
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,  # This allows using aliases
        arbitrary_types_allowed=True
    )
    
    @field_validator('id', mode='before')
    @classmethod
    def validate_id(cls, v):
        """Convert _id to id if needed"""
        if v is None:
            return v
        if isinstance(v, dict) and '_id' in v:
            return str(v['_id'])
        return str(v)


class VoiceAskRequest(BaseModel):
    """Request for AI conversation from voice input"""
    user_id: str = Field(..., description="User ID")
    transcribed_text: str = Field(..., description="Text from voice transcription")
    audio_duration: Optional[float] = Field(0.0, description="Audio duration in seconds")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file")
    transcription_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    language: Optional[str] = Field("fr", description="Language code")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")


class VoiceAskResponse(BaseModel):
    """Response for voice-based AI conversation"""
    success: bool = Field(True, description="Request success")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    user_message_id: Optional[str] = Field(None, description="User message ID")
    ai_message_id: Optional[str] = Field(None, description="AI response message ID")
    transcribed_text: Optional[str] = Field(None, description="Original transcribed text")
    ai_response: str = Field(..., description="AI response text")
    
    # --- AJOUT CRUCIAL ICI ---
    audio_url: Optional[str] = Field(None, description="URL to the generated TTS audio file")
    # -------------------------
    
    language: Optional[str] = Field(None, description="Language used")
    message_count: int = Field(0, description="Total messages in conversation")
    processing_time: float = Field(0.0, description="Total processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict)