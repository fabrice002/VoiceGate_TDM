"""
Request schemas for API endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional


class AudioUploadRequest(BaseModel):
    """
    Request schema for audio upload
    """
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format")


class UserRegistrationRequest(BaseModel):
    """
    Request schema for user registration
    """
    username: str = Field(..., min_length=2, max_length=50, description="Username")
    email: Optional[str] = Field(None, description="Optional email")


class VoiceTestRequest(BaseModel):
    """
    Request schema for voice testing
    """
    audio_base64: str = Field(..., description="Base64 encoded audio for testing")
    expected_user: Optional[str] = Field(None, description="Expected username for verification")