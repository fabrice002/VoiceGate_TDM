# app/models/user.py

"""
User data models with Pydantic V2 compatibility
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime


from core.constants import UserRole, UserStatus


class UserBase(BaseModel):
    """
    Base user model
    """
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Username (alphanumeric with _ or -)"
    )
    email: Optional[EmailStr] = Field(None, description="User email")
    
    @field_validator('username', mode='before')
    @classmethod
    def username_lowercase(cls, v: str) -> str:
        """
        Convert username to lowercase
        
        Args:
            v: Username value
            
        Returns:
            str: Lowercase username
        """
        if v:
            return v.lower()
        return v
    
    model_config = ConfigDict(from_attributes=True)


class UserCreate(UserBase):
    """
    Model for creating a new user
    """
    password: Optional[str] = Field(None, min_length=8, description="Optional password")


class UserInDB(UserBase):
    """
    User model as stored in database
    """
    id: str = Field(..., alias="_id", description="User ID")
    voice_embedding: Optional[List[float]] = Field(None, description="Voice embedding vector")
    voice_samples_count: int = Field(default=0, description="Number of voice samples")
    is_voice_registered: bool = Field(default=False, description="Voice registration status")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="User status")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_voice_activity: Optional[datetime] = Field(None, description="Last voice activity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True
    )


class UserResponse(UserBase):
    """
    User model for API responses
    """
    id: str
    voice_samples_count: int
    is_voice_registered: bool
    role: UserRole
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_voice_activity: Optional[datetime]


class VoiceRegistrationRequest(BaseModel):
    """
    Request for voice registration
    """
    username: str = Field(..., description="Username to register")
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format")
    
    @field_validator('audio_format', mode='before')
    @classmethod
    def validate_audio_format(cls, v: str) -> str:
        """
        Validate audio format
        
        Args:
            v: Audio format string
            
        Returns:
            str: Validated audio format
            
        Raises:
            ValueError: If format is not supported
        """
        allowed = ["wav", "mp3", "ogg", "flac", "m4a"]
        if v.lower() not in allowed:
            raise ValueError(f"Audio format must be one of: {allowed}")
        return v.lower()


class VoiceVerificationRequest(BaseModel):
    """
    Request for voice verification
    """
    audio_base64: str = Field(..., description="Base64 encoded audio")
    username: Optional[str] = Field(None, description="Specific username to verify against")


class VoiceVerificationResponse(BaseModel):
    """
    Response for voice verification
    """
    verified: bool = Field(..., description="Verification result")
    username: Optional[str] = Field(None, description="Identified username")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    message: str = Field(..., description="Result message")
    user_id: Optional[str] = Field(None, description="User ID if verified")


class VoiceRegistrationResponse(BaseModel):
    """
    Response for voice registration
    """
    success: bool = Field(..., description="Registration success")
    username: str = Field(..., description="Registered username")
    embedding_length: int = Field(..., description="Length of voice embedding")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Voice quality score")
    message: str = Field(..., description="Result message")
    user_id: str = Field(..., description="User ID")


class UserVoiceStats(BaseModel):
    """
    Voice statistics for a user
    """
    username: str
    is_voice_registered: bool
    voice_samples_count: int
    last_registration: Optional[datetime]
    average_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    successful_verifications: int = Field(default=0)
    failed_verifications: int = Field(default=0)
    voice_quality: Optional[float] = Field(None, ge=0.0, le=1.0)


class UserListResponse(BaseModel):
    """
    Response for listing users
    """
    total: int
    page: int
    limit: int
    users: List[UserResponse]


class UserUpdate(BaseModel):
    """
    Model for updating user
    """
    email: Optional[EmailStr] = None
    status: Optional[UserStatus] = None
    metadata: Optional[Dict[str, Any]] = None