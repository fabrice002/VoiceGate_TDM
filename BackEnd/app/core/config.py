"""
Application configuration and settings
Simplified version without pydantic-settings
"""

import os
from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    """
    Application settings
    """
    
    # Application
    APP_NAME: str = Field(default="VoiceGate AI Assistant")
    DEBUG: bool = Field(default=False)
    VERSION: str = Field(default="1.0.0")
    
    # Server
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8001)
    
    # Database
    MONGODB_URI: str = Field(default="")
    MONGO_DB_NAME: str = Field(default="voicegate_db")
    USE_MONGODB: bool = Field(default=False)
    
    # Storage
    VOICE_DB_FOLDER: str = Field(default="data/voice_embeddings")
    MODEL_CACHE_DIR: str = Field(default="data/pretrained_models")
    TEMP_DIR: str = Field(default="data/temp")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    
    # AI Models
    WHISPER_MODEL: str = Field(default="base")
    WHISPER_LANGUAGE: str = Field(default="fr")
    ECAPA_MODEL: str = Field(default="speechbrain/spkrec-ecapa-voxceleb")
    
    # Speaker Recognition
    SPEAKER_THRESHOLD: float = Field(default=0.7)
    MIN_VOICE_SAMPLES: int = Field(default=1)
    MAX_VOICE_SAMPLES: int = Field(default=10)
    
    # Audio Processing
    SAMPLE_RATE: int = Field(default=16000)
    SILENCE_THRESHOLD: float = Field(default=0.01)
    MIN_AUDIO_DURATION: float = Field(default=1.0)
    MAX_AUDIO_DURATION: float = Field(default=10.0)
    
    # Security
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024)  # 10MB
    ALLOWED_AUDIO_FORMATS: List[str] = Field(default=["wav", "mp3", "ogg", "flac", "m4a"])
    
    model_config = ConfigDict(validate_assignment=True)
    
    @classmethod
    def from_env(cls):
        """
        Create settings from environment variables
        
        Returns:
            Settings: Configured settings instance
        """
        env_vars = {}
        
        # Simple mapping of environment variables to settings
        env_mapping = {
            "APP_NAME": ("APP_NAME", str),
            "DEBUG": ("DEBUG", lambda x: x.lower() == "true"),
            "VERSION": ("VERSION", str),
            "HOST": ("HOST", str),
            "PORT": ("PORT", int),
            "MONGODB_URI": ("MONGODB_URI", str),
            "MONGO_DB_NAME": ("MONGO_DB_NAME", str),
            "USE_MONGODB": ("USE_MONGODB", lambda x: x.lower() == "true"),
            "VOICE_DB_FOLDER": ("VOICE_DB_FOLDER", str),
            "MODEL_CACHE_DIR": ("MODEL_CACHE_DIR", str),
            "TEMP_DIR": ("TEMP_DIR", str),
            "ALLOWED_ORIGINS": ("ALLOWED_ORIGINS", lambda x: x.split(",")),
            "WHISPER_MODEL": ("WHISPER_MODEL", str),
            "WHISPER_LANGUAGE": ("WHISPER_LANGUAGE", str),
            "ECAPA_MODEL": ("ECAPA_MODEL", str),
            "SPEAKER_THRESHOLD": ("SPEAKER_THRESHOLD", float),
            "MIN_VOICE_SAMPLES": ("MIN_VOICE_SAMPLES", int),
            "MAX_VOICE_SAMPLES": ("MAX_VOICE_SAMPLES", int),
        }
        
        for env_key, (attr_name, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    env_vars[attr_name] = converter(os.environ[env_key])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {env_key}={os.environ[env_key]}")
        
        return cls(**env_vars)
    
    @field_validator('VOICE_DB_FOLDER', 'MODEL_CACHE_DIR', 'TEMP_DIR', mode='before')
    @classmethod
    def validate_and_create_directories(cls, v: str) -> str:
        """
        Validate and create directories if they don't exist
        """
        if v:
            os.makedirs(v, exist_ok=True)
        return v


# Create settings instance
try:
    settings = Settings.from_env()
except Exception as e:
    print(f"Error loading settings from environment: {e}")
    print("Using default settings...")
    settings = Settings()

# Print settings in debug mode
if settings.DEBUG:
    print("=" * 50)
    print(f"{settings.APP_NAME} - Configuration")
    print("=" * 50)
    print(f"App: {settings.APP_NAME} v{settings.VERSION}")
    print(f"Debug: {settings.DEBUG}")
    print(f"Server: {settings.HOST}:{settings.PORT}")
    print(f"MongoDB: {'Enabled' if settings.USE_MONGODB else 'Disabled'}")
    print(f"Voice DB: {settings.VOICE_DB_FOLDER}")
    print("=" * 50)