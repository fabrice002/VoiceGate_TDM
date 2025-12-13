# app/core/config.py
"""
Application configuration and settings
"""

import os
from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import load_dotenv
from typing import Optional

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
    AUDIO_STORAGE_PATH: str = Field(default="data/audio_files")
    MODEL_CACHE_DIR: str = Field(default="data/pretrained_models")
    TEMP_DIR: str = Field(default="data/temp")
    DATA_DIR: str = Field(default="data")  # Added from first config
    ADD_DATA_URL_PREFIX: bool = Field(default=False)  # Add this line
    MAX_AUDIO_FILE_SIZE_MB: int = Field(default=10)
    AUDIO_FILE_RETENTION_DAYS: int = Field(default=7)
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    
    # AI Models
    WHISPER_MODEL: str = Field(default="base")
    WHISPER_LANGUAGE: str = Field(default="fr")
    ECAPA_MODEL: str = Field(default="speechbrain/spkrec-ecapa-voxceleb")
    
    # Hugging Face Model (from first config)
    HF_MODEL_NAME: str = Field(default="openai-community/gpt2")
    HF_TOKEN: str = Field(default="")
    MAX_CONTEXT_LENGTH: int = Field(default=1024)
    
    # Speaker Recognition
    SPEAKER_THRESHOLD: float = Field(default=0.7)
    MIN_VOICE_SAMPLES: int = Field(default=1)
    MAX_VOICE_SAMPLES: int = Field(default=10)
    
    
    WAKE_WORD: str = Field(default="voicegate")  # Le mot magique (en minuscule)
    REQUIRE_WAKE_WORD: bool = Field(default=True)
    WAKE_WORD_SESSION_TIMEOUT: int = Field(default=240)
    
    
    # Audio Processing
    SAMPLE_RATE: int = Field(default=16000)
    SILENCE_THRESHOLD: float = Field(default=0.01)
    MIN_AUDIO_DURATION: float = Field(default=1.0)
    MAX_AUDIO_DURATION: float = Field(default=10.0)
    
    # TTS Settings
    DEFAULT_VOICE: str = Field(default="fr")
    DEFAULT_SPEED: float = Field(default=1.0)
    DEFAULT_VOLUME: float = Field(default=1.0)
    
    # Cache
    REDIS_URL: Optional[str] = None
    CACHE_TTL_SECONDS: int = 3600
    
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
            "AUDIO_STORAGE_PATH": ("AUDIO_STORAGE_PATH", str),
            "MODEL_CACHE_DIR": ("MODEL_CACHE_DIR", str),
            "REDIS_URL": ("REDIS_URL", str),
            "CACHE_TTL_SECONDS": ("CACHE_TTL_SECONDS", str),
            "TEMP_DIR": ("TEMP_DIR", str),
            "DATA_DIR": ("DATA_DIR", str),
            "ALLOWED_ORIGINS": ("ALLOWED_ORIGINS", lambda x: x.split(",") if x else ["*"]),
            "WHISPER_MODEL": ("WHISPER_MODEL", str),
            "WHISPER_LANGUAGE": ("WHISPER_LANGUAGE", str),
            "ECAPA_MODEL": ("ECAPA_MODEL", str),
            "SPEAKER_THRESHOLD": ("SPEAKER_THRESHOLD", float),
            "MIN_VOICE_SAMPLES": ("MIN_VOICE_SAMPLES", int),
            "MAX_AUDIO_FILE_SIZE_MB": ("MAX_AUDIO_FILE_SIZE_MB", int),
            "AUDIO_FILE_RETENTION_DAYS": ("AUDIO_FILE_RETENTION_DAYS", int),
            "MAX_VOICE_SAMPLES": ("MAX_VOICE_SAMPLES", int),
            "HF_MODEL_NAME": ("HF_MODEL_NAME", str),
            "HF_TOKEN": ("HF_TOKEN", str),
            "MAX_CONTEXT_LENGTH": ("MAX_CONTEXT_LENGTH", int),
            "SAMPLE_RATE": ("SAMPLE_RATE", int),
            "SILENCE_THRESHOLD": ("SILENCE_THRESHOLD", float),
            "MIN_AUDIO_DURATION": ("MIN_AUDIO_DURATION", float),
            "MAX_AUDIO_DURATION": ("MAX_AUDIO_DURATION", float),
            "DEFAULT_VOICE": ("DEFAULT_VOICE", str),
            "DEFAULT_SPEED": ("DEFAULT_SPEED", float),
            "DEFAULT_VOLUME": ("DEFAULT_VOLUME", float),
            "MAX_FILE_SIZE": ("MAX_FILE_SIZE", int),
            "WAKE_WORD_SESSION_TIMEOUT": ("WAKE_WORD_SESSION_TIMEOUT", int),
            "WAKE_WORD": ("WAKE_WORD", str),
            "REQUIRE_WAKE_WORD": ("REQUIRE_WAKE_WORD", bool),
        }
        
        # Handle ALLOWED_AUDIO_FORMATS specially
        if "ALLOWED_AUDIO_FORMATS" in os.environ:
            env_vars["ALLOWED_AUDIO_FORMATS"] = [
                fmt.strip() for fmt in os.environ["ALLOWED_AUDIO_FORMATS"].split(",")
            ]
        
        for env_key, (attr_name, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    env_vars[attr_name] = converter(os.environ[env_key])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {env_key}={os.environ[env_key]}: {e}")
        
        return cls(**env_vars)
    
    @field_validator('VOICE_DB_FOLDER', 'MODEL_CACHE_DIR', 'TEMP_DIR', 'DATA_DIR', mode='before')
    @classmethod
    def validate_and_create_directories(cls, v: str) -> str:
        """
        Validate and create directories if they don't exist
        """
        if v:
            os.makedirs(v, exist_ok=True)
        return v
    
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def validate_allowed_origins(cls, v):
        """
        Validate ALLOWED_ORIGINS format
        """
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",")]
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
    print(f"Model Cache: {settings.MODEL_CACHE_DIR}")
    print(f"Hugging Face Model: {settings.HF_MODEL_NAME}")
    print("=" * 50)