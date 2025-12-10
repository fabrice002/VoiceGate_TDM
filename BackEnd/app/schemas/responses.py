# app/schemas/responses.py

"""
Response schemas for API endpoints
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class APIResponse(BaseModel):
    """
    Generic API response
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class VoiceTestResult(BaseModel):
    """
    Voice test result
    """
    test_id: str
    timestamp: datetime
    audio_duration: float
    test_type: str
    result: Dict[str, Any]
    confidence: Optional[float]
    verified: bool


class SystemHealth(BaseModel):
    """
    System health status
    """
    status: str
    timestamp: datetime
    database: str
    biometrics: str
    storage: Dict[str, bool]