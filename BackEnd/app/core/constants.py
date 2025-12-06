"""
Application constants
"""

from enum import Enum


class UserRole(str, Enum):
    """User roles"""
    USER = "user"
    ADMIN = "admin"


class UserStatus(str, Enum):
    """User status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class VoiceStatus(str, Enum):
    """Voice registration status"""
    REGISTERED = "registered"
    UNREGISTERED = "unregistered"
    PENDING = "pending"


class AudioQuality(str, Enum):
    """Audio quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"