# app/api/routes/health.py

from fastapi import APIRouter
from datetime import datetime
import os

router = APIRouter()


@router.get("/")
async def health_check():
    """Health check endpoint"""
    from core.database import db_connection
    from core.config import settings
    
    # Check storage
    storage_status = {}
    for name, path in [
        ("voice_db", settings.VOICE_DB_FOLDER),
        ("models", settings.MODEL_CACHE_DIR),
        ("temp", settings.TEMP_DIR)
    ]:
        try:
            os.makedirs(path, exist_ok=True)
            storage_status[name] = os.access(path, os.W_OK)
        except:
            storage_status[name] = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "MongoDB" if db_connection.is_connected else "Mock",
        "storage": storage_status
    }