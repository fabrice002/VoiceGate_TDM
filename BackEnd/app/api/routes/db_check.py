# app/api/routes/db_check.py
"""
Database health and data verification endpoints
"""

from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        from core.database import db_connection
        
        # Get counts from all collections
        user_count = 0
        voice_sample_count = 0
        registered_users = 0
        
        if hasattr(db_connection, '_db') and db_connection._db:
            db = db_connection._db
            if hasattr(db, 'users'):
                user_count = db.users.count_documents({})
                registered_users = db.users.count_documents({"is_voice_registered": True})
            if hasattr(db, 'voice_samples'):
                voice_sample_count = db.voice_samples.count_documents({})
        
        return {
            "timestamp": datetime.now().isoformat(),
            "database": "MongoDB" if db_connection.is_connected else "Mock",
            "collections": {
                "users": user_count,
                "voice_samples": voice_sample_count,
                "registered_users": registered_users
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_all_users():
    """List all users in database"""
    try:
        from core.database import db_connection
        
        users = []
        if hasattr(db_connection, '_db') and db_connection._db:
            db = db_connection._db
            if hasattr(db, 'users'):
                cursor = db.users.find({})
                for doc in cursor:
                    try:
                        # Convert _id to id for response
                        if "_id" in doc:
                            doc["id"] = str(doc["_id"])
                        users.append(doc)
                    except Exception as e:
                        logger.error(f"Error processing user doc: {e}")
        
        return {
            "total": len(users),
            "users": users
        }
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check/{username}")
async def check_user_data(username: str):
    """Check if user data is properly stored"""
    try:
        from services.database.user_repo import UserRepository
        
        user_repo = UserRepository()
        user = user_repo.get_by_username(username)
        
        if not user:
            return {
                "found": False,
                "message": f"User '{username}' not found"
            }
        
        # Check what data is stored
        return {
            "found": True,
            "username": user.username,
            "voice_registered": user.is_voice_registered,
            "voice_samples_count": user.voice_samples_count,
            "has_embedding": user.voice_embedding is not None,
            "embedding_length": len(user.voice_embedding) if user.voice_embedding else 0,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_updated": user.updated_at.isoformat() if user.updated_at else None
        }
        
    except Exception as e:
        logger.error(f"Error checking user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))