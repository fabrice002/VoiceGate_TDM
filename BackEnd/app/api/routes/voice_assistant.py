# app/api/routes/voice_assistant.py - Fix imports
"""
Voice assistant endpoints combining recognition and transcription
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
import logging
import uuid
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/process")
async def process_voice_command(file: UploadFile = File(...)):
    """
    Process voice command: identify speaker and transcribe
    """
    try:
        # Import services within function to handle missing dependencies
        from services.database.user_repo import UserRepository
        from services.ai.biometrics import VoiceBiometricsService
        from services.audio.processor import AudioProcessor
        
        # Initialize services
        user_repo = UserRepository()
        biometrics = VoiceBiometricsService()
        audio_proc = AudioProcessor()
        
        # Read audio
        audio_bytes = await file.read()
        audio_data = audio_proc.bytes_to_audio_array(audio_bytes)
        
        # Step 1: Speaker Identification
        test_embedding = biometrics.extract_embedding(audio_data)
        
        if test_embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract voice features")
        
        # Compare with registered users
        registered_users = user_repo.get_voice_registered_users()
        best_match = None
        best_score = 0.0
        
        for uname in registered_users:
            user = user_repo.get_by_username(uname)
            if user and user.voice_embedding:
                ref_emb = user.voice_embedding
                if ref_emb:  # Check if not None
                    import numpy as np
                    score = biometrics.compare_embeddings(np.array(ref_emb), test_embedding)
                    
                    if score > best_score:
                        best_score = score
                        best_match = user
        
        identification_result = {
            "identified": best_match is not None and best_score >= biometrics.threshold,
            "username": best_match.username if best_match else None,
            "user_id": best_match.id if best_match else None,
            "confidence": best_score,
            "threshold": biometrics.threshold
        }
        
        # Step 2: Transcription
        # Try to import transcription service, fallback to mock
        try:
            from services.ai.transcription import TranscriptionService
            transcribe_service = TranscriptionService()
            audio_for_transcription = audio_proc.prepare_for_transcription(audio_data)
            
            # Use user's preferred language if identified
            language = None
            if best_match and hasattr(best_match, 'metadata'):
                language = best_match.metadata.get('preferred_language')
            
            transcription_result = transcribe_service.transcribe(
                audio_for_transcription,
                language=language
            )
        except ImportError:
            # Fallback to mock transcription
            logger.warning("Transcription service not available, using mock")
            transcription_result = {
                "text": f"Mock transcription for {len(audio_data)} samples",
                "language": "en",
                "confidence": 0.8,
                "segments": [],
                "duration": len(audio_data) / 16000
            }
        
        # Step 3: Save command to history
        command_id = str(uuid.uuid4())
        command_entry = {
            "_id": command_id,
            "user_id": best_match.id if best_match else None,
            "username": best_match.username if best_match else None,
            "transcription": transcription_result.get("text", ""),
            "language": transcription_result.get("language", "unknown"),
            "identification_confidence": best_score,
            "transcription_confidence": transcription_result.get("confidence", 0.0),
            "timestamp": datetime.now(),
            "audio_duration": len(audio_data) / 16000,
            "processed": False
        }
        
        # Save to database
        from core.database import db_connection
        if db_connection.db and hasattr(db_connection.db, 'voice_commands'):
            db_connection.db.voice_commands.insert_one(command_entry)
        
        # Step 4: Prepare response
        response = {
            "command_id": command_id,
            "identification": identification_result,
            "transcription": {
                "text": transcription_result.get("text", ""),
                "language": transcription_result.get("language", "unknown"),
                "duration": command_entry["audio_duration"],
                "word_count": len(transcription_result.get("text", "").split()),
                "segments": transcription_result.get("segments", []),
                "confidence": transcription_result.get("confidence", 0.0)
            },
            "timestamp": command_entry["timestamp"].isoformat(),
            "next_action": "process_command" if identification_result["identified"] else "require_identification"
        }
        
        # Add greeting if user identified
        if identification_result["identified"]:
            username = identification_result["username"]
            response["greeting"] = f"Hello {username}! How can I help you today?"
        
        logger.info(f"Processed voice command from {identification_result['username'] or 'unknown'}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
async def get_command_history(user_id: str, limit: int = 10):
    """
    Get voice command history for a user
    """
    try:
        from core.database import db_connection
        
        if not db_connection.db or not hasattr(db_connection.db, 'voice_commands'):
            return {"commands": [], "count": 0}
        
        cursor = db_connection.db.voice_commands.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)
        
        commands = list(cursor)
        
        # Convert ObjectId to string
        for cmd in commands:
            if "_id" in cmd:
                cmd["id"] = str(cmd["_id"])
        
        return {
            "commands": commands,
            "count": len(commands)
        }
        
    except Exception as e:
        logger.error(f"Error getting command history: {e}")
        raise HTTPException(status_code=500, detail=str(e))