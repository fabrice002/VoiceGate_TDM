from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import base64
import logging
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register")
def register_voice(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    """Register user voice"""
    try:
        from services.database.user_repo import UserRepository
        from services.ai.biometrics import VoiceBiometricsService
        from services.audio.processor import AudioProcessor
        
        user_repo = UserRepository()
        biometrics = VoiceBiometricsService()
        audio_proc = AudioProcessor()
        
        # Get user
        user = user_repo.get_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        
        # Read audio
        audio_bytes = file.file.read()
        audio_data = audio_proc.bytes_to_audio_array(audio_bytes)
        
        # Extract embedding
        embedding = biometrics.extract_embedding(audio_data)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract voice features")
        
        # Save
        success = user_repo.update_voice_embedding(username, embedding.tolist())
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save voice")
        
        return {
            "success": True,
            "username": username,
            "message": "Voice registered successfully",
            "embedding_length": len(embedding)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify")
def identify_voice(file: UploadFile = File(...)):
    """Identify speaker from voice"""
    try:
        from services.database.user_repo import UserRepository
        from services.ai.biometrics import VoiceBiometricsService
        from services.audio.processor import AudioProcessor
        
        user_repo = UserRepository()
        biometrics = VoiceBiometricsService()
        audio_proc = AudioProcessor()
        
        # Read audio
        audio_bytes = file.file.read()
        audio_data = audio_proc.bytes_to_audio_array(audio_bytes)
        
        # Extract embedding
        test_embedding = biometrics.extract_embedding(audio_data)
        if test_embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract voice features")
        
        # Compare with all users
        registered_users = user_repo.get_voice_registered_users()
        best_match = None
        best_score = 0.0
        
        for uname in registered_users:
            user = user_repo.get_by_username(uname)
            if user and user.voice_embedding:
                ref_emb = np.array(user.voice_embedding)
                score = biometrics.compare_embeddings(ref_emb, test_embedding)
                
                if score > best_score:
                    best_score = score
                    best_match = user
        
        if best_match and best_score >= biometrics.threshold:
            return {
                "identified": True,
                "username": best_match.username,
                "confidence": best_score,
                "user_id": best_match.id
            }
        else:
            return {
                "identified": False,
                "message": "No matching user found",
                "best_score": best_score
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error identifying voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))