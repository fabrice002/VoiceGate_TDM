# app/api/routes/voice_conversation.py
"""
API endpoints for voice-to-AI conversations
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
import time
import tempfile
import os

from models.conversation import VoiceAskRequest, VoiceAskResponse
from schemas.responses import APIResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/voice-ask", response_model=VoiceAskResponse)
async def voice_ask_ai(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: str = Form("fr"),
    conversation_id: str = Form(None)
):
    """
    Complete voice-to-AI pipeline
    
    1. Upload voice audio
    2. Transcribe to text
    3. Send to AI with conversation context
    4. Return AI response
    
    This is the main endpoint for voice conversations.
    """
    start_time = time.time()
    
    try:
        # 1. Save audio to temp file
        temp_audio_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio_path = tmp.name
            audio_content = await audio_file.read()
            tmp.write(audio_content)
        
        try:
            # 2. Transcribe audio using existing transcription service
            from services.ai.transcription import transcription_service
            from services.audio.processor import AudioProcessor
            
            # Read and analyze audio
            audio_processor = AudioProcessor()
            audio_bytes = audio_content
            audio_array, _ = audio_processor.decode_base64_audio(
                audio_processor.encode_audio_to_base64(audio_bytes),
                return_bytes=False
            )
            
            # Transcribe
            transcription_result = transcription_service.transcribe(
                audio_data=audio_array,
                language=language,
                task="transcribe"
            )
            
            if not transcription_result.get("success", False):
                raise HTTPException(
                    status_code=400,
                    detail=f"Transcription failed: {transcription_result.get('error', 'Unknown error')}"
                )
            
            transcribed_text = transcription_result.get("text", "").strip()
            if not transcribed_text:
                raise HTTPException(
                    status_code=400,
                    detail="No speech detected in audio"
                )
            
            # 3. Prepare AI request
            request = VoiceAskRequest(
                user_id=user_id,
                transcribed_text=transcribed_text,
                audio_duration=transcription_result.get("duration", 0),
                audio_file_path=temp_audio_path,
                transcription_confidence=transcription_result.get("confidence"),
                language=language,
                conversation_id=conversation_id
            )
            
            # 4. Process through AI conversation service
            from services.ai.voice_conversation_service import VoiceConversationService
            
            voice_service = VoiceConversationService()
            response = voice_service.process_voice_input(request)
            
            # Add transcription metadata
            response.metadata.update({
                "transcription_duration": transcription_result.get("duration"),
                "transcription_confidence": transcription_result.get("confidence"),
                "transcription_word_count": transcription_result.get("word_count", 0),
                "audio_file_size": len(audio_content),
                "total_processing_time": time.time() - start_time
            })
            
            logger.info(f"Voice conversation completed: "
                       f"{len(transcribed_text)} chars → {len(response.ai_response)} chars")
            
            return response
            
        finally:
            # Cleanup temp file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice-ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text-ask", response_model=VoiceAskResponse)
async def text_ask_ai(request: VoiceAskRequest):
    """
    Text-based AI conversation with voice conversation context
    
    Useful for:
    - Testing without audio
    - Text input from UI
    - Follow-up questions via text
    """
    try:
        from services.ai.voice_conversation_service import VoiceConversationService
        
        voice_service = VoiceConversationService()
        response = voice_service.process_voice_input(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in text-ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{user_id}/voice")
async def get_voice_conversations(user_id: str, limit: int = 20):
    """
    Get all voice conversations for a user
    """
    try:
        from services.database.conversation_repo import ConversationRepository
        from services.database.user_repo import UserRepository
        
        user_repo = UserRepository()
        user = user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        conversation_repo = ConversationRepository()
        conversations = conversation_repo.get_user_conversations(user_id, limit)
        
        # Filter for voice conversations
        voice_conversations = [conv for conv in conversations if conv.voice_session]
        
        return {
            "user_id": user_id,
            "username": user.username,
            "total_conversations": len(voice_conversations),
            "conversations": [conv.model_dump() for conv in voice_conversations]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: str):
    """
    Get summary of a voice conversation
    """
    try:
        from services.ai.voice_conversation_service import VoiceConversationService
        
        voice_service = VoiceConversationService()
        summary = voice_service.get_conversation_summary(conversation_id)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))