# app/api/routes/voice_conversation.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from pydantic import BaseModel
import logging
import time
import tempfile
import os
from typing import Optional, Dict, Any
from datetime import datetime

# Existing imports
from models.conversation import VoiceAskRequest, VoiceAskResponse
from services.ai.voice_conversation_service import VoiceConversationService

# TTS Imports
from services.ai.tts_service import TTSService
from models.tts import TTSPayload, EngineType
from core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Instantiate TTS service
tts_service = TTSService()

# Input model for text requests
class TextAskInput(BaseModel):
    user_id: str
    text: str
    language: str = "fr"
    conversation_id: Optional[str] = None

# --- UTILITY FUNCTION ---
async def generate_ai_audio(text: str, language: str) -> Optional[str]:
    """
    Generate audio for the AI response with automatic offline fallback.
    
    Attempts to use Google TTS (Online) first. If it fails (connection error),
    automatically falls back to pyttsx3 (Offline system voice).

    Args:
        text: The text content to convert to speech.
        language: The language code (e.g., 'fr', 'en').

    Returns:
        Optional[str]: The URL to the generated audio file, or None if both fail.
    """
    if not text or len(text.strip()) < 2:
        return None

    # 1. Attempt Online TTS (Google / gTTS)
    try:
        payload = TTSPayload(
            text=text,
            language=language,
            engine=EngineType.GTTS,
            voice="neutral"
        )

        success, message, filepath = await tts_service.generate_audio(payload)

        if success and filepath:
            filename = os.path.basename(filepath)
            return f"http://localhost:{settings.PORT}/api/tts/audio/{filename}"
        
        logger.warning(f"Online TTS failed ({message}). Switching to offline engine.")

    except Exception as e:
        logger.warning(f"Online TTS error ({e}). Switching to offline engine.")

    # 2. Fallback to Offline TTS (pyttsx3)
    try:
        logger.info("Attempting offline TTS generation...")
        payload_offline = TTSPayload(
            text=text,
            language=language,
            engine=EngineType.PYTTSX3, # Uses system voice
            voice="neutral"
        )

        success, message, filepath = await tts_service.generate_audio(payload_offline)

        if success and filepath:
            filename = os.path.basename(filepath)
            return f"http://localhost:{settings.PORT}/api/tts/audio/{filename}"
            
        logger.error(f"Offline TTS also failed: {message}")
        return None

    except Exception as e:
        logger.error(f"Critical error generating TTS: {e}")
        return None


@router.post("/text-ask")
async def text_ask_ai(input_data: TextAskInput):
    """
    Handle text-based AI conversation and generate TTS response.

    Args:
        input_data: The text input payload containing user_id and text.

    Returns:
        dict: The AI response object including audio URL.
    """
    try:
        # 1. Generate text response
        internal_request = VoiceAskRequest(
            user_id=input_data.user_id,
            transcribed_text=input_data.text,
            language=input_data.language,
            conversation_id=input_data.conversation_id,
            audio_duration=0,
            audio_file_path=None, 
            transcription_confidence=1.0
        )

        voice_service = VoiceConversationService()
        response = voice_service.process_voice_input(internal_request)
        
        # 2. Integrate TTS
        if response.ai_response:
            audio_url = await generate_ai_audio(response.ai_response, input_data.language)
            response.audio_url = audio_url 
            
        return response
        
    except Exception as e:
        logger.error(f"Error in text-ask: {e}")
        return {
            "ai_response": f"Backend error: {str(e)}",
            "metadata": {},
            "audio_url": None
        }


@router.post("/voice-ask", response_model=VoiceAskResponse)
async def voice_ask_ai(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: str = Form("fr"),
    conversation_id: str = Form(None)
):
    """
    Process complete voice-to-AI pipeline with Wake Word detection.
    
    Includes transcription, wake word verification, AI generation, and TTS.

    Args:
        user_id: The ID of the user.
        audio_file: The uploaded audio file (WebM/WAV).
        language: The language code.
        conversation_id: Optional existing conversation ID.

    Returns:
        VoiceAskResponse: Structured response containing AI text and audio URL.
    """
    start_time = time.time()
    
    try:
        # 1. Save audio to temporary file
        temp_audio_path = None
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            temp_audio_path = tmp.name
            audio_content = await audio_file.read()
            tmp.write(audio_content)
        
        try:
            # 2. Transcribe audio
            import librosa
            from services.ai.transcription import transcription_service
            
            try:
                # Load audio using librosa (uses ffmpeg internally)
                audio_array, _ = librosa.load(temp_audio_path, sr=16000)
            except Exception:
                transcribed_text = "..."
                audio_array = None

            if audio_array is not None:
                transcription_result = transcription_service.transcribe(
                    audio_data=audio_array,
                    language=language,
                    task="transcribe"
                )
                transcribed_text = transcription_result.get("text", "").strip()
            
            if not transcribed_text:
                transcribed_text = "..."

            # =================================================================
            # WAKE WORD LOGIC (Smart Session)
            # =================================================================
            from services.database.user_repo import UserRepository
            user_repo = UserRepository()
            
            user_input_for_ai = transcribed_text
            should_process = False
            is_session_active = False
            
            if settings.REQUIRE_WAKE_WORD:
                # A. Check if a session is already active
                last_interaction = user_repo.get_last_interaction(user_id)
                now = datetime.now()
                
                if last_interaction:
                    # Calculate elapsed time
                    delta = (now - last_interaction).total_seconds()
                    if delta < settings.WAKE_WORD_SESSION_TIMEOUT:
                        is_session_active = True
                        logger.info(f"Session active (Last activity {int(delta)}s ago)")

                # Clean text for detection
                clean_text = transcribed_text.lower().strip()
                wake_word = settings.WAKE_WORD.lower()
                
                # Check for wake word at start of sentence
                text_start = clean_text.replace(',', '').replace('.', '').strip()
                has_wake_word = text_start.startswith(wake_word)

                # B. Decision Tree
                if is_session_active:
                    # Session active: accept input
                    should_process = True
                    # Remove wake word if repeated
                    if has_wake_word:
                        user_input_for_ai = transcribed_text[len(wake_word):].strip()
                
                elif has_wake_word:
                    # New session triggered by wake word
                    should_process = True
                    user_input_for_ai = transcribed_text[len(wake_word):].strip()
                    logger.info("Wake word detected. Starting new session.")
                
                else:
                    # No session and no wake word: Reject
                    should_process = False
                    logger.info(f"Ignored input: No wake word and no active session.")

                # C. Final Cleanup
                if should_process:
                    # Remove leading punctuation
                    user_input_for_ai = user_input_for_ai.lstrip(",. ").strip()
                    # Update timer
                    user_repo.update_last_interaction(user_id)

            else:
                # Wake word feature disabled
                should_process = True

            # =================================================================

            if not should_process:
                return VoiceAskResponse(
                    success=False,
                    ai_response=f"(Say '{settings.WAKE_WORD}' to activate assistant)",
                    transcribed_text=transcribed_text,
                    message_count=0,
                    processing_time=time.time() - start_time,
                    audio_url=None 
                )

            # 3. Process AI Request
            request = VoiceAskRequest(
                user_id=user_id,
                transcribed_text=user_input_for_ai, 
                audio_duration=0, 
                audio_file_path=temp_audio_path,
                transcription_confidence=0.8,
                language=language,
                conversation_id=conversation_id
            )
            
            voice_service = VoiceConversationService()
            response = voice_service.process_voice_input(request)
            
            # 4. Generate TTS Audio
            if response.ai_response:
                audio_url = await generate_ai_audio(response.ai_response, language)
                response.audio_url = audio_url
            
            # Add metadata
            if response.metadata is None: response.metadata = {}
            response.metadata.update({
                "total_processing_time": time.time() - start_time,
                "tts_generated": bool(response.audio_url)
            })
            
            return response
            
        finally:
            # Clean up temp file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error in voice-ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{user_id}/voice")
async def get_voice_conversations(user_id: str, limit: int = 20):
    """
    Retrieve voice conversation history for a user.

    Args:
        user_id: The ID of the user.
        limit: Maximum number of conversations to return.

    Returns:
        dict: Object containing list of conversations.
    """
    try:
        from services.database.conversation_repo import ConversationRepository
        from services.database.user_repo import UserRepository
        
        # Secure ID conversion
        try:
            uid = int(user_id)
        except:
            uid = user_id

        user_repo = UserRepository()
        user = user_repo.get_by_id(uid) 
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        conversation_repo = ConversationRepository()
        conversations = conversation_repo.get_user_conversations(user_id, limit)
        
        voice_conversations = [conv for conv in conversations if conv.voice_session]
        
        return {
            "user_id": user_id,
            "username": user.username,
            "total_conversations": len(voice_conversations),
            "conversations": [conv.model_dump() for conv in voice_conversations]
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return {"conversations": []} 


@router.get("/conversation/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: str):
    """
    Get the summary of a specific voice conversation.

    Args:
        conversation_id: The unique ID of the conversation.

    Returns:
        dict: Summary object.
    """
    try:
        from services.ai.voice_conversation_service import VoiceConversationService
        
        voice_service = VoiceConversationService()
        summary = voice_service.get_conversation_summary(conversation_id)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/conversations/{user_id}/last/messages")
async def get_last_conversation_messages(user_id: str, limit: int = 50):
    """
    Retrieve messages from the user's most recent active conversation.
    Allows resuming context from previous session.

    Args:
        user_id: The ID of the user.
        limit: Maximum number of messages to return.

    Returns:
        dict: Object containing conversation ID and list of messages.
    """
    try:
        from services.database.conversation_repo import ConversationRepository
        repo = ConversationRepository()
        
        # 1. Find last active conversation
        last_conv_id = repo.get_user_active_conversation(user_id)
        
        if not last_conv_id:
            return {"messages": []} 
            
        # 2. Get messages
        messages = repo.get_conversation_messages(last_conv_id, limit=limit)
        
        # 3. Format for frontend
        return {
            "conversation_id": last_conv_id,
            "messages": [msg.model_dump() for msg in messages]
        }
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return {"messages": []}


@router.delete("/conversations/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_conversations(user_id: str):
    """
    Delete all conversations and messages associated with a user.

    Args:
        user_id: The ID of the user whose history should be cleared.

    Returns:
        None: Returns 204 No Content on success.
    """
    try:
        from services.database.conversation_repo import ConversationRepository
        repo = ConversationRepository()
        
        success = repo.delete_all_for_user(user_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete conversations")
            
        return None 

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversations for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))