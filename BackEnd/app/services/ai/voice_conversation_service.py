# app/services/ai/voice_conversation_service.py

"""
Service that bridges voice transcription with AI conversation
"""

import time
import logging
from typing import Dict, Any 

from models.conversation import VoiceAskRequest, VoiceAskResponse
from services.database.conversation_repo import ConversationRepository
from services.database.user_repo import UserRepository

logger = logging.getLogger(__name__)


class VoiceConversationService:
    """
    Service that handles voice-to-AI conversation flow
    """
    
    def __init__(self):
        self.conversation_repo = ConversationRepository()
        self.user_repo = UserRepository()
        self.ai_service = None
    
    def _get_ai_service(self):
        """Lazy initialization of AI service"""
        if self.ai_service is None:
            try:
                # Import your HuggingFace model
                from services.ai.pre_trained_model import get_hf_model
                self.ai_service = get_hf_model()
                
                # Ensure model is loaded
                if not self.ai_service.is_loaded:
                    logger.info("Loading AI model...")
                    self.ai_service.load_model()
                    
            except ImportError as e:
                logger.error(f"AI model module not found: {e}")
                self.ai_service = None
            except Exception as e:
                logger.error(f"Failed to initialize AI service: {e}")
                self.ai_service = None
        
        return self.ai_service
    
    def process_voice_input(self, request: VoiceAskRequest) -> VoiceAskResponse:
        """
        Main method: Process voice input through the full pipeline
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing voice input for user {request.user_id}: "
                       f"'{request.transcribed_text[:50]}...'")
            
            # 1. Verify user exists
            user = self.user_repo.get_by_id(request.user_id)
            if not user:
                logger.warning(f"User {request.user_id} not found")
                return VoiceAskResponse(
                    success=False,
                    conversation_id="error",
                    user_message_id="error",
                    ai_message_id="error",
                    transcribed_text=request.transcribed_text,
                    ai_response="User not found. Please register first.",
                    language=request.language,
                    message_count=0,
                    processing_time=time.time() - start_time,
                    metadata={
                        "error": "user_not_found",
                        "user_id": request.user_id
                    }
                )
            
            # 2. Get or create conversation
            conversation_id = request.conversation_id
            if not conversation_id:
                conversation_id = self.conversation_repo.get_user_active_conversation(request.user_id)
            
            is_new_conversation = False
            if not conversation_id:
                # Create new voice conversation
                conversation = self.conversation_repo.create_voice_conversation(
                    user_id=request.user_id,
                    username=user.username
                )
                conversation_id = conversation.id
                is_new_conversation = True
                logger.info(f"Created new conversation: {conversation_id}")
            else:
                logger.info(f"Using existing conversation: {conversation_id}")
            
            # 3. Save user's transcribed message
            try:
                user_message = self.conversation_repo.add_voice_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.transcribed_text,
                    audio_duration=request.audio_duration,
                    audio_file_path=request.audio_file_path,
                    transcription_confidence=request.transcription_confidence,
                    language=request.language
                )
                user_message_id = user_message.id
                logger.debug(f"Saved user message with ID: {user_message_id}")
            except Exception as e:
                logger.error(f"Failed to save user message: {e}")
                # Continue anyway, but mark as error
                user_message_id = "save_error"
            
            # 4. Get conversation history for AI context
            try:
                history_messages = self.conversation_repo.get_formatted_conversation_history(
                    conversation_id, limit=10
                )
                
                # Convert to AI service format (list of tuples)
                ai_history = []
                current_user_msg = None
                
                for msg in history_messages:
                    if msg["role"] == "user":
                        current_user_msg = msg["content"]
                    elif msg["role"] == "assistant" and current_user_msg is not None:
                        ai_history.append((current_user_msg, msg["content"]))
                        current_user_msg = None
                
                logger.debug(f"Prepared {len(ai_history)} conversation pairs for AI")
                
            except Exception as e:
                logger.error(f"Error preparing conversation history: {e}")
                ai_history = []
            
            # 5. Generate AI response
            ai_service = self._get_ai_service()
            ai_response_text = ""
            
            if ai_service and hasattr(ai_service, 'is_loaded') and ai_service.is_loaded:
                try:
                    logger.debug(f"Generating AI response with history length: {len(ai_history)}")
                    ai_response_text = ai_service.generate_response(
                        user_input=request.transcribed_text,
                        history=ai_history,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9
                    )
                    logger.info(f"AI generated response: {ai_response_text[:50]}...")
                except Exception as ai_error:
                    logger.error(f"AI generation error: {ai_error}")
                    ai_response_text = "I apologize, but I'm having trouble generating a response right now."
            else:
                # Fallback response
                logger.warning("AI service not available, using fallback")
                ai_response_text = f"I received your message: '{request.transcribed_text}'. This is a fallback response."
            
            # 6. Save AI response
            try:
                ai_message = self.conversation_repo.add_voice_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=ai_response_text,
                    language=request.language
                )
                ai_message_id = ai_message.id
                logger.debug(f"Saved AI message with ID: {ai_message_id}")
            except Exception as e:
                logger.error(f"Failed to save AI message: {e}")
                ai_message_id = "save_error"
            
            # 7. Get updated message count
            try:
                conversation = self.conversation_repo.get_conversation(conversation_id)
                message_count = conversation.message_count if conversation else 0
            except:
                message_count = 0
            
            # 8. Calculate total processing time
            processing_time = time.time() - start_time
            
            logger.info(f"Voice conversation processed in {processing_time:.2f}s: "
                       f"User: {len(request.transcribed_text)} chars → "
                       f"AI: {len(ai_response_text)} chars")
            
            return VoiceAskResponse(
                success=True,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                ai_message_id=ai_message_id,
                transcribed_text=request.transcribed_text,
                ai_response=ai_response_text,
                language=request.language,
                message_count=message_count,
                processing_time=processing_time,
                metadata={
                    "is_new_conversation": is_new_conversation,
                    "history_messages_count": len(history_messages) if 'history_messages' in locals() else 0,
                    "user_audio_duration": request.audio_duration,
                    "transcription_confidence": request.transcription_confidence,
                    "username": user.username
                }
            )
            
        except Exception as e:
            logger.error(f" Error processing voice input: {e}", exc_info=True)
            processing_time = time.time() - start_time
            
            return VoiceAskResponse(
                success=False,
                conversation_id=request.conversation_id or "error",
                user_message_id="error",
                ai_message_id="error",
                transcribed_text=request.transcribed_text,
                ai_response=f"System error: {str(e)[:100]}",
                language=request.language,
                message_count=0,
                processing_time=processing_time,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of a voice conversation"""
        try:
            messages = self.conversation_repo.get_conversation_messages(conversation_id)
            
            # Calculate statistics
            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]
            
            total_audio_duration = sum(
                m.audio_duration for m in user_messages 
                if m.audio_duration is not None
            )
            
            return {
                "conversation_id": conversation_id,
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "total_audio_duration": total_audio_duration,
                "first_message": messages[0].content[:100] + "..." if messages else None,
                "last_message": messages[-1].content[:100] + "..." if messages else None,
                "has_voice_messages": any(m.source == "voice" for m in messages)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {"error": str(e)}