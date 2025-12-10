# app/services/database/conversation_repo.py
"""
Unified conversation repository for VoiceGate
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

from models.conversation import Conversation, ConversationMessage, MessageSource
from core.database import db

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Unified repository for conversation database operations
    """
    
    def __init__(self):
        self.conversations_collection = db.conversations
        self.messages_collection = db.conversation_messages
    
    def create_voice_conversation(self, user_id: str, username: str, 
                                title: str = None) -> Conversation:
        """
        Create a new voice conversation
        
        Args:
            user_id: User ID
            username: Username
            title: Optional conversation title
            
        Returns:
            Conversation: Created conversation
        """
        try:
            conversation_id = str(uuid.uuid4())
            now = datetime.now()
            
            # FIX: Create document with _id
            conversation_doc = {
                "_id": conversation_id,
                "user_id": user_id,
                "username": username,
                "title": title or f"Voice conversation with {username}",
                "voice_session": True,
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "total_audio_duration": 0.0,
                "metadata": {"source": "voice"}
            }
            
            self.conversations_collection.insert_one(conversation_doc)
            logger.info(f"Created voice conversation {conversation_id} for {username}")
            
            # FIX: Pass the document directly
            return Conversation(**conversation_doc)
            
        except Exception as e:
            logger.error(f"Error creating voice conversation: {e}")
            raise
    
    def add_voice_message(self, conversation_id: str, role: str, 
                         content: str, audio_duration: float = None,
                         audio_file_path: str = None,
                         transcription_confidence: float = None,
                         language: str = None) -> ConversationMessage:
        """
        Add a voice message to conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Transcribed text content
            audio_duration: Original audio duration
            audio_file_path: Path to audio file
            transcription_confidence: Confidence score from transcription
            language: Language code
            
        Returns:
            ConversationMessage: Created message
        """
        try:
            message_id = str(uuid.uuid4())
            now = datetime.now()
            
            # FIX: Create document with _id
            message_doc = {
                "_id": message_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "source": MessageSource.VOICE.value,
                "audio_duration": audio_duration,
                "audio_file_path": audio_file_path,
                "transcription_confidence": transcription_confidence,
                "language": language,
                "timestamp": now,
                "metadata": {
                    "source": "voice_transcription",
                    "timestamp": now.isoformat()
                }
            }
            
            # Remove None values
            message_doc = {k: v for k, v in message_doc.items() if v is not None}
            
            # Insert message
            self.messages_collection.insert_one(message_doc)
            
            # Update conversation stats
            update_fields = {
                "$inc": {"message_count": 1},
                "$set": {"updated_at": now}
            }
            
            if audio_duration:
                update_fields["$inc"]["total_audio_duration"] = audio_duration
            
            self.conversations_collection.update_one(
                {"_id": conversation_id},
                update_fields
            )
            
            logger.debug(f"Added voice message to conversation {conversation_id}")
            
            # FIX: Pass the document directly
            return ConversationMessage(**message_doc)
            
        except Exception as e:
            logger.error(f"Error adding voice message: {e}")
            raise
    
    def get_conversation_messages(self, conversation_id: str, 
                                 limit: int = 50) -> List[ConversationMessage]:
        """Get all messages from a conversation"""
        try:
            cursor = self.messages_collection.find(
                {"conversation_id": conversation_id}
            ).sort("timestamp", 1).limit(limit)
            
            messages = []
            for doc in cursor:
                try:
                    # FIX: Convert MongoDB document to dict with proper ID
                    doc_dict = dict(doc)
                    messages.append(ConversationMessage(**doc_dict))
                except Exception as e:
                    logger.error(f"Error parsing message: {e}, doc keys: {list(doc.keys())}")
                    # Try a fallback approach
                    try:
                        # Ensure _id is a string
                        doc['_id'] = str(doc['_id'])
                        messages.append(ConversationMessage(**doc))
                    except Exception as e2:
                        logger.error(f"Fallback also failed: {e2}")
                        continue
            
            logger.debug(f"Retrieved {len(messages)} messages from conversation {conversation_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    def get_user_conversations(self, user_id: str, 
                              limit: int = 20) -> List[Conversation]:
        """Get all conversations for a user"""
        try:
            cursor = self.conversations_collection.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(limit)
            
            conversations = []
            for doc in cursor:
                try:
                    # FIX: Convert MongoDB document to dict
                    doc_dict = dict(doc)
                    conversations.append(Conversation(**doc_dict))
                except Exception as e:
                    logger.error(f"Error parsing conversation: {e}")
                    continue
            
            logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []
    
    def get_user_active_conversation(self, user_id: str) -> Optional[str]:
        """
        Get user's most recent active conversation ID
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[str]: Conversation ID or None
        """
        try:
            # Get most recent conversation
            cursor = self.conversations_collection.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(1)
            
            conversations = list(cursor)
            if conversations:
                return str(conversations[0]["_id"])
            return None
            
        except Exception as e:
            logger.error(f"Error getting active conversation: {e}")
            return None
    
    def get_formatted_conversation_history(self, conversation_id: str, 
                                          limit: int = 10) -> List[Dict]:
        """
        Get formatted conversation history for AI context
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to include
            
        Returns:
            List[Dict]: Formatted messages for AI prompt
        """
        messages = self.get_conversation_messages(conversation_id, limit)
        
        formatted = []
        for msg in messages:
            # Include only user and assistant messages (no system)
            if msg.role in ["user", "assistant"]:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                })
        
        logger.debug(f"Formatted {len(formatted)} messages for AI context")
        return formatted
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        try:
            doc = self.conversations_collection.find_one({"_id": conversation_id})
            if doc:
                return Conversation(**dict(doc))
            return None
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None