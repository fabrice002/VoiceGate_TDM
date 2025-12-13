# app/services/database/conversation_repo.py
"""
Unified conversation repository for VoiceGate with Offline/Mock support
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

from models.conversation import Conversation, ConversationMessage, MessageSource

logger = logging.getLogger(__name__)

class ConversationRepository:
    """
    Unified repository for conversation database operations.
    Handles both MongoDB connections and Mock/Offline fallbacks.
    """
    
    def __init__(self):
        try:
            # Import dynamique pour éviter les cycles et gérer l'absence de DB
            from core.database import db_connection
            
            # Tentative de récupération de l'objet DB
            # On cherche 'db' ou '_db' selon l'implémentation de votre connecteur
            self.db = getattr(db_connection, 'db', None) or getattr(db_connection, '_db', None)
            
            if self.db is not None:
                self.conversations_collection = self.db.conversations
                self.messages_collection = self.db.conversation_messages
            else:
                self.conversations_collection = None
                self.messages_collection = None
                logger.warning("⚠️ Database not connected. ConversationRepository running in Mock Mode.")
                
        except Exception as e:
            logger.error(f"Error initializing ConversationRepository: {e}")
            self.conversations_collection = None
            self.messages_collection = None
    
    def create_voice_conversation(self, user_id: str, username: str, 
                                title: str = None) -> Conversation:
        """Create a new voice conversation"""
        try:
            conversation_id = str(uuid.uuid4())
            now = datetime.now()
            
            conversation_doc = {
                "_id": conversation_id,
                "id": conversation_id, # Double mapping pour compatibilité Pydantic
                "user_id": user_id,
                "username": username,
                "title": title or f"Voice conversation with {username}",
                "voice_session": True,
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "total_audio_duration": 0.0,
                "metadata": {"source": "voice"},
                "messages": [] # Init empty list for mock/safety
            }
            
            # --- MOCK MODE ---
            if self.conversations_collection is None:
                logger.info(f"[MOCK] Created conversation {conversation_id} for {username}")
                # On retourne l'objet Pydantic directement
                return Conversation(**conversation_doc)
            
            # --- REAL DB MODE ---
            self.conversations_collection.insert_one(conversation_doc)
            logger.info(f"Created voice conversation {conversation_id} for {username}")
            
            return Conversation(**conversation_doc)
            
        except Exception as e:
            logger.error(f"Error creating voice conversation: {e}")
            # Fallback en cas d'erreur d'écriture DB
            return Conversation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                username=username,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
    
    def add_voice_message(self, conversation_id: str, role: str, 
                          content: str, audio_duration: float = None,
                          audio_file_path: str = None,
                          transcription_confidence: float = None,
                          language: str = None) -> ConversationMessage:
        """Add a voice message to conversation"""
        try:
            message_id = str(uuid.uuid4())
            now = datetime.now()
            
            message_doc = {
                "_id": message_id,
                "id": message_id,
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
            
            # --- MOCK MODE ---
            if self.messages_collection is None or self.conversations_collection is None:
                logger.info(f"[MOCK] Added voice message to {conversation_id}")
                return ConversationMessage(**message_doc)

            # --- REAL DB MODE ---
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
            return ConversationMessage(**message_doc)
            
        except Exception as e:
            logger.error(f"Error adding voice message: {e}")
            # Fallback pour ne pas bloquer l'UI
            return ConversationMessage(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role=role,
                content=content,
                timestamp=datetime.now()
            )
    
    def get_conversation_messages(self, conversation_id: str, 
                                 limit: int = 50) -> List[ConversationMessage]:
        """Get all messages from a conversation"""
        # MOCK MODE
        if self.messages_collection is None:
            return []

        try:
            cursor = self.messages_collection.find(
                {"conversation_id": conversation_id}
            ).sort("timestamp", 1).limit(limit)
            
            messages = []
            for doc in cursor:
                try:
                    doc_dict = dict(doc)
                    # S'assurer que l'ID est une string pour Pydantic
                    if '_id' in doc_dict:
                        doc_dict['id'] = str(doc_dict['_id'])
                    messages.append(ConversationMessage(**doc_dict))
                except Exception as e:
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    def get_user_conversations(self, user_id: str, 
                              limit: int = 20) -> List[Conversation]:
        """Get all conversations for a user"""
        # MOCK MODE
        if self.conversations_collection is None:
            return []

        try:
            cursor = self.conversations_collection.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(limit)
            
            conversations = []
            for doc in cursor:
                try:
                    doc_dict = dict(doc)
                    if '_id' in doc_dict:
                        doc_dict['id'] = str(doc_dict['_id'])
                    conversations.append(Conversation(**doc_dict))
                except Exception as e:
                    continue
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []
    
    def get_user_active_conversation(self, user_id: str) -> Optional[str]:
        """Get user's most recent active conversation ID"""
        # MOCK MODE
        if self.conversations_collection is None:
            return None

        try:
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
        """Get formatted conversation history for AI context"""
        messages = self.get_conversation_messages(conversation_id, limit)
        
        formatted = []
        for msg in messages:
            if msg.role in ["user", "assistant"]:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                })
        
        return formatted
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        # MOCK MODE
        if self.conversations_collection is None:
            return None

        try:
            doc = self.conversations_collection.find_one({"_id": conversation_id})
            if doc:
                doc_dict = dict(doc)
                if '_id' in doc_dict:
                    doc_dict['id'] = str(doc_dict['_id'])
                return Conversation(**doc_dict)
            return None
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None


    def delete_all_for_user(self, user_id: str) -> bool:
        """Delete all conversations and messages for a user"""
        
        # --- VERIFICATION ROBUSTE ---
        # Si les collections sont None ou sont des Mocks incomplets
        if (self.conversations_collection is None or 
            self.messages_collection is None or 
            not hasattr(self.conversations_collection, 'delete_many')):
            
            logger.info(f"[MOCK] Deleted all conversations for user {user_id}")
            return True # On considère que c'est un succès en mode Mock

        try:
            # 1. Trouver toutes les conversations de l'utilisateur
            cursor = self.conversations_collection.find({"user_id": user_id}, {"_id": 1})
            conversation_ids = [str(doc["_id"]) for doc in cursor]

            if not conversation_ids:
                logger.info(f"No conversations found for user {user_id} to delete.")
                return True

            # 2. Supprimer les messages liés
            self.messages_collection.delete_many({"conversation_id": {"$in": conversation_ids}})

            # 3. Supprimer les conversations
            self.conversations_collection.delete_many({"user_id": user_id})
            
            logger.info(f"Deleted {len(conversation_ids)} conversations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user conversations: {e}")
            # En production, on voudrait peut-être renvoyer False, 
            # mais ici on évite de bloquer l'utilisateur pour une erreur technique
            return False