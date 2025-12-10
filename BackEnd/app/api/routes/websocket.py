# app/api/routes/websocket.py
"""
WebSocket endpoints for real-time audio streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import logging
import json
import asyncio
import numpy as np
from datetime import datetime

from services.audio.wake_word import WakeWordDetector, WakeWordResult
from services.ai.transcription import TranscriptionService
from services.database.user_repo import UserRepository
from services.ai.biometrics import VoiceBiometricsService

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept connection and store"""
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from user connections
        user_id = None
        for uid, ws in self.user_connections.items():
            if ws == websocket:
                user_id = uid
                break
        if user_id:
            del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict):
        """Send message to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")


manager = ConnectionManager()


@router.websocket("/ws/audio/{user_id}")
async def websocket_audio_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket for real-time audio streaming and processing
    """
    await manager.connect(websocket, user_id)
    
    # Initialize services
    wake_detector = WakeWordDetector()
    transcription_service = TranscriptionService()
    user_repo = UserRepository()
    biometrics = VoiceBiometricsService()
    
    audio_buffer = []
    sample_rate = 16000
    is_recording = False
    recording_start_time = None
    
    try:
        while True:
            # Receive audio data (base64 encoded)
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "audio_chunk":
                # Process audio chunk
                audio_base64 = data.get("data")
                
                # Decode base64 audio
                from services.audio.processor import AudioProcessor
                processor = AudioProcessor()
                audio_chunk = processor.decode_base64_audio(audio_base64)
                
                if not is_recording:
                    # Check for wake word
                    wake_result = wake_detector.detect_from_audio(audio_chunk, sample_rate)
                    
                    if wake_result.status.value == "detected":
                        # Wake word detected, start recording
                        is_recording = True
                        recording_start_time = datetime.now()
                        audio_buffer = [audio_chunk]
                        
                        await manager.send_personal_message({
                            "type": "wake_word_detected",
                            "word": wake_result.detected_word,
                            "confidence": wake_result.confidence,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                        logger.info(f"Wake word detected for user {user_id}")
                    
                else:
                    # Recording in progress
                    audio_buffer.append(audio_chunk)
                    
                    # Check for end of speech (silence detection)
                    if len(audio_buffer) > 10:  # Process every 10 chunks
                        # Simple silence detection
                        last_chunks = np.concatenate(audio_buffer[-5:])
                        energy = np.sqrt(np.mean(last_chunks ** 2))
                        
                        if energy < 0.01:  # Silence threshold
                            # End of speech, process recording
                            full_audio = np.concatenate(audio_buffer)
                            
                            # 1. Speaker identification
                            embedding = biometrics.extract_embedding(full_audio)
                            user = None
                            confidence = 0.0
                            
                            if embedding is not None:
                                registered_users = user_repo.get_voice_registered_users()
                                for uname in registered_users:
                                    db_user = user_repo.get_by_username(uname)
                                    if db_user and db_user.voice_embedding:
                                        score = biometrics.compare_embeddings(
                                            np.array(db_user.voice_embedding), 
                                            embedding
                                        )
                                        if score > confidence:
                                            confidence = score
                                            user = db_user
                            
                            # 2. Transcription
                            transcription_result = transcription_service.transcribe(
                                full_audio,
                                language="fr"
                            )
                            
                            # 3. Intent recognition
                            intent = await _detect_intent(transcription_result.get("text", ""))
                            
                            # 4. Generate response
                            response_text = await _generate_response(
                                transcription_result.get("text", ""),
                                intent,
                                user.username if user else "Guest"
                            )
                            
                            # Send results
                            await manager.send_personal_message({
                                "type": "transcription_result",
                                "text": transcription_result.get("text", ""),
                                "language": transcription_result.get("language", "unknown"),
                                "confidence": transcription_result.get("confidence", 0.0),
                                "user_identified": user is not None,
                                "username": user.username if user else None,
                                "identification_confidence": confidence,
                                "intent": intent,
                                "response": response_text,
                                "processing_time": data.get("processing_time", 0),
                                "timestamp": datetime.now().isoformat()
                            }, websocket)
                            
                            # Reset recording state
                            is_recording = False
                            audio_buffer = []
                            recording_start_time = None
            
            elif message_type == "control":
                # Handle control messages
                action = data.get("action")
                
                if action == "start_recording":
                    is_recording = True
                    recording_start_time = datetime.now()
                    await manager.send_personal_message({
                        "type": "recording_started",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
                elif action == "stop_recording":
                    is_recording = False
                    await manager.send_personal_message({
                        "type": "recording_stopped",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
                elif action == "cancel":
                    is_recording = False
                    audio_buffer = []
                    await manager.send_personal_message({
                        "type": "recording_cancelled",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message_type == "ping":
                # Keep-alive ping
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/logs")
async def websocket_logs_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time logs streaming
    """
    await manager.connect(websocket)
    
    try:
        # Send initial system status
        await manager.send_personal_message({
            "type": "system_status",
            "status": "connected",
            "timestamp": datetime.now().isoformat(),
            "active_connections": len(manager.active_connections)
        }, websocket)
        
        while True:
            # In production, this would stream real logs
            # For now, just keep connection alive
            await asyncio.sleep(5)
            
            # Send heartbeat
            await manager.send_personal_message({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "active_connections": len(manager.active_connections)
            }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Logs WebSocket error: {e}")
        manager.disconnect(websocket)


async def _detect_intent(text: str) -> str:
    """Simple intent detection"""
    text_lower = text.lower()
    
    intent_keywords = {
        "greeting": ["bonjour", "hello", "salut", "hi", "hey"],
        "farewell": ["au revoir", "goodbye", "bye", "à plus"],
        "question": ["quoi", "comment", "pourquoi", "quand", "où", "qui"],
        "weather": ["temps", "météo", "weather"],
        "time": ["heure", "time", "quelle heure"],
        "help": ["aide", "help", "assistance"]
    }
    
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return intent
    
    return "unknown"


async def _generate_response(text: str, intent: str, username: str = None) -> str:
    """Generate response based on intent"""
    
    responses = {
        "greeting": f"Bonjour {username if username else ''}! Comment puis-je vous aider aujourd'hui ?",
        "farewell": f"Au revoir {username if username else ''}! À bientôt.",
        "question": f"Je vais essayer de répondre à votre question: {text}",
        "weather": "Je ne peux pas accéder aux données météo pour le moment.",
        "time": f"Il est {datetime.now().strftime('%H:%M')}.",
        "help": "Je peux vous aider avec la reconnaissance vocale, la transcription et les réponses de base.",
        "unknown": "Je n'ai pas bien compris. Pouvez-vous reformuler ?"
    }
    
    return responses.get(intent, responses["unknown"])