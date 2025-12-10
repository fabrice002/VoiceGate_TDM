# app/main.py

"""
VoiceGate FastAPI Application - Complete Real-time Implementation
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from api.routes import monitoring
from core.config import settings

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[str, dict] = {}  # connection_id -> user data
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[client_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "user_id": None,
            "session_id": None
        }
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Find and remove from connection_data
        to_remove = []
        for client_id, data in self.connection_data.items():
            if data["websocket"] == websocket:
                to_remove.append(client_id)
        for client_id in to_remove:
            del self.connection_data[client_id]
        logger.info(f"Client disconnected")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):

        """Send a message to a specific WebSocket connection

        Args:
            message (dict): The message to send
            websocket (WebSocket): The WebSocket connection to send the message to

        Raises:
            Exception: If there is an error sending the message
        """

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
    
    async def broadcast_log(self, log_data: dict):
        """Broadcast log messages to all dashboard clients"""
        log_message = {
            "type": "log",
            "timestamp": time.time(),
            "data": log_data
        }
        await self.broadcast(log_message)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Server: {settings.HOST}:{settings.PORT}")
    
    # Initialize database - IMPORTANT: This is the key fix
    from core.database import db_connection
    from core.database import db as global_db_variable
    
    # Connect to database
    database = db_connection.connect()
    
    # Update global db reference - CRITICAL STEP
    import core.database as database_module
    database_module.db = database
    
    # Verify the database connection
    try:
        if hasattr(database, 'users'):
            count = database.users.count_documents({})
            logger.info(f"Database connected. Users collection has {count} documents")
        else:
            logger.info("Using mock database (users collection created on first access)")
    except Exception as e:
        logger.warning(f"Database test failed: {e}")
    
    logger.info(f"Database initialized: {'MongoDB' if db_connection.is_connected else 'Mock'}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    db_connection.close()

# Update app/main.py to include the new routes
def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import routes
    from api.routes import (
        health, users, voice, voice_conversation, tts, 
        transcription, voice_assistant, db_check, websocket
    )
    
    # Include core routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(users.router, prefix="/api/users", tags=["Users"])
    app.include_router(voice.router, prefix="/api/voice", tags=["Voice"])
    app.include_router(tts.router, prefix="/api/tts", tags=["Text to Speech"])
    app.include_router(transcription.router, prefix="/api/transcription", tags=["Transcription"])
    app.include_router(voice_assistant.router, prefix="/api/assistant", tags=["Voice Assistant"])
    app.include_router(voice_conversation.router, prefix="/api/voice-conversation", tags=["Voice Conversation"])
    app.include_router(db_check.router, prefix="/api/db", tags=["Database"])
    app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
    
    # Try to include new WebSocket and monitoring routes
    try:
        from api.routes import websocket, monitoring
        app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
        app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
        logger.info("WebSocket and monitoring routes loaded")
    except ImportError as e:
        logger.warning(f"WebSocket routes not available: {e}")
    
    # Include intent recognizer if available
    try:
        from services.ai.intent_recognizer import IntentRecognizer
        # Initialize intent recognizer
        app.state.intent_recognizer = IntentRecognizer(use_llm=True)
        logger.info("Intent recognizer initialized")
    except ImportError as e:
        logger.warning(f"Intent recognizer not available: {e}")
    
    @app.get("/")
    async def root():
        from core.database import db_connection
        
        endpoints = {
            "docs": "/docs",
            "health": "/health",
            "users": "/api/users",
            "voice": "/api/voice",
            "tts": "/api/tts",
            "transcription": "/api/transcription",
            "assistant": "/api/assistant",
            "voice_conversation": "/api/voice-conversation",
            "db_stats": "/api/db/stats",
            "monitoring": "/api/monitoring/metrics",
        }
        
        # Add WebSocket endpoints if available
        try:
            endpoints["ws_audio"] = "/ws/ws/audio/{user_id}"
            endpoints["ws_logs"] = "/ws/ws/logs"
            endpoints["ws_monitoring"] = "/ws/ws/monitoring"
        except:
            pass
        
        return {
            "app": settings.APP_NAME,
            "version": settings.VERSION,
            "status": "running",
            "database": "MongoDB" if db_connection.is_connected else "Mock",
            "real_time": "WebSocket available" if 'ws_audio' in endpoints else "REST only",
            "endpoints": endpoints
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )