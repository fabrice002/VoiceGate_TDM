"""
VoiceGate FastAPI Application
Main entry point for the voice assistant backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Server: {settings.HOST}:{settings.PORT}")
    
    # Initialize database - IMPORTANT: Use the global db_connection
    from core.database import db_connection
    from core.database import db as global_db
    
    # Connect to database
    db = db_connection.connect()
    
    # Update global db reference
    import core.database as database_module
    database_module.db = db
    
    logger.info(f"Database initialized: {'MongoDB' if db_connection.is_connected else 'Mock'}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    db_connection.close()



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
    from api.routes import health, users, voice
    
    # Include core routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(users.router, prefix="/api/users", tags=["Users"])
    app.include_router(voice.router, prefix="/api/voice", tags=["Voice"])
    
    # Try to include transcription routes
    try:
        from api.routes import transcription
        app.include_router(transcription.router, prefix="/api/transcription", tags=["Transcription"])
        logger.info("Transcription routes loaded")
    except ImportError as e:
        logger.info(f"Transcription routes not available: {e}")
    except Exception as e:
        logger.warning(f"Error loading transcription routes: {e}")
    
    # Try to include voice assistant routes
    try:
        from api.routes import voice_assistant
        app.include_router(voice_assistant.router, prefix="/api/assistant", tags=["Voice Assistant"])
        logger.info("Voice assistant routes loaded")
    except ImportError as e:
        logger.info(f"Voice assistant routes not available: {e}")
    except Exception as e:
        logger.warning(f"Error loading voice assistant routes: {e}")
    
    # Try to include db_check if available
    try:
        from api.routes import db_check
        app.include_router(db_check.router, prefix="/api/db", tags=["Database"])
        logger.info("Database check routes loaded")
    except ImportError as e:
        logger.info(f"Database check routes not available: {e}")
    except Exception as e:
        logger.warning(f"Error loading db_check routes: {e}")
    
    @app.get("/")
    async def root():
        from core.database import db_connection
        endpoints = {
            "docs": "/docs",
            "health": "/health",
            "users": "/api/users",
            "voice": "/api/voice",
            "db_stats": "/api/db/stats"
        }
        
        # Add dynamic endpoints if available
        try:
            endpoints["transcription"] = "/api/transcription"
            endpoints["assistant"] = "/api/assistant"
        except:
            pass
        
        return {
            "app": settings.APP_NAME,
            "version": settings.VERSION,
            "status": "running",
            "database": "MongoDB" if db_connection.is_connected else "Mock",
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