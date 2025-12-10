# app/api/routes/tts.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
import io
from pathlib import Path

from models.tts import TTSPayload, TTSResponse, EngineType, AudioFormat
from services.ai.tts_service import TTSService
from core.config import settings

router = APIRouter()
tts_service = TTSService()

@router.post("/generate", response_model=TTSResponse)
async def generate_audio(
    payload: TTSPayload,
    background_tasks: BackgroundTasks
):
    """
    Convert text to speech and generate audio file
    
    - **text**: Text to convert (max 5000 characters)
    - **voice**: Voice type (male/female/neutral)
    - **engine**: TTS engine to use
    - **language**: Language code
    - **slow**: Slow down speech
    - **speed**: Speech speed multiplier (0.5 to 2.0)
    - **format**: Output audio format
    """
    
    # Generate audio
    success, message, filepath = await tts_service.generate_audio(payload)
    
    if not success or not filepath:
        raise HTTPException(status_code=500, detail=message)
    
    # Get audio information
    audio_info = tts_service.get_audio_info(filepath)
    
    # Schedule cleanup
    background_tasks.add_task(
        tts_service.cleanup_old_files,
        settings.AUDIO_FILE_RETENTION_DAYS
    )
    
    return TTSResponse(
        success=True,
        message=message,
        audio_url=f"/tts/audio/{filepath.name}",
        file_path=f'{settings.HOST}:{settings.PORT}/{str(filepath)}',
        duration_seconds=audio_info.get('duration_seconds'),
        file_size_mb=audio_info.get('file_size_mb')
    )

@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Retrieve generated audio file"""
    filepath = Path(settings.AUDIO_STORAGE_PATH) / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=filepath,
        media_type=f"audio/{filepath.suffix[1:]}",
        filename=filename
    )

@router.get("/stream")
async def stream_audio(
    text: str = Query(..., min_length=1, max_length=1000),
    engine: EngineType = Query(EngineType.GTTS),
    language: str = Query("en"),
    format: AudioFormat = Query(AudioFormat.MP3)
):
    """Stream audio directly without saving to file"""
    payload = TTSPayload(
        text=text,
        engine=engine,
        language=language,
        format=format
    )
    
    # Create temporary file
    from tempfile import NamedTemporaryFile
    import asyncio
    
    success, message, filepath = await tts_service.generate_audio(payload)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
    
    # Read file content
    with open(filepath, 'rb') as f:
        audio_content = f.read()
    
    # Delete temporary file
    filepath.unlink(missing_ok=True)
    
    return StreamingResponse(
        io.BytesIO(audio_content),
        media_type=f"audio/{format}",
        headers={"Content-Disposition": f"inline; filename=audio.{format}"}
    )

@router.get("/engines")
async def get_available_engines():
    """Get list of available TTS engines"""
    engines = [
        {"id": "gtts", "name": "Google Text-to-Speech", "online": True},
        {"id": "pyttsx3", "name": "pyttsx3 (Offline)", "online": tts_service._check_pyttsx3_available()}
    ]
    
    # Check system TTS
    try:
        import subprocess
        subprocess.run(['say', '--help'], capture_output=True, check=True)
        engines.append({"id": "system", "name": "System TTS", "online": True})
    except:
        engines.append({"id": "system", "name": "System TTS", "online": False})
    
    return {"engines": engines}

@router.get("/voices")
async def get_available_voices():
    """Get available voices for pyttsx3 engine"""
    voices = []
    
    if tts_service.pyttsx_engine:
        for voice in tts_service.pyttsx_engine.getProperty('voices'):
            voices.append({
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages
            })
    
    return {"voices": voices}