"""
Transcription API – Universal audio format support
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import logging
import time
import uuid
from datetime import datetime
import os
import traceback

from models.transcription import TranscriptionRequest, TranscriptionResponse
from schemas.responses import APIResponse

router = APIRouter()
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# 1) MAIN ENDPOINTS (UPLOAD / BASE64)
# ────────────────────────────────────────────────────────────────

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(None),
    user_id: str = Form(None),
    task: str = Form("transcribe"),
    keep_converted_file: str = Form("true")
):
    """
    Transcribe ANY uploaded audio file to text.
    Supports: MP3, WAV, FLAC, OGG, M4A, AAC, OPUS, etc.
    """
    start_time = time.time()

    try:
        from services.ai.transcription import transcription_service

        if task not in ["transcribe", "translate"]:
            raise HTTPException(400, "Task must be 'transcribe' or 'translate'")

        keep_file_bool = keep_converted_file.lower() in ['true', '1', 'yes', 'y', 't']
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(400, "Uploaded file is empty")

        logger.info(f"Processing file: {file.filename}, keep={keep_file_bool}")

        # Call transcription service
        transcription_result = transcription_service.transcribe_bytes(
            audio_bytes=file_content,
            original_filename=file.filename,
            language=language,
            task=task,
            keep_converted_file=keep_file_bool
        )

        processing_time = time.time() - start_time

        if not transcription_result.get("success", False):
            raise HTTPException(500, f"Transcription failed: {transcription_result.get('error')}")

        # Build response
        response = _build_response(transcription_result, language, processing_time)

        # Save conversation if requested
        if user_id:
            await _save_conversation_entry(user_id, response)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@router.post("/transcribe-base64", response_model=TranscriptionResponse)
async def transcribe_audio_base64(request: TranscriptionRequest):
    """
    Transcribe Base64-encoded audio to text.
    """
    start_time = time.time()

    try:
        from services.audio.processor import AudioProcessor
        from services.ai.transcription import transcription_service

        if request.task not in ["transcribe", "translate"]:
            raise HTTPException(400, "Task must be 'transcribe' or 'translate'")

        audio_bytes = AudioProcessor().decode_base64_simple(request.audio_base64)

        if len(audio_bytes) == 0:
            raise HTTPException(400, "Decoded audio is empty")

        transcription_result = transcription_service.transcribe_bytes(
            audio_bytes=audio_bytes,
            original_filename=f"audio.{request.audio_format}",
            language=request.language,
            task=request.task,
            keep_converted_file=True
        )

        processing_time = time.time() - start_time

        if not transcription_result.get("success"):
            raise HTTPException(500, f"Transcription failed: {transcription_result.get('error')}")

        response = _build_response(
            transcription_result,
            request.language,
            processing_time,
            original_format=request.audio_format
        )

        if request.user_id:
            await _save_conversation_entry(request.user_id, response)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Transcription failed: {str(e)}")


# ────────────────────────────────────────────────────────────────
# 2) FILE MANAGEMENT (LIST / CLEANUP)
# ────────────────────────────────────────────────────────────────

@router.get("/list-converted-files")
async def list_converted_files():
    """List all converted WAV files in the data directory."""
    try:
        from pathlib import Path

        data_dir = Path("data")

        if not data_dir.exists():
            return {"message": "Data directory does not exist", "files": []}

        wav_files = list(data_dir.rglob("*.wav"))

        files_info = [{
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "size_mb": f.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        } for f in wav_files]

        return {
            "total_files": len(files_info),
            "files": files_info[:50]
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))


@router.delete("/cleanup-converted-files")
async def cleanup_converted_files(older_than_hours: int = Query(24, ge=1)):
    """Delete converted WAV files older than X hours."""
    try:
        from pathlib import Path
        import shutil

        data_dir = Path("data")
        if not data_dir.exists():
            return {"message": "Data directory does not exist", "deleted": 0}

        cutoff = time.time() - (older_than_hours * 3600)
        deleted = 0

        for f in data_dir.rglob("*.wav"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1

        # Delete temp folders
        for t in data_dir.rglob("temp*"):
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)

        return {
            "deleted_files": deleted,
            "older_than_hours": older_than_hours
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))


# ────────────────────────────────────────────────────────────────
# 3) SERVICE INFO & HEALTH
# ────────────────────────────────────────────────────────────────

@router.get("/formats")
async def get_supported_formats():
    """Return supported audio formats."""
    try:
        from services.ai.transcription import transcription_service

        info = transcription_service.get_supported_formats()

        return {
            "supported": "All formats supported via conversion to WAV",
            "converter_status": info["converter_status"],
            "recommended_tool": "FFmpeg",
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))


@router.get("/model-info")
async def model_info():
    """Return model metadata."""
    try:
        from services.ai.transcription import transcription_service
        return transcription_service.get_model_info()
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))


@router.get("/health")
async def transcription_health():
    """Health check for transcription backend."""
    try:
        from services.ai.transcription import transcription_service

        info = transcription_service.get_model_info()

        return {
            "status": "healthy" if info["is_loaded"] else "degraded",
            "model_name": info["model_name"],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(e)
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/status")
async def get_transcription_status():
    """Return global system status."""
    try:
        from pathlib import Path
        from services.ai.transcription import transcription_service

        model = transcription_service.get_model_info()
        data_dir = Path("data")
        wav_files = list(data_dir.rglob("*.wav")) if data_dir.exists() else []

        size = sum(f.stat().st_size for f in wav_files)

        return {
            "service": "transcription",
            "model": model,
            "storage": {
                "files": len(wav_files),
                "size_mb": size / (1024 * 1024)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))


@router.get("/test")
async def test_transcription():
    """Internal test: generate small silent WAV and transcribe."""
    try:
        from services.ai.transcription import transcription_service
        import wave
        import io

        sample_rate = 16000
        duration = 0.1
        frames = int(sample_rate * duration)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b'\x00\x00' * frames)

        result = transcription_service.transcribe_bytes(
            audio_bytes=buffer.getvalue(),
            original_filename="test.wav",
            language="en",
            task="transcribe",
            keep_converted_file=False
        )

        return {
            "success": True,
            "model_loaded": not result.get("is_mock", False),
            "transcription_result": result
        }

    except Exception as e:
        logger.error(e)
        return {"success": False, "error": str(e)}


# ────────────────────────────────────────────────────────────────
# 4) INTERNAL UTILITIES
# ────────────────────────────────────────────────────────────────

def _build_response(result, language, processing_time, original_format=None):
    """
    Build a standard TranscriptionResponse object.
    """
    response = TranscriptionResponse(
        success=True,
        text=result.get("text", ""),
        language=result.get("language", language or "unknown"),
        duration=result.get("audio_duration", 0),
        segments=[],
        confidence=result.get("confidence"),
        word_count=result.get("word_count", 0),
        processing_time=processing_time,
        model_info=result.get("model_info"),
        audio_analysis=None,
        is_mock=result.get("is_mock", False),
        original_format=result.get("original_format", original_format),
        converted_file=result.get("converted_file")
    )

    # Add segments if any
    segments = result.get("segments", [])
    if segments:
        from models.transcription import TranscriptionSegment
        response.segments = [
            TranscriptionSegment(
                id=i,
                start=s.get("start", 0),
                end=s.get("end", 0),
                text=s.get("text", ""),
                confidence=s.get("confidence")
            )
            for i, s in enumerate(segments)
        ]

    return response


async def _save_conversation_entry(user_id: str, transcription: TranscriptionResponse):
    """
    Save transcription to user history in DB.
    """
    try:
        from services.database.user_repo import UserRepository
        from core.database import db

        user_repo = UserRepository()
        user = user_repo.get_by_id(user_id)

        if user and hasattr(db, "conversations"):
            db.conversations.insert_one({
                "_id": str(uuid.uuid4()),
                "user_id": user_id,
                "username": user.username,
                "audio_duration": transcription.duration,
                "transcription": transcription.text,
                "language": transcription.language,
                "timestamp": datetime.now(),
                "confidence": transcription.confidence or 0.0,
                "metadata": {
                    "processing_time": transcription.processing_time,
                    "word_count": transcription.word_count,
                    "model_info": transcription.model_info,
                    "is_mock": transcription.is_mock,
                    "original_format": transcription.original_format
                }
            })

    except Exception as e:
        logger.error(f"Error saving conversation entry: {e}")
