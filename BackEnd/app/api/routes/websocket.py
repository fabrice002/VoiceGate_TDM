"""
WebSocket endpoint for real-time voice interaction
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.audio.buffer import AudioBuffer
from services.ai.whisper_service import WhisperService
from services.ai.speaker_id import SpeakerIdentificationService
from services.ai.intent_service import IntentService
from services.ai.tts_service import TTSService

router = APIRouter()

# Initialize services
whisper = WhisperService()
speaker_id = SpeakerIdentificationService()
intent_service = IntentService()
tts = TTSService()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time voice interaction via WebSocket"""
    await websocket.accept()
    
    buffer = AudioBuffer()
    state = {
        "listening": False,
        "registering": False,
        "username": None
    }
    
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.add_chunk(data)
            
            if buffer.duration() < 1.5:
                continue
            
            audio = buffer.get_audio()
            
            # Process audio
            transcript = whisper.transcribe(audio)
            if not transcript:
                continue
            
            # Handle wake word
            if not state["listening"] and whisper.detect_wake_word(transcript):
                state["listening"] = True
                await websocket.send_json({
                    "type": "wake_word",
                    "message": "Je vous écoute..."
                })
                buffer.clear()
                continue
            
            # Handle command
            if state["listening"]:
                # Identify speaker
                speaker = speaker_id.identify_speaker(audio)
                username = speaker["username"] if speaker else "Utilisateur"
                
                # Get response
                intent, response = intent_service.process(transcript, username)
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "transcript": transcript,
                    "speaker": speaker,
                    "intent": intent,
                    "response": response
                })
                
                # TTS
                audio_response = tts.synthesize(response)
                await websocket.send_bytes(audio_response)
                
                state["listening"] = False
                buffer.clear()
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")