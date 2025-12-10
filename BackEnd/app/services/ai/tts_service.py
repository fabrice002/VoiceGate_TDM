# app/services/ai/tts_service.py

from pathlib import Path
from typing import Tuple, Optional
import uuid
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment 

from models.tts import TTSPayload, AudioFormat
from core.config import settings

class TTSService:
    """Text-to-Speech service supporting multiple engines"""
    
    def __init__(self):
        self.storage_path = Path(settings.AUDIO_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize pyttsx3 engine
        self.pyttsx_engine = None
        if self._check_pyttsx3_available():
            self.pyttsx_engine = pyttsx3.init()
    
    def _check_pyttsx3_available(self) -> bool:
        """Check if pyttsx3 is available on the system"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say("test")
            engine.stop()
            return True
        except:
            return False
    
    async def generate_audio(self, payload: TTSPayload) -> Tuple[bool, str, Optional[Path]]:
        """Generate audio from text using specified engine"""
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.{payload.format}"
        filepath = self.storage_path / filename
        
        try:
            if payload.engine == "gtts":
                success = await self._generate_with_gtts(payload, filepath)
            elif payload.engine == "pyttsx3":
                success = await self._generate_with_pyttsx3(payload, filepath)
            else:
                success = await self._generate_with_system(payload, filepath)
            
            if success and filepath.exists():
                return True, "Audio generated successfully", filepath
            else:
                return False, "Failed to generate audio", None
                
        except Exception as e:
            return False, f"Error generating audio: {str(e)}", None
    
    async def _generate_with_gtts(self, payload: TTSPayload, filepath: Path) -> bool:
        """Generate audio using Google Text-to-Speech"""
        try:
            tts = gTTS(
                text=payload.text,
                lang=payload.language,
                slow=payload.slow
            )
            tts.save(str(filepath))
            
            # Adjust speed if needed
            if payload.speed != 1.0:
                await self._adjust_audio_speed(filepath, payload.speed)
                
            return True
        except Exception as e:
            print(f"gTTS Error: {e}")
            return False
    
    async def _generate_with_pyttsx3(self, payload: TTSPayload, filepath: Path) -> bool:
        """Generate audio using pyttsx3 (offline)"""
        if not self.pyttsx_engine:
            return False
        
        try:
            # Save to temporary WAV file
            temp_wav = filepath.with_suffix('.wav')
            
            self.pyttsx_engine.save_to_file(payload.text, str(temp_wav))
            self.pyttsx_engine.runAndWait()
            
            # Convert to desired format
            audio = AudioSegment.from_wav(str(temp_wav))
            
            # Adjust speed
            if payload.speed != 1.0:
                audio = self._change_speed_pydub(audio, payload.speed)
            
            # Export to desired format
            audio.export(str(filepath), format=payload.format)
            
            # Clean up temporary file
            temp_wav.unlink(missing_ok=True)
            
            return True
        except Exception as e:
            print(f"pyttsx3 Error: {e}")
            return False
    
    async def _generate_with_system(self, payload: TTSPayload, filepath: Path) -> bool:
        """Generate audio using system TTS (macOS/Linux)"""
        try:
            import subprocess
            
            # Use say command on macOS
            temp_wav = filepath.with_suffix('.wav')
            
            subprocess.run([
                'say', 
                '-v', self._get_system_voice(payload.language),
                '-o', str(temp_wav),
                payload.text
            ], check=True)
            
            # Convert to desired format
            audio = AudioSegment.from_file(str(temp_wav), format='wav')
            
            # Adjust speed
            if payload.speed != 1.0:
                audio = self._change_speed_pydub(audio, payload.speed)
            
            audio.export(str(filepath), format=payload.format)
            
            # Clean up
            temp_wav.unlink(missing_ok=True)
            
            return True
        except Exception as e:
            print(f"System TTS Error: {e}")
            return await self._generate_with_gtts(payload, filepath)  # Fallback to gTTS
    
    def _get_system_voice(self, language: str) -> str:
        """Get appropriate system voice based on language"""
        voices = {
            'en': 'Alex',
            'es': 'Juan',
            'fr': 'Thomas',
            'de': 'Anna'
        }
        return voices.get(language, 'Alex')
    
    async def _adjust_audio_speed(self, filepath: Path, speed: float):
        """Adjust audio speed using pydub"""
        try:
            audio = AudioSegment.from_file(str(filepath))
            adjusted_audio = self._change_speed_pydub(audio, speed)
            adjusted_audio.export(str(filepath), format=filepath.suffix[1:])
        except Exception as e:
            print(f"Speed adjustment error: {e}")
    
    def _change_speed_pydub(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """Change audio speed while maintaining pitch"""
        # Simple speed change (affects pitch)
        sound_with_altered_frame_rate = audio._spawn(
            audio.raw_data, 
            overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        
        # Set correct frame rate to maintain duration
        return sound_with_altered_frame_rate.set_frame_rate(audio.frame_rate)
    
    def get_audio_info(self, filepath: Path) -> dict:
        """Get audio file information"""
        try:
            audio = AudioSegment.from_file(str(filepath))
            return {
                'duration_seconds': len(audio) / 1000.0,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate
            }
        except:
            return {}
    
    def cleanup_old_files(self, max_age_days: int = 7):
        """Clean up audio files older than specified days"""
        import time
        current_time = time.time()
        
        for file in self.storage_path.glob('*.*'):
            if file.is_file():
                file_age = current_time - file.stat().st_mtime
                if file_age > (max_age_days * 24 * 3600):
                    file.unlink()