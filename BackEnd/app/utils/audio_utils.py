import wave
import contextlib
from pathlib import Path

def get_audio_duration(filepath: Path) -> float:
    """Get audio duration in seconds"""
    try:
        with contextlib.closing(wave.open(str(filepath), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(filepath))
            return len(audio) / 1000.0
        except:
            return 0.0

def validate_audio_file(filepath: Path, max_size_mb: int = 10) -> bool:
    """Validate audio file"""
    if not filepath.exists():
        return False
    
    # Check file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False
    
    # Check if it's a valid audio file
    try:
        duration = get_audio_duration(filepath)
        return duration > 0
    except:
        return False