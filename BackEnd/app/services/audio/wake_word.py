# app/services/audio/wake_word.py
"""
Wake word detection service
"""

import logging
import numpy as np
import speech_recognition as sr
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WakeWordStatus(Enum):
    """Wake word detection status"""
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    ERROR = "error"


@dataclass
class WakeWordResult:
    """Wake word detection result"""
    status: WakeWordStatus
    detected_word: Optional[str] = None
    confidence: float = 0.0
    audio_data: Optional[np.ndarray] = None
    timestamp: float = 0.0


class WakeWordDetector:
    """Wake word detector using multiple methods"""
    
    def __init__(self, wake_words: list = None, method: str = "keyword"):
        """
        Initialize wake word detector
        
        Args:
            wake_words: List of wake words to detect
            method: Detection method ('keyword' or 'energy')
        """
        self.wake_words = wake_words or ["okay", "ok", "hey", "hello", "voicegate", "assistant"]
        self.method = method
        self.recognizer = sr.Recognizer()
        self.energy_threshold = 300  # Adjust based on environment
        self.pause_threshold = 0.8   # Seconds of silence to consider end of speech
        self.dynamic_energy_threshold = True
        
    def detect_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> WakeWordResult:
        """
        Detect wake word from audio data
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            WakeWordResult object
        """
        try:
            if self.method == "keyword":
                return self._detect_with_keywords(audio_data, sample_rate)
            elif self.method == "energy":
                return self._detect_with_energy(audio_data, sample_rate)
            else:
                raise ValueError(f"Unknown detection method: {self.method}")
                
        except Exception as e:
            logger.error(f"Error detecting wake word: {e}")
            return WakeWordResult(status=WakeWordStatus.ERROR)
    
    def _detect_with_keywords(self, audio_data: np.ndarray, sample_rate: int) -> WakeWordResult:
        """Detect using keyword matching"""
        try:
            # Convert numpy array to AudioData for speech recognition
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            audio_sr = sr.AudioData(audio_bytes, sample_rate, 2)
            
            # Perform speech recognition
            text = self.recognizer.recognize_google(audio_sr, language="en-US").lower()
            
            # Check for wake words
            for wake_word in self.wake_words:
                if wake_word in text:
                    logger.info(f"Wake word '{wake_word}' detected")
                    return WakeWordResult(
                        status=WakeWordStatus.DETECTED,
                        detected_word=wake_word,
                        confidence=0.9,
                        audio_data=audio_data,
                        timestamp=np.random.random()  # In production, use actual timestamp
                    )
            
            return WakeWordResult(status=WakeWordStatus.NOT_DETECTED)
            
        except sr.UnknownValueError:
            # Speech not understood
            return WakeWordResult(status=WakeWordStatus.NOT_DETECTED)
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return WakeWordResult(status=WakeWordStatus.ERROR)
    
    def _detect_with_energy(self, audio_data: np.ndarray, sample_rate: int) -> WakeWordResult:
        """Detect using energy threshold"""
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        if energy > self.energy_threshold / 1000:  # Normalized threshold
            logger.info(f"Wake word detected by energy threshold: {energy}")
            return WakeWordResult(
                status=WakeWordStatus.DETECTED,
                detected_word="energy_based",
                confidence=min(1.0, energy * 2),
                audio_data=audio_data,
                timestamp=np.random.random()
            )
        
        return WakeWordResult(status=WakeWordStatus.NOT_DETECTED)
    
    def stream_detection(self, callback: Callable[[WakeWordResult], Any], 
                        sample_rate: int = 16000, chunk_size: int = 1024):
        """
        Stream audio from microphone and detect wake words
        
        Args:
            callback: Function to call when wake word is detected
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks to process
        """
        import sounddevice as sd
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            # Convert to mono if stereo
            if len(indata.shape) > 1:
                audio_chunk = np.mean(indata, axis=1)
            else:
                audio_chunk = indata.flatten()
            
            # Detect wake word
            result = self.detect_from_audio(audio_chunk, sample_rate)
            
            if result.status == WakeWordStatus.DETECTED:
                callback(result)
        
        # Start streaming
        with sd.InputStream(callback=audio_callback, 
                          channels=1,
                          samplerate=sample_rate,
                          blocksize=chunk_size):
            logger.info("Wake word detection streaming started...")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                logger.info("Wake word detection stopped")