# app/services/audio/processor.py

"""
Audio processing utilities for voice biometrics with transcription support
"""

import base64
import numpy as np
import tempfile
import os
import warnings
import librosa
import soundfile as sf
import wave
import io
from typing import Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processing utilities for voice biometrics and transcription
    
    Features:
    - Base64 encoding/decoding with format detection
    - Audio format conversion and file operations
    - Audio analysis, validation, and quality assessment
    - Silence detection, trimming, and normalization
    - Voice feature extraction (MFCC)
    - Transcription-specific audio preparation
    - Audio chunking for long recordings
    """
    
    # Audio format support
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
    PCM_FORMATS = ['.wav', '.pcm']
    
    # Transcription defaults
    TRANSCRIPTION_SAMPLE_RATE = 16000  # Whisper standard sample rate
    MAX_TRANSCRIPTION_CHUNK_DURATION = 30.0  # 30 seconds per chunk
    
    @staticmethod
    def decode_base64_audio(base64_string: str, 
                           return_bytes: bool = True) -> Tuple[bytes, Optional[str]]:
        """
        Decode base64 audio with padding correction and format detection
        
        Args:
            base64_string: Base64 encoded audio string (with or without data URL)
            return_bytes: If True, return bytes; if False, return numpy array
            
        Returns:
            Tuple[bytes/ndarray, str]: Audio data and detected format
            
        Raises:
            ValueError: If base64 decoding fails
        """
        try:
            # Clean the string - remove data URL prefix if present
            if base64_string.startswith('data:audio/'):
                # Extract format from data URL
                format_hint = base64_string.split(';')[0].split('/')[-1]
                base64_string = base64_string.split(',', 1)[1]
            elif base64_string.startswith('data:'):
                format_hint = None
                base64_string = base64_string.split(',', 1)[1]
            else:
                format_hint = None
            
            # Remove whitespace and newlines
            base64_string = base64_string.strip().replace('\n', '').replace('\r', '')
            
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            
            # Decode
            audio_bytes = base64.b64decode(base64_string)
            
            if len(audio_bytes) == 0:
                raise ValueError("Decoded audio is empty")
            
            # Try to detect format from magic bytes
            detected_format = AudioProcessor._detect_audio_format(audio_bytes)
            format_to_use = format_hint or detected_format or 'unknown'
            
            logger.debug(f"Decoded {len(audio_bytes)} bytes, format: {format_to_use}")
            
            if return_bytes:
                return audio_bytes, format_to_use
            else:
                # Convert to numpy array
                audio_array = AudioProcessor.bytes_to_audio_array(audio_bytes, format_hint=format_to_use)
                return audio_array, format_to_use
            
        except base64.binascii.Error as e:
            logger.error(f"Base64 decoding error: {e}")
            raise ValueError(f"Invalid base64 data: {str(e)}")
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {e}")
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    @staticmethod
    def decode_base64_simple(base64_string: str) -> bytes:
        """
        Simple base64 decoding for backward compatibility
        
        Args:
            base64_string: Base64 encoded audio string
            
        Returns:
            bytes: Decoded audio bytes
        """
        try:
            # Clean the string - remove data URL prefix if present
            if base64_string.startswith('data:audio/'):
                # Extract base64 part after comma
                base64_string = base64_string.split(',', 1)[1]
            elif base64_string.startswith('data:'):
                base64_string = base64_string.split(',', 1)[1]
            
            # Remove whitespace and newlines
            base64_string = base64_string.strip().replace('\n', '').replace('\r', '')
            
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            
            # Decode
            audio_bytes = base64.b64decode(base64_string)
            
            if len(audio_bytes) == 0:
                raise ValueError("Decoded audio is empty")
            
            logger.debug(f"Decoded {len(audio_bytes)} bytes from base64")
            return audio_bytes
            
        except base64.binascii.Error as e:
            logger.error(f"Base64 decoding error: {e}")
            raise ValueError(f"Invalid base64 data: {str(e)}")
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {e}")
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    @staticmethod
    def _detect_audio_format(audio_bytes: bytes) -> Optional[str]:
        """Detect audio format from magic bytes"""
        if len(audio_bytes) < 4:
            return None
        
        magic = audio_bytes[:4]
        
        # WAV: "RIFF"
        if magic.startswith(b'RIFF'):
            return 'wav'
        # MP3: ID3 or FF FB/FA/F3/F2
        elif magic.startswith(b'ID3') or (magic[0] == 0xFF and magic[1] & 0xE0 == 0xE0):
            return 'mp3'
        # OGG: "OggS"
        elif magic.startswith(b'OggS'):
            return 'ogg'
        # FLAC: "fLaC"
        elif magic.startswith(b'fLaC'):
            return 'flac'
        # MP4/M4A: "ftyp"
        elif magic.startswith(b'ftyp'):
            return 'm4a'
        else:
            # Try to decode as WAV without header (raw PCM)
            try:
                # Check if it might be raw PCM by trying to read as 16-bit samples
                if len(audio_bytes) % 2 == 0:
                    np.frombuffer(audio_bytes, dtype=np.int16)
                    return 'pcm'
            except:
                pass
        
        return None
    
    @staticmethod
    def encode_audio_to_base64(audio_data, 
                              format: str = 'wav',
                              sample_rate: int = settings.SAMPLE_RATE) -> str:
        """
        Encode audio to base64 string
        
        Args:
            audio_data: Audio data (bytes, ndarray, or file path)
            format: Output format ('wav', 'mp3', etc.)
            sample_rate: Sample rate for numpy arrays
            
        Returns:
            str: Base64 encoded string
            
        Raises:
            ValueError: If encoding fails
        """
        try:
            # Convert different input types to bytes
            if isinstance(audio_data, bytes):
                audio_bytes = audio_data
            elif isinstance(audio_data, np.ndarray):
                audio_bytes = AudioProcessor.array_to_bytes(
                    audio_data, 
                    format=format, 
                    sample_rate=sample_rate
                )
            elif isinstance(audio_data, (str, Path)):
                # Read from file
                with open(audio_data, 'rb') as f:
                    audio_bytes = f.read()
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
            
            encoded = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Add data URL prefix if requested
            if settings.ADD_DATA_URL_PREFIX:
                mime_type = AudioProcessor._format_to_mime(format)
                encoded = f"data:{mime_type};base64,{encoded}"
            
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding audio to base64: {e}")
            raise ValueError(f"Failed to encode audio: {str(e)}")
    
    @staticmethod
    def _format_to_mime(format: str) -> str:
        """Convert format string to MIME type"""
        mime_map = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4',
            'pcm': 'audio/pcm'
        }
        return mime_map.get(format.lower(), 'audio/wav')
    
    @staticmethod
    def save_audio_to_file(audio_data, 
                          filename: str, 
                          directory: str = None,
                          format: str = None,
                          sample_rate: int = settings.SAMPLE_RATE) -> str:
        """
        Save audio to file
        
        Args:
            audio_data: Audio data (bytes, ndarray, or file path)
            filename: Output filename (with or without extension)
            directory: Directory to save to (default: TEMP_DIR)
            format: Output format (auto-detected from filename if None)
            sample_rate: Sample rate for numpy arrays
            
        Returns:
            str: Full path to saved file
            
        Raises:
            ValueError: If saving fails
        """
        try:
            if directory is None:
                directory = settings.TEMP_DIR
            
            os.makedirs(directory, exist_ok=True)
            
            # Determine format from filename if not specified
            if format is None:
                if '.' in filename:
                    format = Path(filename).suffix[1:]  # Remove leading dot
                else:
                    format = 'wav'
                    filename = f"{filename}.wav"
            
            filepath = os.path.join(directory, filename)
            
            # Convert input to appropriate format
            if isinstance(audio_data, bytes):
                with open(filepath, 'wb') as f:
                    f.write(audio_data)
            elif isinstance(audio_data, np.ndarray):
                AudioProcessor.save_array_to_file(
                    audio_data, 
                    filepath, 
                    sample_rate=sample_rate,
                    format=format
                )
            elif isinstance(audio_data, (str, Path)):
                # Copy file
                import shutil
                shutil.copy2(audio_data, filepath)
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
            
            logger.debug(f"Audio saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
            raise ValueError(f"Failed to save audio: {str(e)}")
    
    @staticmethod
    def save_array_to_file(audio_array: np.ndarray,
                          filepath: str,
                          sample_rate: int = settings.SAMPLE_RATE,
                          format: str = 'wav'):
        """Save numpy array to audio file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            
            if format.lower() == 'wav':
                # Use wave module for WAV files
                audio_int16 = (audio_array * 32767).astype(np.int16)
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 2 bytes per sample
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())
            else:
                # Use soundfile for other formats
                sf.write(filepath, audio_array, sample_rate)
                
        except Exception as e:
            logger.error(f"Error saving array to file {filepath}: {e}")
            raise
    
    @staticmethod
    def bytes_to_audio_array(audio_bytes: bytes, 
                           sample_rate: int = settings.SAMPLE_RATE,
                           mono: bool = True,
                           format_hint: str = None) -> np.ndarray:
        """
        Convert audio bytes to numpy array with robust format handling
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Target sample rate
            mono: Whether to convert to mono
            format_hint: Hint about audio format
            
        Returns:
            np.ndarray: Audio samples as numpy array (float32, -1 to 1)
            
        Raises:
            ValueError: If audio conversion fails
        """
        try:
            # Try librosa first (handles many formats)
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            
            try:
                # Suppress warnings during loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Load with librosa - FIXED: proper parameter usage
                    audio, orig_sr = librosa.load(
                        temp_path,
                        sr=None,  # Get original sample rate
                        mono=mono,
                        duration=settings.MAX_AUDIO_DURATION
                    )
                    
                    # Resample if needed - FIXED: proper librosa.resample call
                    if orig_sr != sample_rate:
                        audio = librosa.resample(
                            y=audio,
                            orig_sr=orig_sr,
                            target_sr=sample_rate,
                            res_type='kaiser_best'
                        )
                    
                    # Normalize
                    audio = AudioProcessor.normalize_audio(audio)
                    
                    return audio.astype(np.float32)
                    
            except Exception as librosa_error:
                logger.warning(f"Librosa failed, trying fallback methods: {librosa_error}")
                
                # Fallback 1: Try soundfile
                try:
                    import soundfile as sf
                    audio, orig_sr = sf.read(io.BytesIO(audio_bytes))
                    
                    if mono and len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    if orig_sr != sample_rate:
                        audio = librosa.resample(
                            y=audio,
                            orig_sr=orig_sr,
                            target_sr=sample_rate
                        )
                    
                    return AudioProcessor.normalize_audio(audio.astype(np.float32))
                    
                except Exception as sf_error:
                    logger.warning(f"Soundfile failed: {sf_error}")
                    
                    # Fallback 2: Assume raw PCM (16-bit)
                    try:
                        # Try to detect if it's PCM
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio = audio_int16.astype(np.float32) / 32768.0
                        
                        # If stereo, convert to mono
                        if len(audio.shape) > 1 and mono:
                            audio = np.mean(audio, axis=1)
                        elif len(audio.shape) == 1 and not mono:
                            # Mono to stereo
                            audio = np.stack([audio, audio], axis=1)
                        
                        return AudioProcessor.normalize_audio(audio)
                        
                    except Exception as pcm_error:
                        logger.error(f"PCM fallback failed: {pcm_error}")
                        raise ValueError("Failed to decode audio with all methods")
                        
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Error converting audio bytes to array: {e}")
            raise ValueError(f"Failed to process audio: {str(e)}")
    
    @staticmethod
    def array_to_bytes(audio_array: np.ndarray,
                      format: str = 'wav',
                      sample_rate: int = settings.SAMPLE_RATE) -> bytes:
        """
        Convert numpy array to audio bytes
        
        Args:
            audio_array: Audio samples as numpy array
            format: Output format
            sample_rate: Sample rate
            
        Returns:
            bytes: Audio bytes
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
                temp_path = tmp.name
            
            # Save to file
            AudioProcessor.save_array_to_file(audio_array, temp_path, sample_rate, format)
            
            # Read back as bytes
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error converting array to bytes: {e}")
            raise
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray,
                      orig_sr: int,
                      target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio_data: Audio samples
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            np.ndarray: Resampled audio
        """
        try:
            if orig_sr == target_sr:
                return audio_data
            
            # FIXED: proper librosa.resample call
            return librosa.resample(
                y=audio_data,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data
    
    @staticmethod
    def prepare_for_transcription(audio_data: np.ndarray,
                                 target_sample_rate: int = 16000,
                                 normalize: bool = True) -> np.ndarray:
        """
        Prepare audio data for transcription
        
        Args:
            audio_data: Input audio samples
            target_sample_rate: Target sample rate (Whisper uses 16kHz)
            normalize: Whether to normalize audio
            
        Returns:
            np.ndarray: Prepared audio data
        """
        try:
            if len(audio_data) == 0:
                return audio_data
            
            current_rate = settings.SAMPLE_RATE
            
            # Resample if needed - FIXED: proper librosa.resample call
            if current_rate != target_sample_rate:
                audio_data = librosa.resample(
                    y=audio_data,
                    orig_sr=current_rate,
                    target_sr=target_sample_rate,
                    res_type='kaiser_best'
                )
            
            # Normalize if requested
            if normalize:
                audio_data = AudioProcessor.normalize_audio(audio_data, target_max=0.9)
            
            # Remove silence for better transcription
            audio_data = AudioProcessor.trim_silence(audio_data, top_db=25)
            
            # Ensure we have some audio left
            if len(audio_data) < target_sample_rate * 0.5:  # Less than 0.5 seconds
                logger.warning("Audio too short after trimming for transcription")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error preparing audio for transcription: {e}")
            return audio_data
    
    @staticmethod
    def split_long_audio(audio_data: np.ndarray,
                        sample_rate: int = 16000,
                        max_duration: float = 30.0) -> List[np.ndarray]:
        """
        Split long audio into chunks for better transcription
        
        Args:
            audio_data: Audio samples
            sample_rate: Audio sample rate
            max_duration: Maximum duration per chunk in seconds
            
        Returns:
            List[np.ndarray]: List of audio chunks
        """
        try:
            total_samples = len(audio_data)
            chunk_samples = int(max_duration * sample_rate)
            
            if total_samples <= chunk_samples:
                return [audio_data]
            
            chunks = []
            for start in range(0, total_samples, chunk_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = audio_data[start:end]
                
                # Don't include very short chunks (less than 1 second)
                if len(chunk) > sample_rate:
                    chunks.append(chunk)
            
            logger.info(f"Split {total_samples/sample_rate:.1f}s audio into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [audio_data]
    
    @staticmethod
    def analyze_for_transcription(audio_data: np.ndarray,
                                 sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze audio specifically for transcription quality
        
        Args:
            audio_data: Audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Dict[str, Any]: Transcription analysis results
        """
        # Get basic analysis
        analysis = AudioProcessor.analyze_audio(audio_data, sample_rate)
        
        if "error" in analysis:
            return analysis
        
        duration = analysis.get("duration", 0)
        snr = analysis.get("snr_db", 0)
        is_silent = analysis.get("is_silent", True)
        silence_percentage = analysis.get("silence_percentage", 100)
        
        # Transcription quality score (0-1)
        if is_silent or duration < 0.5:
            quality_score = 0.0
        elif snr > 20 and silence_percentage < 20:
            quality_score = 0.9
        elif snr > 15 and silence_percentage < 30:
            quality_score = 0.8
        elif snr > 10 and silence_percentage < 40:
            quality_score = 0.7
        elif snr > 5 and silence_percentage < 50:
            quality_score = 0.6
        else:
            quality_score = 0.4
        
        # Determine if suitable for transcription
        is_suitable = quality_score > 0.5 and duration > 0.5
        
        # Estimate word count (roughly 2.5 words per second of speech)
        estimated_words = 0
        if duration > 0:
            speech_duration = duration * (1 - silence_percentage / 100)
            estimated_words = int(speech_duration * 2.5)
        
        # Recommended action
        if quality_score < 0.4:
            action = "improve_audio_quality"
        elif quality_score < 0.7:
            action = "transcribe_with_caution"
        else:
            action = "transcribe"
        
        analysis.update({
            "transcription_quality": quality_score,
            "is_suitable_for_transcription": is_suitable,
            "estimated_word_count": estimated_words,
            "recommended_action": action,
            "speech_duration": duration * (1 - silence_percentage / 100),
            "needs_chunking": duration > 30.0
        })
        
        return analysis
    
    @staticmethod
    def is_silent(audio_data: np.ndarray, 
                  threshold: float = settings.SILENCE_THRESHOLD,
                  min_silence_duration: float = 0.1) -> Tuple[bool, float]:
        """
        Check if audio is silent
        
        Args:
            audio_data: Audio samples
            threshold: RMS threshold for silence
            min_silence_duration: Minimum duration to consider as silence
            
        Returns:
            Tuple[bool, float]: (is_silent, silence_percentage)
        """
        if len(audio_data) == 0:
            return True, 100.0
        
        try:
            # Calculate RMS in chunks
            chunk_size = int(settings.SAMPLE_RATE * 0.01)  # 10ms chunks
            chunks = [audio_data[i:i+chunk_size] 
                     for i in range(0, len(audio_data), chunk_size)]
            
            silent_chunks = 0
            for chunk in chunks:
                if len(chunk) > 0:
                    rms = np.sqrt(np.mean(chunk ** 2))
                    if rms < threshold:
                        silent_chunks += 1
            
            silence_percentage = (silent_chunks / len(chunks)) * 100 if chunks else 0
            
            # Check if enough continuous silence
            is_silent = silence_percentage > 90  # More than 90% silent
            
            return is_silent, silence_percentage
            
        except Exception as e:
            logger.error(f"Error checking if audio is silent: {e}")
            return True, 100.0
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, 
                       target_max: float = 0.95) -> np.ndarray:
        """
        Normalize audio to target range with headroom
        
        Args:
            audio_data: Audio samples
            target_max: Maximum amplitude after normalization (0-1)
            
        Returns:
            np.ndarray: Normalized audio
        """
        if len(audio_data) == 0:
            return audio_data
        
        try:
            current_max = np.max(np.abs(audio_data))
            
            if current_max > 0:
                # Apply normalization with headroom
                scale = target_max / current_max
                normalized = audio_data * scale
                
                # Soft clipping to prevent hard clipping
                if np.any(np.abs(normalized) > 0.99):
                    normalized = np.tanh(normalized) * 0.99
                
                return normalized.astype(np.float32)
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio_data
    
    @staticmethod
    def trim_silence(audio_data: np.ndarray, 
                    top_db: int = 20,
                    frame_length: int = 2048,
                    hop_length: int = 512,
                    **kwargs) -> np.ndarray:
        """
        Trim leading and trailing silence with enhanced detection
        
        Args:
            audio_data: Audio samples
            top_db: Threshold in decibels below reference
            frame_length: FFT window size
            hop_length: Number of samples between successive frames
            **kwargs: Additional parameters for librosa.effects.trim
            
        Returns:
            np.ndarray: Trimmed audio
        """
        if len(audio_data) == 0:
            return audio_data
        
        try:
            # Use librosa's trim function - FIXED: proper function call
            trimmed, (start, end) = librosa.effects.trim(
                y=audio_data,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length,
                **kwargs
            )
            
            logger.debug(f"Trimmed {len(audio_data)} to {len(trimmed)} samples "
                        f"(kept {len(trimmed)/len(audio_data)*100:.1f}%)")
            
            return trimmed
            
        except Exception as e:
            logger.warning(f"Librosa trim failed, using fallback: {e}")
            
            # Fallback: energy-based trim
            energy = np.abs(audio_data)
            threshold = np.percentile(energy, 10)  # 10th percentile as threshold
            
            # Find first non-silent sample with hysteresis
            start = 0
            consecutive_good = 0
            for i in range(len(energy)):
                if energy[i] > threshold * 2:  # Higher threshold for start
                    consecutive_good += 1
                    if consecutive_good >= 100:  # Require 100 consecutive good samples
                        start = max(0, i - 500)  # Keep some buffer
                        break
                else:
                    consecutive_good = 0
            
            # Find last non-silent sample
            end = len(energy)
            consecutive_good = 0
            for i in range(len(energy) - 1, -1, -1):
                if energy[i] > threshold * 1.5:
                    consecutive_good += 1
                    if consecutive_good >= 100:
                        end = min(len(energy), i + 500)
                        break
                else:
                    consecutive_good = 0
            
            return audio_data[start:end] if end > start else audio_data
    
    @staticmethod
    def calculate_snr(audio_data: np.ndarray, 
                     noise_floor_percentile: int = 5) -> float:
        """
        Calculate signal-to-noise ratio
        
        Args:
            audio_data: Audio samples
            noise_floor_percentile: Percentile to estimate noise floor
            
        Returns:
            float: SNR in decibels
        """
        if len(audio_data) < 100:
            return 0.0
        
        try:
            # Estimate noise from quiet parts
            energy = audio_data ** 2
            
            # Use percentile for noise floor estimation
            noise_power = np.percentile(energy, noise_floor_percentile)
            
            # Signal power (excluding quietest parts)
            signal_mask = energy > noise_power * 10
            if np.any(signal_mask):
                signal_power = np.mean(energy[signal_mask])
            else:
                signal_power = np.mean(energy)
            
            # Avoid division by zero
            if noise_power < 1e-10:
                noise_power = 1e-10
            
            snr_db = 10 * np.log10(signal_power / noise_power)
            
            return snr_db
            
        except Exception as e:
            logger.error(f"Error calculating SNR: {e}")
            return 0.0
    
    @staticmethod
    def analyze_audio(audio_data: np.ndarray, 
                     sample_rate: int = settings.SAMPLE_RATE,
                     detailed: bool = False) -> dict:
        """
        Comprehensive audio analysis
        
        Args:
            audio_data: Audio samples
            sample_rate: Audio sample rate
            detailed: Whether to include detailed features
            
        Returns:
            dict: Audio analysis results
        """
        if len(audio_data) == 0:
            return {"error": "Empty audio data"}
        
        try:
            # Basic properties
            duration = len(audio_data) / sample_rate
            energy = np.mean(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zero_crossing_rate = zero_crossings / len(audio_data)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            ).mean()
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate
            ).mean()
            
            # SNR
            snr_db = AudioProcessor.calculate_snr(audio_data)
            
            # Silence analysis
            is_silent, silence_percentage = AudioProcessor.is_silent(audio_data)
            
            # Create result dictionary
            result = {
                "duration": float(duration),
                "sample_count": len(audio_data),
                "sample_rate": sample_rate,
                "energy": float(energy),
                "rms": float(rms),
                "zero_crossing_rate": float(zero_crossing_rate),
                "snr_db": float(snr_db),
                "is_silent": is_silent,
                "silence_percentage": float(silence_percentage),
                "max_amplitude": float(np.max(np.abs(audio_data))),
                "mean_amplitude": float(np.mean(np.abs(audio_data))),
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
            }
            
            if detailed:
                # Add more detailed features
                result.update({
                    "crest_factor": float(np.max(np.abs(audio_data)) / rms if rms > 0 else 0),
                    "dynamic_range": float(20 * np.log10(np.max(np.abs(audio_data)) / 
                                                       np.percentile(np.abs(audio_data), 95) + 1e-10)),
                    "harmonic_ratio": float(AudioProcessor._calculate_harmonic_ratio(audio_data, sample_rate)),
                    "pitch": float(AudioProcessor._estimate_pitch(audio_data, sample_rate)),
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _calculate_harmonic_ratio(audio_data: np.ndarray, 
                                 sample_rate: int) -> float:
        """Calculate harmonic ratio (simplified)"""
        try:
            # Use librosa's harmonic-percussive separation
            harmonic, _ = librosa.effects.hpss(audio_data)
            if len(harmonic) > 0 and np.sum(np.abs(audio_data)) > 0:
                return np.sum(np.abs(harmonic)) / np.sum(np.abs(audio_data))
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def _estimate_pitch(audio_data: np.ndarray, sample_rate: int) -> float:
        """Estimate fundamental pitch (simplified)"""
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=sample_rate,
                fmin=50,
                fmax=500
            )
            if len(pitches) > 0:
                # Get pitch with highest magnitude
                max_index = np.unravel_index(
                    np.argmax(magnitudes), 
                    magnitudes.shape
                )
                return pitches[max_index]
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def validate_audio_duration(audio_data: np.ndarray, 
                              sample_rate: int = settings.SAMPLE_RATE) -> Tuple[bool, str, float]:
        """
        Validate audio duration
        
        Args:
            audio_data: Audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Tuple[bool, str, float]: (is_valid, message, duration)
        """
        duration = len(audio_data) / sample_rate
        
        if duration < settings.MIN_AUDIO_DURATION:
            return False, f"Audio too short ({duration:.2f}s < {settings.MIN_AUDIO_DURATION}s)", duration
        
        if duration > settings.MAX_AUDIO_DURATION:
            return False, f"Audio too long ({duration:.2f}s > {settings.MAX_AUDIO_DURATION}s)", duration
        
        return True, f"Valid duration: {duration:.2f}s", duration
    
    @staticmethod
    def validate_audio_quality(audio_data: np.ndarray,
                             sample_rate: int = settings.SAMPLE_RATE,
                             min_snr: float = 10.0,
                             max_silence: float = 80.0) -> Tuple[bool, str, dict]:
        """
        Validate audio quality
        
        Args:
            audio_data: Audio samples
            sample_rate: Audio sample rate
            min_snr: Minimum acceptable SNR in dB
            max_silence: Maximum acceptable silence percentage
            
        Returns:
            Tuple[bool, str, dict]: (is_valid, message, quality_metrics)
        """
        analysis = AudioProcessor.analyze_audio(audio_data, sample_rate)
        
        if "error" in analysis:
            return False, f"Analysis failed: {analysis['error']}", analysis
        
        issues = []
        
        # Check SNR
        if analysis["snr_db"] < min_snr:
            issues.append(f"Low SNR ({analysis['snr_db']:.1f}dB < {min_snr}dB)")
        
        # Check silence
        if analysis["silence_percentage"] > max_silence:
            issues.append(f"Too silent ({analysis['silence_percentage']:.1f}% > {max_silence}%)")
        
        # Check amplitude
        if analysis["max_amplitude"] < 0.01:
            issues.append(f"Very low amplitude ({analysis['max_amplitude']:.3f})")
        
        if issues:
            return False, "; ".join(issues), analysis
        else:
            return True, "Good quality", analysis
    
    @staticmethod
    def extract_voice_features(audio_data: np.ndarray,
                             sample_rate: int = settings.SAMPLE_RATE,
                             n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features for voice recognition
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            np.ndarray: MFCC features
        """
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            # Add delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Stack features
            features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            
            # Normalize features
            features = (features - np.mean(features, axis=1, keepdims=True)) / \
                      (np.std(features, axis=1, keepdims=True) + 1e-10)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            raise
    
    @staticmethod
    def preprocess_for_verification(audio_data: np.ndarray,
                                  sample_rate: int = settings.SAMPLE_RATE) -> np.ndarray:
        """
        Preprocess audio for voice verification
        
        Args:
            audio_data: Raw audio samples
            sample_rate: Sample rate
            
        Returns:
            np.ndarray: Preprocessed audio
        """
        # 1. Normalize
        audio_norm = AudioProcessor.normalize_audio(audio_data)
        
        # 2. Trim silence
        audio_trimmed = AudioProcessor.trim_silence(audio_norm, top_db=25)
        
        # 3. Ensure minimum length
        min_samples = int(settings.MIN_AUDIO_DURATION * sample_rate)
        if len(audio_trimmed) < min_samples:
            # Pad with zeros if too short
            padding = min_samples - len(audio_trimmed)
            audio_trimmed = np.pad(audio_trimmed, (0, padding), mode='constant')
        
        # 4. Ensure maximum length
        max_samples = int(settings.MAX_AUDIO_DURATION * sample_rate)
        if len(audio_trimmed) > max_samples:
            # Truncate if too long
            audio_trimmed = audio_trimmed[:max_samples]
        
        return audio_trimmed
    
    @staticmethod
    def get_audio_info(filepath: str) -> dict:
        """
        Get information about audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            dict: Audio file information
        """
        try:
            import soundfile as sf
            
            info = sf.info(filepath)
            
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "subtype": info.subtype,
                "frames": info.frames,
                "file_size": os.path.getsize(filepath)
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info for {filepath}: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def convert_format(input_path: str,
                      output_path: str,
                      output_format: str = 'wav',
                      sample_rate: int = settings.SAMPLE_RATE,
                      mono: bool = True) -> str:
        """
        Convert audio file format
        
        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Output format
            sample_rate: Target sample rate
            mono: Convert to mono
            
        Returns:
            str: Path to converted file
        """
        try:
            # Load audio
            audio, orig_sr = librosa.load(input_path, sr=None, mono=mono)
            
            # Resample if needed
            if orig_sr != sample_rate:
                audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=sample_rate)
            
            # Save to new format
            AudioProcessor.save_array_to_file(audio, output_path, sample_rate, output_format)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            raise
    
    @staticmethod
    def process_for_transcription(audio_data: np.ndarray,
                                 original_sample_rate: int = settings.SAMPLE_RATE,
                                 target_sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Complete processing pipeline for transcription
        
        Args:
            audio_data: Raw audio samples
            original_sample_rate: Original sample rate of audio
            target_sample_rate: Target sample rate for transcription
            
        Returns:
            Dict[str, Any]: Processing results including prepared audio and analysis
        """
        try:
            # Analyze original audio
            original_analysis = AudioProcessor.analyze_audio(audio_data, original_sample_rate)
            
            # Prepare for transcription
            prepared_audio = AudioProcessor.prepare_for_transcription(
                audio_data, 
                target_sample_rate=target_sample_rate,
                normalize=True
            )
            
            # Analyze prepared audio
            prepared_analysis = AudioProcessor.analyze_for_transcription(prepared_audio, target_sample_rate)
            
            # Split into chunks if needed
            chunks = []
            if prepared_analysis.get("needs_chunking", False):
                chunks = AudioProcessor.split_long_audio(
                    prepared_audio,
                    sample_rate=target_sample_rate,
                    max_duration=AudioProcessor.MAX_TRANSCRIPTION_CHUNK_DURATION
                )
            
            return {
                "success": True,
                "original_analysis": original_analysis,
                "prepared_analysis": prepared_analysis,
                "prepared_audio": prepared_audio,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "total_duration": prepared_analysis.get("duration", 0),
                "speech_duration": prepared_analysis.get("speech_duration", 0),
                "estimated_words": prepared_analysis.get("estimated_word_count", 0),
                "transcription_quality": prepared_analysis.get("transcription_quality", 0),
                "is_suitable": prepared_analysis.get("is_suitable_for_transcription", False)
            }
            
        except Exception as e:
            logger.error(f"Error processing audio for transcription: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_analysis": {},
                "prepared_analysis": {},
                "prepared_audio": audio_data,
                "chunks": [],
                "chunk_count": 0
            }


# Global instance for convenience
audio_processor = AudioProcessor()