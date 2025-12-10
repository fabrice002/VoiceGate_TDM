# app/services/ai/transcription.py

"""
Audio transcription service using Whisper with universal audio conversion
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import os
import time
import uuid
import traceback

from services.audio.processor import AudioProcessor
from services.audio.universal_converter import audio_converter

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Unified, robust audio transcription service using Whisper with universal format support.
    """

    def __init__(self):
        self.model = None
        self.processor = AudioProcessor()
        self._load_model()

        # Output dirs
        self.output_dir = Path("data/transcriptions").absolute()
        self.subtitles_dir = Path("data/subtitles").absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring
        self._init_monitoring()

        logger.info(f"TranscriptionService initialized.")

    def _init_monitoring(self):
        """Initialize monitoring integration"""
        try:
            from api.routes.monitoring import monitor
            self.monitor = monitor
            logger.info("Monitoring system initialized")
        except ImportError:
            self.monitor = None
            logger.warning("Monitoring system not available")

    # ---------------------------------------------------------
    #                    MODEL LOADING
    # ---------------------------------------------------------
    def _load_model(self):
        """Load Whisper model with cache"""
        try:
            import whisper
            from core.config import settings

            model_name = settings.WHISPER_MODEL
            cache_dir = Path(settings.MODEL_CACHE_DIR).absolute() / "whisper"
            cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Loading Whisper model: {model_name}")

            self.model = whisper.load_model(
                model_name,
                download_root=str(cache_dir)
            )

            logger.info(f"Whisper model '{model_name}' loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    # ---------------------------------------------------------
    #               SUBTITLE GENERATION (SRT)
    # ---------------------------------------------------------
    def generate_subtitles(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> Path:
        """
        Generate SRT subtitle file using Whisper.
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded.")

        logger.info(f"Generating subtitles for: {audio_path}")

        srt_path = self.subtitles_dir / f"{audio_path.stem}.srt"

        try:
            options = {
                "word_timestamps": word_timestamps,
                "verbose": False
            }
            if language:
                options["language"] = language

            result = self.model.transcribe(str(audio_path), **options)

            # Write SRT file
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, segment in enumerate(result["segments"], start=1):
                    start_time = self._format_timestamp(segment["start"])
                    end_time = self._format_timestamp(segment["end"])
                    text = segment["text"].strip()

                    f.write(f"{idx}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            logger.info(f"SRT created: {srt_path}")
            return srt_path

        except Exception as e:
            logger.error(f"Subtitle generation failed: {str(e)}")
            raise

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds → SRT timestamp"""
        ms = int((seconds % 1) * 1000)
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    # ---------------------------------------------------------
    #                MAIN TRANSCRIPTION LOGIC
    # ---------------------------------------------------------
    def transcribe(self,
                  audio_data: Union[np.ndarray, str, bytes],
                  sample_rate: int = 16000,
                  language: str = None,
                  task: str = "transcribe",
                  original_filename: Optional[str] = None,
                  keep_converted_file: bool = True,
                  generate_srt: bool = False) -> Dict[str, Any]:
        """Transcribe audio in ANY format"""

        # Start monitoring
        request_id = str(uuid.uuid4())
        if self.monitor:
            self.monitor.start_request(request_id, "transcription")

        start_time = time.time()
        wav_file_path = None
        success = False

        try:
            # Check if model is loaded
            if self.model is None:
                logger.warning("Whisper model not loaded, using fallback")
                result = self._fallback_transcription(
                    audio_data, sample_rate, language, original_filename
                )
                success = result.get("success", False)
                
                # Record failed request
                if self.monitor:
                    self.monitor.end_request(request_id, success=success)
                return result

            # Prepare WAV
            wav_file_path = self._prepare_any_audio_for_whisper(
                audio_data, sample_rate, original_filename, keep_converted_file
            )

            if not os.path.exists(wav_file_path):
                raise FileNotFoundError(f"WAV file not created: {wav_file_path}")

            # Get audio metadata for monitoring
            audio_size = os.path.getsize(wav_file_path)
            logger.info(f"Transcribing {wav_file_path} ({audio_size/1024:.1f} KB)")

            # Verify file
            self._verify_file_for_whisper(wav_file_path)

            # Transcribe with timing
            transcribe_start = time.time()
            result = self.model.transcribe(
                str(wav_file_path),
                language=language,
                task=task,
                fp16=False,
                verbose=False
            )
            transcribe_time = time.time() - transcribe_start

            # Get audio duration
            duration = self._get_audio_duration(wav_file_path)
            
            # Extract text and compute metrics
            text = result.get("text", "").strip()
            word_count = len(text.split()) if text else 0
            char_count = len(text)
            
            # Compute confidence
            confidence = self._compute_confidence(result)
            
            # Build result
            result.update({
                "success": True,
                "processing_time": time.time() - start_time,
                "transcribe_time": transcribe_time,
                "audio_duration": duration,
                "sample_rate": sample_rate,
                "language": language or result.get("language", "unknown"),
                "text": text,
                "word_count": word_count,
                "char_count": char_count,
                "is_mock": False,
                "original_format": self._detect_original_format(audio_data, original_filename),
                "converted_file": wav_file_path if keep_converted_file else None,
                "confidence": confidence
            })

            # Save transcription
            if keep_converted_file:
                self._save_transcription_to_file(result, original_filename)

            # Generate SRT if requested
            if generate_srt:
                try:
                    srt_path = self.generate_subtitles(Path(wav_file_path), language)
                    result["srt_file"] = str(srt_path)
                except Exception as e:
                    logger.error(f"SRT generation failed: {e}")
                    result["srt_error"] = str(e)

            success = True
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Transcription failed: {error_msg}")
            logger.debug(traceback.format_exc())
            
            result = {
                "success": False,
                "error": error_msg,
                "text": "",
                "segments": [],
                "language": language or "unknown",
                "is_mock": True,
                "processing_time": time.time() - start_time
            }
            return result

        finally:
            # Record monitoring
            if self.monitor:
                self.monitor.end_request(
                    request_id, 
                    success=success,
                    metadata={
                        "duration": result.get("audio_duration", 0),
                        "word_count": result.get("word_count", 0),
                        "language": result.get("language", "unknown"),
                        "confidence": result.get("confidence", 0),
                        "processing_time": result.get("processing_time", 0)
                    }
                )
            
            # Cleanup temp file
            if not keep_converted_file and wav_file_path and os.path.exists(wav_file_path):
                try:
                    os.unlink(wav_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {wav_file_path}: {e}")

    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        original_filename: str = "audio",
        language: str = None,
        task: str = "transcribe",
        keep_converted_file: bool = False
    ):
        """
        Transcribe audio passed as raw bytes.
        """
        request_id = str(uuid.uuid4())
        if self.monitor:
            self.monitor.start_request(request_id, "transcription")

        start_time = time.time()
        success = False
        wav_path = None

        try:
            # Convert bytes to WAV
            wav_path, temp_files = audio_converter.convert_bytes_to_wav(
                audio_bytes=audio_bytes,
                original_filename=original_filename,
                keep_temp=keep_converted_file
            )

            # Call the unified transcription pipeline
            result = self.transcribe(
                audio_data=str(wav_path),
                language=language,
                task=task,
                original_filename=original_filename,
                keep_converted_file=keep_converted_file
            )

            result["converted_wav_path"] = wav_path
            success = result.get("success", False)
            
            return result

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            result = {
                "success": False,
                "error": error_msg,
                "converted_wav_path": wav_path
            }
            return result

        finally:
            if self.monitor:
                self.monitor.end_request(
                    request_id,
                    success=success,
                    metadata={
                        "original_filename": original_filename,
                        "audio_size": len(audio_bytes),
                        "processing_time": time.time() - start_time
                    }
                )

    
    # ---------------------------------------------------------
    #              ALTERNATIVE ARRAY TRANSCRIPTION
    # ---------------------------------------------------------
    def _transcribe_with_audio_array(self, wav_path, language, task):
        import whisper
        audio_array = whisper.load_audio(wav_path)
        return self.model.transcribe(
            audio_array,
            language=language,
            task=task,
            fp16=False,
            verbose=False
        )

    # ---------------------------------------------------------
    #               FILE VALIDATION
    # ---------------------------------------------------------
    def _verify_file_for_whisper(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")

        try:
            import whisper
            whisper.load_audio(file_path)
        except Exception as e:
            logger.warning(f"Whisper audio load failed, trying librosa: {e}")
            try:
                import librosa
                librosa.load(file_path, sr=None)
            except Exception as e2:
                raise ValueError(f"Audio file is not valid: {e2}")

    # ---------------------------------------------------------
    #          UNIVERSAL AUDIO PREPARATION → WAV
    # ---------------------------------------------------------
    def _prepare_any_audio_for_whisper(self, audio_data, sample_rate, original_filename, keep_file):
        if isinstance(audio_data, bytes):
            wav, _ = audio_converter.convert_bytes_to_wav(
                audio_bytes=audio_data,
                original_filename=original_filename,
                sample_rate=sample_rate,
                mono=True,
                keep_temp=keep_file
            )
            return str(Path(wav).absolute())

        if isinstance(audio_data, np.ndarray):
            return self._prepare_audio_array(audio_data, sample_rate, keep_file)

        if isinstance(audio_data, str):
            audio_data = str(Path(audio_data).absolute())

            if not os.path.exists(audio_data):
                raise FileNotFoundError(audio_data)

            if audio_data.lower().endswith(".wav"):
                return audio_data

            import uuid
            out = self.output_dir / f"{Path(audio_data).stem}_{uuid.uuid4().hex[:8]}.wav"

            return audio_converter.convert_to_wav(
                audio_data, out, sample_rate=sample_rate, mono=True, keep_temp=keep_file
            )

        raise ValueError(f"Unsupported audio type: {type(audio_data)}")

    # ---------------------------------------------------------
    #             NUMPY ARRAY → WAV
    # ---------------------------------------------------------
    def _prepare_audio_array(self, audio_array, sample_rate, keep_file):
        import soundfile as sf
        import uuid

        out = self.output_dir / f"array_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"

        normalized = self.processor.normalize_audio(audio_array, 0.95)
        sf.write(out, normalized, sample_rate)

        return str(out.absolute())

    # ---------------------------------------------------------
    #               AUDIO DURATION
    # ---------------------------------------------------------
    def _get_audio_duration(self, path):
        try:
            import librosa
            return librosa.get_duration(path=path)
        except Exception as e:
            logger.warning(f"Failed to get duration with librosa: {e}")
            # Estimate duration from file size (16-bit mono)
            try:
                file_size = os.path.getsize(path)
                # Assuming 16-bit = 2 bytes per sample, mono, 16000 Hz
                duration = file_size / (2 * 16000)
                return duration
            except:
                return 0

    # ---------------------------------------------------------
    #            CONFIDENCE FROM LOGPROBS
    # ---------------------------------------------------------
    def _compute_confidence(self, result):
        if "segments" not in result:
            return 0.0

        confidences = []
        for seg in result["segments"]:
            lp = seg.get("avg_logprob")
            if lp is not None:
                try:
                    c = min(1.0, max(0.0, np.exp(lp)))
                    seg["confidence"] = c
                    confidences.append(c)
                except:
                    seg["confidence"] = 0.0
                    confidences.append(0.0)

        if confidences:
            confidence = sum(confidences) / len(confidences)
            result["confidence"] = confidence
            return confidence
        return 0.0

    # ---------------------------------------------------------
    #            ORIGINAL FORMAT DETECTION
    # ---------------------------------------------------------
    def _detect_original_format(self, audio_data, filename):
        if filename:
            ext = Path(filename).suffix.lower()
            if ext:
                return ext
        if isinstance(audio_data, str):
            ext = Path(audio_data).suffix.lower()
            if ext:
                return ext
        if isinstance(audio_data, bytes):
            return "bytes"
        if isinstance(audio_data, np.ndarray):
            return "numpy_array"
        return "unknown"

    # ---------------------------------------------------------
    #               MOCK / FALLBACK
    # ---------------------------------------------------------
    def _fallback_transcription(self, audio_data, sample_rate, language, filename):
        """Provide fallback when Whisper is not available"""
        logger.warning("Using fallback transcription")
        
        # Try simple speech recognition as fallback
        try:
            import speech_recognition as sr
            from io import BytesIO
            
            recognizer = sr.Recognizer()
            
            # Convert to audio data for speech_recognition
            if isinstance(audio_data, bytes):
                audio_file = BytesIO(audio_data)
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
            elif isinstance(audio_data, np.ndarray):
                # Convert numpy array to bytes
                import soundfile as sf
                audio_file = BytesIO()
                sf.write(audio_file, audio_data, sample_rate, format='WAV')
                audio_file.seek(0)
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
            else:
                # Try to load from file
                with sr.AudioFile(audio_data) as source:
                    audio = recognizer.record(source)
            
            # Recognize
            text = recognizer.recognize_google(audio, language=language or "en-US")
            
            return {
                "success": True,
                "text": text,
                "segments": [{"text": text, "start": 0, "end": 1}],
                "language": language or "en",
                "confidence": 0.7,
                "is_mock": True,
                "warning": "Using Google Speech Recognition as fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback recognition also failed: {e}")
            return {
                "success": False,
                "text": "",
                "segments": [],
                "error": str(e),
                "is_mock": True
            }

    # ---------------------------------------------------------
    #               SAVE TRANSCRIPTION
    # ---------------------------------------------------------
    def _save_transcription_to_file(self, result, original_name):
        name = original_name or f"transcript_{int(time.time())}.txt"
        out = self.output_dir / (Path(name).stem + ".txt")

        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(result.get("text", ""))
            logger.info(f"Transcription saved to {out}")
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")

    # ---------------------------------------------------------
    #               SERVICE INFO & HEALTH
    # ---------------------------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status"""
        status = {
            "is_loaded": self.model is not None,
            "model_name": "Whisper",
            "service": "transcription",
            "output_dir": str(self.output_dir),
            "subtitles_dir": str(self.subtitles_dir)
        }
        
        if self.model:
            status.update({
                "model_type": str(type(self.model)),
                "memory_usage": "loaded"
            })
        
        return status
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported formats"""
        return {
            "supported_formats": "All formats via conversion to WAV",
            "converter_status": "available" if audio_converter else "unavailable",
            "recommended_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
            "max_file_size": "10MB"
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            # Check if directories are writable
            dir_status = {}
            for dir_name, dir_path in [
                ("output_dir", self.output_dir),
                ("subtitles_dir", self.subtitles_dir),
                ("cache_dir", Path("data/pretrained_models/whisper"))
            ]:
                try:
                    test_file = dir_path / ".write_test"
                    test_file.touch(exist_ok=True)
                    test_file.unlink()
                    dir_status[dir_name] = "writable"
                except:
                    dir_status[dir_name] = "not_writable"
            
            return {
                "model_loaded": self.model is not None,
                "monitoring_enabled": self.monitor is not None,
                "directories": dir_status,
                "service": "running",
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "error": str(e),
                "service": "degraded",
                "timestamp": time.time()
            }


# Global instance
transcription_service = TranscriptionService()