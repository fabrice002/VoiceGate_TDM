"""
Audio transcription service using Whisper with universal audio conversion
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import os
import time

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

        logger.info(f"TranscriptionService initialized.")

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

        if self.model is None:
            return self._fallback_transcription(
                audio_data, sample_rate, language, original_filename
            )

        start_time = time.time()
        wav_file_path = None

        try:
            # Prepare WAV
            wav_file_path = self._prepare_any_audio_for_whisper(
                audio_data, sample_rate, original_filename, keep_converted_file
            )

            if not os.path.exists(wav_file_path):
                raise FileNotFoundError(wav_file_path)

            logger.info(f"Transcribing {wav_file_path}")
            self._verify_file_for_whisper(wav_file_path)

            # Transcribe
            result = self.model.transcribe(
                wav_file_path,
                language=language,
                task=task,
                fp16=False,
                verbose=False
            )

            # Metadata
            duration = self._get_audio_duration(wav_file_path)
            text = result.get("text", "")
            result.update({
                "success": True,
                "processing_time": time.time() - start_time,
                "audio_duration": duration,
                "sample_rate": sample_rate,
                "language": language or result.get("language", "unknown"),
                "word_count": len(text.split()),
                "char_count": len(text),
                "is_mock": False,
                "original_format": self._detect_original_format(audio_data, original_filename),
                "converted_file": wav_file_path if keep_converted_file else None
            })

            self._compute_confidence(result)

            # Save transcription
            if keep_converted_file:
                self._save_transcription_to_file(result, original_filename)

            # ---- NEW: Auto-generate SRT ----
            if generate_srt:
                srt_path = self.generate_subtitles(Path(wav_file_path), language)
                result["srt_file"] = str(srt_path)

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": [],
                "language": language or "unknown",
                "is_mock": True
            }

        finally:
            if not keep_converted_file and wav_file_path and os.path.exists(wav_file_path):
                try:
                    os.unlink(wav_file_path)
                except:
                    pass

    
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
        try:
            # Convert bytes to WAV
            wav_path, temp_files = audio_converter.convert_bytes_to_wav(
                audio_bytes=audio_bytes,
                original_filename=original_filename,
                keep_temp=keep_converted_file
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Audio conversion failed: {str(e)}"
            }

        # Call the unified transcription pipeline
        try:
            result = self.transcribe(
                audio_data=str(wav_path),
                language=language,
                task=task,
                original_filename=original_filename,
                keep_converted_file=keep_converted_file
            )

            result["converted_wav_path"] = wav_path
            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}",
                "converted_wav_path": wav_path
            }



    
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
        except:
            import librosa
            librosa.load(file_path, sr=None)

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
        except:
            return os.path.getsize(path) / (2 * 16000)

    # ---------------------------------------------------------
    #            CONFIDENCE FROM LOGPROBS
    # ---------------------------------------------------------
    def _compute_confidence(self, result):
        if "segments" not in result:
            return

        confidences = []
        for seg in result["segments"]:
            lp = seg.get("avg_logprob")
            if lp is not None:
                c = min(1.0, max(0.0, np.exp(lp)))
                seg["confidence"] = c
                confidences.append(c)

        if confidences:
            result["confidence"] = sum(confidences) / len(confidences)

    # ---------------------------------------------------------
    #            ORIGINAL FORMAT DETECTION
    # ---------------------------------------------------------
    def _detect_original_format(self, audio_data, filename):
        if filename:
            return Path(filename).suffix.lower()
        if isinstance(audio_data, str):
            return Path(audio_data).suffix.lower()
        return "bytes/numpy"

    # ---------------------------------------------------------
    #               MOCK / FALLBACK
    # ---------------------------------------------------------
    def _fallback_transcription(self, *args):
        return {
            "success": False,
            "text": "",
            "segments": [],
            "is_mock": True
        }

    # ---------------------------------------------------------
    #               SAVE TRANSCRIPTION
    # ---------------------------------------------------------
    def _save_transcription_to_file(self, result, original_name):
        name = original_name or f"transcript_{int(time.time())}.txt"
        out = self.output_dir / (Path(name).stem + ".txt")

        with open(out, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))


# Global instance
transcription_service = TranscriptionService()
