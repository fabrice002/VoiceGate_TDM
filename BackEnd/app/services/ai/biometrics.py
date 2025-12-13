# app/services/ai/biometrics.py

"""
Voice biometrics and speech recognition service
"""

import torch
import numpy as np
import tempfile
import os
from typing import Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path

# =================================================================
# COMPATIBILITY PATCH FOR TORCHAUDIO (Fixes Windows/SpeechBrain crash)
# =================================================================
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    # Mock the missing function to prevent SpeechBrain from crashing
    # This function was deprecated and removed in recent torchaudio versions
    torchaudio.set_audio_backend = lambda backend: None
# =================================================================

# Additional safety cleanup for torchaudio backend internal attribute
try:
    if hasattr(torchaudio, '_backend'):
        delattr(torchaudio, '_backend')
except:
    pass

logger = logging.getLogger(__name__)


class VoiceBiometricsService:
    """Voice biometrics and speech recognition service"""
    
    def __init__(self):
        self.classifier = None
        self.whisper_model = None
        self.threshold = 0.7  # Default threshold for cosine similarity
        self._load_models()
    
    def _load_models(self):
        """Load ECAPA-TDNN and Whisper models"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            from app.core.config import settings
            
            # Load ECAPA-TDNN for voice biometrics
            model_dir = Path(settings.MODEL_CACHE_DIR) / "spkrec-ecapa-voxceleb"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Loading ECAPA-TDNN model...")
            
            # Note: run_opts={"device": "cpu"} forces CPU usage which is generally safer/easier for deployment
            # Change to "cuda" if GPU is available and configured
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir),
                run_opts={"device": "cpu"}
            )
            
            logger.info("ECAPA-TDNN model loaded successfully")
            
        except ImportError:
            logger.warning("SpeechBrain not installed. Using mock embeddings.")
            self.classifier = None
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model: {e}")
            logger.info("Using mock embeddings for testing due to load failure.")
            self.classifier = None
        
        # Whisper model will be loaded lazily (on first use) to save startup time
    
    def _load_whisper(self):
        """Load Whisper model on-demand"""
        if self.whisper_model is None:
            try:
                import whisper
                from app.core.config import settings
                
                logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
                
                self.whisper_model = whisper.load_model(settings.WHISPER_MODEL)
                
                logger.info("Whisper model loaded")
                
            except ImportError:
                logger.warning("Whisper not available, using mock transcription")
                self.whisper_model = None
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                logger.info("Using mock transcription for testing")
                self.whisper_model = None
    
    def extract_embedding(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract voice embedding from audio using ECAPA-TDNN
        
        Args:
            audio_data: Audio samples as numpy array (PCM)
            
        Returns:
            Optional[np.ndarray]: Voice embedding vector (192-dim) or None if failed
        """
        try:
            if len(audio_data) == 0:
                return None
            
            # Mock embedding if model failed to load
            if self.classifier is None:
                logger.debug("Using mock embedding")
                return self._generate_mock_embedding(audio_data)
            
            # Real embedding extraction
            # Ensure tensor is float32 and has batch dimension
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # encode_batch returns tensor of shape [batch, 1, 192]
                embedding = self.classifier.encode_batch(audio_tensor)
            
            # Flatten to 1D array
            embedding_np = embedding.squeeze().cpu().numpy()
            
            # Normalize embedding vector (critical for cosine similarity)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def transcribe_audio(self, audio_data: np.ndarray, 
                        sample_rate: int = 16000,
                        language: str = "fr") -> Dict[str, Any]:
        """
        Transcribe audio to text using OpenAI Whisper
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Audio sample rate (Whisper expects 16k)
            language: Language code for transcription hint
            
        Returns:
            Dict[str, Any]: Transcription results with text, segments, etc.
        """
        try:
            # Load Whisper if not already loaded
            self._load_whisper()
            
            if self.whisper_model is None:
                logger.debug("Using mock transcription")
                return self._generate_mock_transcription(audio_data, language)
            
            # Whisper requires a file path or direct audio array
            # We save to temporary WAV to ensure compatibility
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            
            try:
                # Convert float32 array to int16 PCM for standard WAV writing
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Save using scipy if available, fallback to soundfile
                try:
                    import scipy.io.wavfile as wavfile
                    wavfile.write(temp_path, sample_rate, audio_int16)
                except:
                    import soundfile as sf
                    sf.write(temp_path, audio_int16, sample_rate)
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    temp_path,
                    language=language,
                    task="transcribe",
                    fp16=False  # Disable FP16 to prevent errors on CPU-only machines
                )
                
                # Ensure a success flag and duration are present
                if 'duration' not in result:
                    result['duration'] = len(audio_data) / sample_rate
                result['success'] = True

                return result
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "text": "",
                "segments": [],
                "language": language,
                "duration": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two voice embeddings using cosine similarity
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            float: Similarity score between -1.0 and 1.0 (Higher is more similar)
        """
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            # Ensure vectors are normalized
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
            
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Clip to valid range to avoid floating point errors
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def _generate_mock_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Generate deterministic mock embedding for testing purposes only.
        Used when SpeechBrain fails to load.
        """
        energy = np.mean(np.abs(audio_data))
        zcr = np.sum(np.diff(np.sign(audio_data)) != 0) / max(len(audio_data), 1)
        
        # Deterministic seed based on audio properties
        np.random.seed(int(energy * 1000 + zcr * 100))
        embedding = np.random.randn(192).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _generate_mock_transcription(self, audio_data: np.ndarray, 
                                   language: str = "fr") -> Dict[str, Any]:
        """
        Generate mock transcription result for testing purposes only.
        """
        # Mock transcriptions
        mock_texts = {
            "fr": [
                "Bonjour, c'est un test de transcription.",
                "Je voudrais accéder à mon compte.",
                "Quelle est la météo aujourd'hui ?"
            ],
            "en": [
                "Hello, this is a transcription test.",
                "I would like to access my account.",
                "What is the weather today?"
            ]
        }
        
        energy = np.mean(np.abs(audio_data))
        duration = len(audio_data) / 16000
        
        # Pick text based on audio energy to simulate variety
        texts = mock_texts.get(language, mock_texts["en"])
        idx = int(energy * 1000) % len(texts)
        text = texts[idx]
        
        return {
            "text": text,
            "segments": [],
            "language": language,
            "duration": duration,
            "is_mock": True,
            "success": True
        }