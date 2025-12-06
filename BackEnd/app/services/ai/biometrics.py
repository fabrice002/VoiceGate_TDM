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

logger = logging.getLogger(__name__)

# Fix torchaudio backend issue
try:
    import torchaudio
    if hasattr(torchaudio, '_backend'):
        delattr(torchaudio, '_backend')
except:
    pass


class VoiceBiometricsService:
    """Voice biometrics and speech recognition service"""
    
    def __init__(self):
        self.classifier = None
        self.whisper_model = None
        self.threshold = 0.7  # Default threshold
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
            
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir),
                run_opts={"device": "cpu"}
            )
            
            logger.info("ECAPA-TDNN model loaded")
            
        except ImportError:
            logger.warning("SpeechBrain not available, using mock embeddings")
            self.classifier = None
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model: {e}")
            logger.info("Using mock embeddings for testing")
            self.classifier = None
        
        # Whisper model will be loaded lazily
    
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
        Extract voice embedding from audio
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Optional[np.ndarray]: Voice embedding vector or None if failed
        """
        try:
            if len(audio_data) == 0:
                return None
            
            # Mock embedding if model not loaded
            if self.classifier is None:
                logger.debug("Using mock embedding")
                return self._generate_mock_embedding(audio_data)
            
            # Real embedding extraction
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.classifier.encode_batch(audio_tensor)
            
            embedding_np = embedding.squeeze().cpu().numpy()
            
            # Normalize
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
        Transcribe audio to text using Whisper
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Audio sample rate
            language: Language code for transcription
            
        Returns:
            Dict[str, Any]: Transcription results with text, segments, etc.
        """
        try:
            # Load Whisper if not already loaded
            self._load_whisper()
            
            if self.whisper_model is None:
                logger.debug("Using mock transcription")
                return self._generate_mock_transcription(audio_data, language)
            
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            
            try:
                # Convert to 16-bit PCM for Whisper
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Save using scipy if available, otherwise use soundfile
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
                    fp16=False  # Disable FP16 for CPU compatibility
                )
                
                return result
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "text": f"[Transcription Error: {str(e)}]",
                "segments": [],
                "language": language
            }
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare embeddings using cosine similarity
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        try:
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
            
            similarity = np.dot(emb1_norm, emb2_norm)
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def _generate_mock_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Generate deterministic mock embedding for testing
        
        Args:
            audio_data: Audio samples
            
        Returns:
            np.ndarray: Mock embedding vector
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
        Generate mock transcription for testing
        
        Args:
            audio_data: Audio samples
            language: Language code
            
        Returns:
            Dict[str, Any]: Mock transcription results
        """
        # Mock transcriptions in different languages
        mock_texts = {
            "fr": [
                "Bonjour, comment allez-vous aujourd'hui?",
                "Je voudrais prendre un rendez-vous pour demain.",
                "Le système de reconnaissance vocale fonctionne très bien.",
                "Pouvez-vous m'aider avec cette tâche s'il vous plaît?",
                "La météo est agréable aujourd'hui, n'est-ce pas?"
            ],
            "en": [
                "Hello, how are you doing today?",
                "I would like to schedule an appointment for tomorrow.",
                "The voice recognition system is working very well.",
                "Can you help me with this task please?",
                "The weather is nice today, isn't it?"
            ],
            "es": [
                "Hola, ¿cómo estás hoy?",
                "Me gustaría programar una cita para mañana.",
                "El sistema de reconocimiento de voz funciona muy bien.",
                "¿Puedes ayudarme con esta tarea por favor?",
                "El clima está agradable hoy, ¿no es así?"
            ]
        }
        
        # Select based on audio energy for variety
        energy = np.mean(np.abs(audio_data))
        duration = len(audio_data) / 16000
        
        # Use audio properties to select text
        texts = mock_texts.get(language, mock_texts["en"])
        idx = int(energy * 1000) % len(texts)
        text = texts[idx]
        
        # Generate mock segments
        segments = []
        word_count = len(text.split())
        avg_word_duration = duration / max(word_count, 1)
        
        words = text.split()
        current_time = 0
        
        for word in words:
            word_duration = avg_word_duration * (0.8 + np.random.random() * 0.4)
            segments.append({
                "id": len(segments),
                "seek": int(current_time * 100),
                "start": current_time,
                "end": current_time + word_duration,
                "text": word,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.5,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.1
            })
            current_time += word_duration
        
        return {
            "text": text,
            "segments": segments,
            "language": language,
            "duration": duration,
            "is_mock": True
        }