# # app/services/ai/speaker_id.py

"""
Speaker Identification Service
Manages voice database and speaker recognition
"""

import os
import numpy as np
import torch
from typing import Optional, Dict
from core.config import settings
from services.ai.biometrics import VoiceBiometrics

class SpeakerIdentificationService:
    """Service for speaker identification and enrollment"""
    
    def __init__(self):
        self.db_folder = settings.VOICE_DB_FOLDER
        self.biometrics = VoiceBiometrics()
        self.threshold = settings.SPEAKER_THRESHOLD
        
        # Ensure database folder exists
        os.makedirs(self.db_folder, exist_ok=True)
    
    def enroll_user(self, username: str, audio_data: np.ndarray) -> bool:
        """Enroll a new user with voice sample"""
        try:
            # Extract embedding
            embedding = self.biometrics.extract_embedding(audio_data)
            if embedding is None:
                return False
            
            # Save embedding
            save_path = os.path.join(self.db_folder, f"{username}.npy")
            np.save(save_path, embedding.numpy())
            
            print(f"User {username} enrolled successfully")
            return True
            
        except Exception as e:
            print(f"Enrollment failed: {e}")
            return False
    
    def identify_speaker(self, audio_data: np.ndarray) -> Optional[Dict]:
        """Identify speaker from audio"""
        try:
            # Extract embedding
            unknown_embedding = self.biometrics.extract_embedding(audio_data)
            if unknown_embedding is None:
                return None
            
            # Load all known profiles
            profiles = [f for f in os.listdir(self.db_folder) if f.endswith(".npy")]
            if not profiles:
                return None
            
            # Find best match
            best_score = -1.0
            best_name = None
            cosine_sim = torch.nn.CosineSimilarity(dim=0)
            
            for profile in profiles:
                username = os.path.splitext(profile)[0]
                known_embedding = torch.from_numpy(
                    np.load(os.path.join(self.db_folder, profile))
                )
                
                score = cosine_sim(unknown_embedding, known_embedding).item()
                if score > best_score:
                    best_score = score
                    best_name = username
            
            # Check threshold
            if best_score > self.threshold:
                return {
                    "username": best_name,
                    "confidence": float(best_score),
                    "identified": True
                }
            
            return None
            
        except Exception as e:
            print(f"Identification failed: {e}")
            return None
    
    def get_registered_users(self) -> list:
        """Get list of registered users"""
        return [f.replace(".npy", "") for f in os.listdir(self.db_folder) 
                if f.endswith(".npy")]