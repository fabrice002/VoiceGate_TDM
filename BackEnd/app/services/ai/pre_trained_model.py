# app/services/ai/pre_trained_model.py

import torch
import logging
import os
import gc # Garbage Collector
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from transformers import pipeline, set_seed
from core.config import settings

logger = logging.getLogger(__name__)

class HuggingFaceModel:
    """
    Hugging Face LLM loader - Safe Mode
    """
    
    def __init__(self, model_name: str = None, token: str = None):
        self.model_name = model_name or settings.HF_MODEL_NAME
        self.token = token or settings.HF_TOKEN
        
        # Setup Cache
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.cache_dir)
        
        # Force CPU pour éviter les erreurs VRAM si GPU trop faible
        self.device = -1 
        self.is_loaded = False
        self.generator = None
        self.tokenizer = None
    
    def load_model(self) -> bool:
        if self.is_loaded:
            return True

        logger.info(f"Tentative de chargement du modèle : {self.model_name}")
        
        # Nettoyage préventif de la mémoire
        gc.collect()
        
        try:
            # 1. Essai Optimisé (avec accelerate si dispo)
            logger.info("Chargement mode optimisé...")
            self.generator = pipeline(
                task="text-generation",
                model=self.model_name,
                device=self.device,
                # low_cpu_mem_usage=True nécessite 'accelerate'
                # Si cela crashait avant, mettez False ici pour tester
                model_kwargs={"low_cpu_mem_usage": True} 
            )
            
        except Exception as e:
            logger.warning(f"Echec du chargement optimisé ({e}). Tentative mode standard...")
            try:
                # 2. Essai Standard (Plus lent mais plus stable sur certaines machines)
                self.generator = pipeline(
                    task="text-generation",
                    model=self.model_name,
                    device=self.device
                )
            except Exception as e2:
                logger.error(f"ECHEC CRITIQUE chargement modèle : {e2}")
                # On ne plante pas l'app, on reste en mode dégradé
                self.is_loaded = False
                return False

        # Configuration du Tokenizer
        if self.generator:
            self.tokenizer = self.generator.tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.is_loaded = True
            logger.info("Modèle chargé avec succès !")
            return True
            
        return False
    
    def generate_response(self, user_input: str, history: list = None, **kwargs) -> str:
        """Génère une réponse ou un message de secours si le modèle est absent"""
        
        # Si le modèle a crashé au chargement, on répond ceci au lieu de planter
        if not self.is_loaded:
            # Tentative de rechargement au cas où
            if not self.load_model():
                return "Je suis désolé, mes systèmes neuronaux sont indisponibles (Mémoire insuffisante)."
        
        try:
            # Prompt simple
            prompt = f"{user_input}"
            
            # Génération
            output = self.generator(
                prompt, 
                max_new_tokens=50, # Court pour être rapide
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extraction propre
            response_text = output[0]['generated_text'] if output else ""
            
            # Nettoyage basique (on garde ce qui est après le prompt)
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
                
            return response_text or "Je n'ai pas de réponse."

        except Exception as e:
            logger.error(f"Erreur génération : {e}")
            return "Une erreur technique m'empêche de répondre."

# Singleton
_hf_instance = None

def get_hf_model() -> HuggingFaceModel:
    global _hf_instance
    if _hf_instance is None:
        _hf_instance = HuggingFaceModel()
    return _hf_instance

hf_model = get_hf_model()