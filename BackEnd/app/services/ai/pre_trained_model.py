# # app/services/ai/pre_trained_model.py
import torch
import logging
from pathlib import Path
import os
from transformers import pipeline, set_seed
from core.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceModel:
    """
    Hugging Face LLM loader - Fixed version
    Uses environment variables for caching to avoid parameter conflicts
    """
    
    def __init__(self, model_name: str = None, token: str = None):
        self.model_name = model_name or settings.HF_MODEL_NAME
        self.token = token or settings.HF_TOKEN
        
        # Set environment variables for Hugging Face cache
        cache_dir = Path(settings.MODEL_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        
        if self.token:
            os.environ["HF_TOKEN"] = self.token
        
        # Device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model components
        self.generator = None
        self.is_loaded = False
        
        logger.info(" Initializing model: %s", self.model_name)
        logger.info(" Device: %s", self.device)
        logger.info(" Cache directory: %s", cache_dir)
    
    def load_model(self) -> bool:
        """Load model using pipeline with environment-based caching"""
        try:
            logger.info("Loading model...")
            
            # Vérifier si le modèle est déjà dans le cache
            model_cache_dir = None
            if hasattr(settings, 'MODEL_CACHE_DIR') and settings.MODEL_CACHE_DIR:
                model_cache_dir = settings.MODEL_CACHE_DIR
                
            model_to_load = self.model_name
            
            if model_cache_dir:
                # Vérifier la présence de fichiers clés du modèle
                possible_paths = [
                    os.path.join(model_cache_dir, f"models--{self.model_name.replace('/', '--')}"),
                    os.path.join(model_cache_dir, self.model_name.replace("/", "--")),
                    os.path.join(model_cache_dir, self.model_name)
                ]
                
                model_found = False
                for model_path in possible_paths:
                    if os.path.exists(model_path):
                        # Vérifier la présence de fichiers essentiels
                        essential_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
                        has_essential_files = any(
                            os.path.exists(os.path.join(model_path, f)) 
                            for f in essential_files
                        )
                        
                        if has_essential_files:
                            logger.info(f"Model found in cache: {model_path}")
                            model_to_load = model_path
                            model_found = True
                            break
                
                if not model_found:
                    logger.info(f"Model not found in cache. Will download to: {model_cache_dir}")
                    # Configurer le cache pour le téléchargement
                    os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
                    os.environ['HF_HOME'] = model_cache_dir
            
            # Determine device for pipeline
            device = 0 if self.device == "cuda" else -1
            
            # Create pipeline
            self.generator = pipeline(
                task="text-generation",
                model=model_to_load,
                device=device,
                torch_dtype=torch.float32,
            )
            
            # Ensure pad token is set
            if hasattr(self.generator.tokenizer, 'pad_token') and self.generator.tokenizer.pad_token is None:
                self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            self.generator = None
            self.is_loaded = False
            return False
    
    def build_prompt(self, user_input: str, history: list | None = None) -> str:
        """Build prompt with conversation history"""
        history = history or []
        prompt_parts = []

        for user, assistant in history:
            prompt_parts.append(f"User: {user}")
            prompt_parts.append(f"Assistant: {assistant}")

        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)
    
    def generate_response(
        self,
        user_input: str,
        history: list | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        seed: int = None,
        **kwargs
    ) -> str:
        """Generate response - ensures no loading parameters are passed"""
        if not self.is_loaded or not self.generator:
            logger.warning("Model not loaded, using fallback response")
            return self._mock_response()
        
        try:
            # Build prompt
            prompt = self.build_prompt(user_input, history)
            
            # Set seed if provided
            if seed is not None:
                set_seed(seed)
            
            # Create clean generation arguments
            # Only include valid generation parameters
            generation_args = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
            }
            
            # Add pad_token_id
            if hasattr(self.generator.tokenizer, 'pad_token_id'):
                if self.generator.tokenizer.pad_token_id is None:
                    self.generator.tokenizer.pad_token_id = self.generator.tokenizer.eos_token_id
                generation_args["pad_token_id"] = self.generator.tokenizer.pad_token_id
            
            # Filter out any loading parameters from kwargs
            loading_params = {'cache_dir', 'token', 'use_auth_token', 'local_files_only', 
                             'force_download', 'resume_download', 'proxies'}
            
            for key, value in kwargs.items():
                if key not in loading_params:
                    generation_args[key] = value
                else:
                    logger.warning("Skipping loading parameter: %s", key)
            
            # Generate response
            with torch.no_grad():
                output = self.generator(prompt, **generation_args)
            
            # Extract and clean response
            if isinstance(output, list) and len(output) > 0:
                result = output[0]
                if isinstance(result, dict) and "generated_text" in result:
                    full_text = result["generated_text"]
                    response = full_text[len(prompt):].strip()
                else:
                    response = str(result)
            else:
                response = str(output)
            
            # Clean response
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            return response or self._mock_response()
            
        except Exception as e:
            logger.exception(" Generation error")
            return self._mock_response()
    
    def _mock_response(self) -> str:
        """Fallback response when generation fails"""
        return (
            "Je suis en mode restreint actuellement, "
            "mais je reste disponible "
        )
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "has_generator": self.generator is not None,
        }


# Global instance
_hf_instance = None

def get_hf_model() -> HuggingFaceModel:
    """Get singleton instance of HuggingFaceModel"""
    global _hf_instance
    if _hf_instance is None:
        _hf_instance = HuggingFaceModel()
    return _hf_instance

# For backward compatibility
hf_model = get_hf_model()


# Optional: Alternative class (keeping for compatibility)
class HuggingFaceModelAlternative(HuggingFaceModel):
    """Alias for compatibility with existing code"""
    pass