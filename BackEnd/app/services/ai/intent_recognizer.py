# app/services/ai/intent_recognizer.py
"""
Intent recognition and LLM integration service
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import re

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Intent recognition result"""
    intent: str
    confidence: float
    entities: Dict[str, str]
    slots: Dict[str, Any]
    raw_response: Optional[str] = None


class IntentRecognizer:
    """Intent recognition service with LLM fallback"""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize intent recognizer
        
        Args:
            use_llm: Whether to use LLM for intent recognition
        """
        self.use_llm = use_llm
        self.patterns = self._load_intent_patterns()
        self.llm_client = None
        
        if use_llm:
            self.llm_client = self._initialize_llm()
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent patterns from configuration"""
        return {
            "greeting": [
                "bonjour", "salut", "hello", "hi", "hey", "coucou",
                "bonsoir", "bon matin", "good morning", "good day"
            ],
            "farewell": [
                "au revoir", "bye", "goodbye", "à plus", "à bientôt",
                "ciao", "adieu", "see you", "take care"
            ],
            "question_time": [
                "quelle heure", "il est quelle heure", "time", "l'heure",
                "quelle heure est-il", "what time", "current time"
            ],
            "question_date": [
                "quelle date", "on est quel jour", "date d'aujourd'hui",
                "quel jour sommes-nous", "what date", "today's date"
            ],
            "weather": [
                "temps", "météo", "weather", "il fait quel temps",
                "prévisions météo", "pleut-il", "fait-il beau",
                "weather forecast", "is it raining"
            ],
            "help": [
                "aide", "help", "assistance", "peux-tu m'aider",
                "comment ça marche", "que peux-tu faire",
                "how does it work", "what can you do"
            ],
            "system_status": [
                "statut", "status", "ça marche", "fonctionne",
                "es-tu opérationnel", "tout va bien",
                "are you working", "system status"
            ],
            "joke": [
                "blague", "joke", "histoire drôle", "raconte une blague",
                "fais-moi rire", "tell me a joke", "make me laugh"
            ],
            "music": [
                "musique", "chanson", "music", "song", "jouer de la musique",
                "mets de la musique", "play music", "play a song"
            ],
            "volume": [
                "volume", "plus fort", "moins fort", "augmente le son",
                "baisse le son", "turn up", "turn down", "louder", "quieter"
            ],
            "thanks": [
                "merci", "thank you", "thanks", "grazie", "danke"
            ],
            "unknown": [
                # Default for unknown intents
            ]
        }
    
    def _initialize_llm(self):
        """Initialize LLM client if available"""
        try:
            from core.config import settings
            
            # Check if HuggingFace model is configured
            if not settings.HF_MODEL_NAME:
                logger.warning("HF_MODEL_NAME not configured in settings")
                self.use_llm = False
                return None
                
            logger.info(f"Loading LLM model: {settings.HF_MODEL_NAME}")
            
            # Try to use custom HuggingFaceModel class if available
            try:
                from services.ai.pre_trained_model import HuggingFaceModel
                
                if settings.HF_TOKEN:
                    hf_model = HuggingFaceModel(
                        model_name=settings.HF_MODEL_NAME, 
                        token=settings.HF_TOKEN
                    )
                    if hf_model.load_model():
                        logger.info(f"Successfully loaded HuggingFace model: {settings.HF_MODEL_NAME}")
                        self.use_llm = True
                        return hf_model
            except ImportError:
                logger.warning("Custom HuggingFaceModel class not available, trying transformers directly")
            except Exception as e:
                logger.error(f"Error loading custom HuggingFace model: {e}")
            
            # Fallback to transformers pipeline directly
            try:
                from transformers import pipeline, AutoTokenizer
                
                # Create pipeline with model name
                tokenizer = AutoTokenizer.from_pretrained(
                    settings.HF_MODEL_NAME,
                    token=settings.HF_TOKEN if settings.HF_TOKEN else None
                )
                
                llm_pipeline = pipeline(
                    "text-generation",
                    model=settings.HF_MODEL_NAME,
                    tokenizer=tokenizer,
                    max_length=settings.MAX_CONTEXT_LENGTH,
                    temperature=0.7,
                    device_map="auto",  # Automatically uses GPU if available
                    token=settings.HF_TOKEN if settings.HF_TOKEN else None
                )
                
                logger.info(f"Successfully loaded model via transformers: {settings.HF_MODEL_NAME}")
                self.use_llm = True
                return llm_pipeline
                
            except ImportError:
                logger.warning("Transformers library not available. LLM disabled.")
                self.use_llm = False
                return None
            except Exception as e:
                logger.error(f"Error creating transformers pipeline: {e}")
                self.use_llm = False
                return None
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.use_llm = False
            return None
    
    def recognize(self, text: str, language: str = "fr") -> IntentResult:
        """
        Recognize intent from text
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            IntentResult object
        """
        text_lower = text.lower().strip()
        
        # First try pattern matching
        intent, confidence, entities = self._pattern_match(text_lower, language)
        
        # If low confidence and LLM enabled, try LLM
        if confidence < 0.6 and self.use_llm and self.llm_client:
            llm_result = self._llm_intent_recognition(text_lower, language)
            if llm_result and llm_result.confidence > confidence:
                intent = llm_result.intent
                confidence = llm_result.confidence
                entities.update(llm_result.entities)
                raw_response = llm_result.raw_response
            else:
                raw_response = None
        else:
            raw_response = None
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            slots=self._extract_slots(text_lower, intent),
            raw_response=raw_response
        )
    
    def _pattern_match(self, text: str, language: str) -> Tuple[str, float, Dict]:
        """Match text against intent patterns"""
        best_intent = "unknown"
        best_confidence = 0.0
        entities = {}
        
        for intent, patterns in self.patterns.items():
            if intent == "unknown":
                continue
                
            for pattern in patterns:
                if pattern in text:
                    # Calculate confidence based on pattern length match
                    pattern_length = len(pattern)
                    text_length = len(text)
                    confidence = pattern_length / text_length if text_length > 0 else 0
                    
                    # Boost confidence for exact matches
                    if pattern == text.strip():
                        confidence = 1.0
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
                        
                        # Extract simple entities
                        if intent == "greeting":
                            name = self._extract_name(text)
                            if name:
                                entities["user_name"] = name
        
        # Ensure confidence is between 0 and 1
        best_confidence = min(1.0, max(0.0, best_confidence))
        
        return best_intent, best_confidence, entities
    
    def _llm_intent_recognition(self, text: str, language: str) -> Optional[IntentResult]:
        """Use LLM for intent recognition"""
        if not self.llm_client:
            return None
        
        try:
            # Create prompt for intent classification
            prompt = self._create_intent_prompt(text, language)
            
            # Generate response
            if hasattr(self.llm_client, 'generate_response'):
                # Custom HuggingFaceModel interface
                response = self.llm_client.generate_response(prompt)
            else:
                # Standard transformers pipeline
                result = self.llm_client(
                    prompt,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.3
                )
                response = result[0]["generated_text"]
            
            # Parse response
            return self._parse_llm_response(response, text)
            
        except Exception as e:
            logger.error(f"LLM intent recognition failed: {e}")
            return None
    
    def _create_intent_prompt(self, text: str, language: str) -> str:
        """Create prompt for LLM intent classification"""
        if language == "fr":
            return f"""Classifie l'intention de ce message: "{text}"

Intentions possibles: greeting, farewell, question_time, question_date, weather, help, system_status, joke, music, volume, thanks, unknown

Réponds en format JSON: {{"intent": "nom_intention", "confidence": 0.95, "entities": {{"entity_name": "value"}}}}
Raisonnement: [brief reasoning]
Réponse:"""
        else:
            return f"""Classify the intent of this message: "{text}"

Possible intents: greeting, farewell, question_time, question_date, weather, help, system_status, joke, music, volume, thanks, unknown

Respond in JSON format: {{"intent": "intent_name", "confidence": 0.95, "entities": {{"entity_name": "value"}}}}
Reasoning: [brief reasoning]
Response:"""
    
    def _parse_llm_response(self, response: str, original_text: str) -> Optional[IntentResult]:
        """Parse LLM response to extract intent"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
                
            json_str = json_match.group()
            data = json.loads(json_str)
            
            intent = data.get("intent", "unknown")
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
            entities = data.get("entities", {})
            
            # Validate intent
            if intent not in self.patterns:
                intent = "unknown"
                confidence = max(0.3, confidence * 0.7)  # Reduce confidence for unknown intents
            
            return IntentResult(
                intent=intent,
                confidence=confidence,
                entities=entities,
                slots={},
                raw_response=response
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from greeting"""
        # Simple name extraction patterns
        patterns = [
            r'je m\'appelle (\w+)',
            r'je suis (\w+)',
            r'my name is (\w+)',
            r'i am (\w+)',
            r'appelle moi (\w+)',
            r'call me (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        return None
    
    def _extract_slots(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract slots/parameters from text based on intent"""
        slots = {}
        
        if intent == "weather":
            # Extract location
            location_patterns = [
                r'à (\w+)',
                r'dans (\w+)',
                r'pour (\w+)',
                r'à ([A-Z][a-z]+(?: [A-Z][a-z]+)*)',  # Captures multi-word city names
                r'in (\w+)',
                r'for (\w+)'
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    slots["location"] = match.group(1)
                    break
        
        elif intent == "music":
            # Extract artist or song
            music_patterns = [
                r'de (\w+)',
                r'par (\w+)',
                r'chanson de (\w+)',
                r'song by (\w+)',
                r'music by (\w+)'
            ]
            
            for pattern in music_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    slots["artist"] = match.group(1)
                    break
        
        elif intent == "volume":
            # Extract volume level
            volume_patterns = [
                r'à (\d+)%',
                r'à (\d+) pourcent',
                r'to (\d+)%',
                r'to (\d+) percent'
            ]
            
            for pattern in volume_patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        slots["level"] = int(match.group(1))
                    except ValueError:
                        pass
                    break
        
        return slots
    
    def generate_response(self, intent: str, entities: Dict, 
                         username: str = None, language: str = "fr") -> str:
        """Generate appropriate response based on intent"""
        
        responses_fr = {
            "greeting": f"Bonjour {username if username else ''}! Comment puis-je vous aider aujourd'hui ?",
            "farewell": f"Au revoir {username if username else ''}! Passez une bonne journée.",
            "question_time": f"Il est {self._get_current_time()}.",
            "question_date": f"Nous sommes le {self._get_current_date()}.",
            "weather": f"Je ne peux pas accéder aux prévisions météo pour {entities.get('location', 'votre région')} pour le moment.",
            "help": "Je peux vous aider avec: reconnaissance vocale, transcription, questions simples, et plus encore. Que souhaitez-vous faire ?",
            "system_status": "Tous les systèmes sont opérationnels. Je suis prêt à vous aider!",
            "joke": "Pourquoi les plongeurs plongent-ils toujours en arrière et jamais en avant ? Parce que sinon ils tombent dans le bateau!",
            "music": f"Je vais essayer de trouver de la musique {entities.get('artist', '')}. Cependant, la lecture de musique n'est pas encore implémentée.",
            "volume": f"Je vais ajuster le volume à {entities.get('level', 'un niveau approprié')}%. Cependant, le contrôle du volume n'est pas encore implémenté.",
            "thanks": "Je vous en prie! C'est un plaisir de vous aider.",
            "unknown": "Je n'ai pas bien compris. Pouvez-vous reformuler votre demande ?"
        }
        
        responses_en = {
            "greeting": f"Hello {username if username else ''}! How can I help you today?",
            "farewell": f"Goodbye {username if username else ''}! Have a great day.",
            "question_time": f"The current time is {self._get_current_time()}.",
            "question_date": f"Today is {self._get_current_date()}.",
            "weather": f"I cannot access weather forecasts for {entities.get('location', 'your area')} at the moment.",
            "help": "I can help you with: voice recognition, transcription, simple questions, and more. What would you like to do?",
            "system_status": "All systems are operational. I'm ready to help!",
            "joke": "Why do divers always dive backwards and never forwards? Because otherwise they fall into the boat!",
            "music": f"I'll try to find music by {entities.get('artist', '')}. However, music playback is not yet implemented.",
            "volume": f"I'll adjust the volume to {entities.get('level', 'an appropriate level')}%. However, volume control is not yet implemented.",
            "thanks": "You're welcome! It's my pleasure to help.",
            "unknown": "I didn't quite understand that. Could you rephrase your request?"
        }
        
        responses = responses_fr if language == "fr" else responses_en
        return responses.get(intent, responses["unknown"])
    
    def _get_current_time(self) -> str:
        """Get current time formatted"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M")
    
    def _get_current_date(self) -> str:
        """Get current date formatted"""
        from datetime import datetime
        # French month names
        french_months = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        
        now = datetime.now()
        day = now.day
        month = french_months[now.month - 1]
        year = now.year
        
        return f"{day} {month} {year}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get intent recognizer status"""
        return {
            "use_llm": self.use_llm,
            "llm_loaded": self.llm_client is not None,
            "patterns_loaded": len(self.patterns) > 0,
            "patterns_count": len(self.patterns),
            "model_name": settings.HF_MODEL_NAME if settings.HF_MODEL_NAME else None
        }