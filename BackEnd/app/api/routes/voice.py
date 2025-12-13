# app/api/routes/voice.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register")
async def register_voice(
    user_id: str = Form(...),    # 1. On accepte str car FormData envoie tout en texte
    username: str = Form(None),  # 2. On accepte le username (optionnel) pour éviter le crash
    file: UploadFile = File(...)
):
    """Register user voice"""
    try:
        from services.database.user_repo import UserRepository
        from services.ai.biometrics import VoiceBiometricsService
        from services.audio.processor import AudioProcessor
        
        user_repo = UserRepository()
        biometrics = VoiceBiometricsService()
        audio_proc = AudioProcessor()
        
        # 3. Conversion intelligente de l'ID (String -> Int si nécessaire)
        # Si votre DB utilise des entiers (ID: 5), ceci est nécessaire
        try:
            uid = int(user_id)
        except ValueError:
            uid = user_id # On garde en string si c'est un ObjectId (MongoDB)

        # 4. Récupération de l'utilisateur
        user = user_repo.get_by_id(uid)
        
        if not user:
            # Fallback : Si on ne trouve pas par ID, on essaie par username si fourni
            if username:
                user = user_repo.get_by_username(username)
            
            if not user:
                raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
        
        # 5. Définition explicite du nom d'utilisateur pour la suite
        # C'est ici que l'erreur "name 'username' is not defined" est corrigée
        target_username = user.username 

        print(f"Processing voice for user: {target_username} (ID: {uid})") # Debug log sûr

        # Read audio
        audio_bytes = await file.read() # Note: 'await' est mieux avec UploadFile async
        audio_data = audio_proc.bytes_to_audio_array(audio_bytes)
        
        # Extract embedding
        embedding = biometrics.extract_embedding(audio_data)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract voice features")
        
        # Save using the safe variable 'target_username'
        success = user_repo.update_voice_embedding(target_username, embedding.tolist())
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save voice to database")
        
        return {
            "success": True,
            "username": target_username,
            "message": "Voice registered successfully",
            "embedding_length": len(embedding)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering voice: {e}")
        # Affiche l'erreur exacte dans la réponse pour déboguer
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify")
async def identify_voice(
    file: UploadFile = File(...),
    username: str = Form(None) # <--- AJOUT : On accepte le username
):
    """Identify speaker or Verify specific user"""
    try:
        from services.database.user_repo import UserRepository
        from services.ai.biometrics import VoiceBiometricsService
        from services.audio.processor import AudioProcessor
        
        user_repo = UserRepository()
        biometrics = VoiceBiometricsService()
        audio_proc = AudioProcessor()
        
        # 1. Traitement Audio (WebM -> Array) - UTILISATION LIBROSA ROBUSTE
        import tempfile
        import os
        import librosa
        
        # Sauvegarde temporaire pour lecture fiable par Librosa
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp_name = tmp.name
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            
        try:
            # Chargement audio (16k est le standard pour la biométrie)
            audio_data, _ = librosa.load(tmp_name, sr=16000)
        finally:
            if os.path.exists(tmp_name):
                try: os.unlink(tmp_name)
                except: pass

        # 2. Extraction Empreinte
        test_embedding = biometrics.extract_embedding(audio_data)
        if test_embedding is None:
            raise HTTPException(status_code=400, detail="Voix non détectée dans l'audio")

        # === MODE VÉRIFICATION (1:1) ===
        if username:
            user = user_repo.get_by_username(username)
            if not user or not user.voice_embedding:
                return {"identified": False, "message": "User has no voice print"}
            
            ref_emb = np.array(user.voice_embedding)
            score = biometrics.compare_embeddings(ref_emb, test_embedding)
            
            logger.info(f"Verification Score for {username}: {score} (Threshold: {biometrics.threshold})")
            
            # Note: Pour la vérification 1:1, on peut être un peu plus tolérant ou strict selon besoin
            if score >= biometrics.threshold:
                return {
                    "identified": True,
                    "username": user.username,
                    "user_id": user.id,
                    "confidence": score,
                    "mode": "verification"
                }
            else:
                return {
                    "identified": False, 
                    "message": "Voice mismatch",
                    "confidence": score
                }

        # === MODE IDENTIFICATION (1:N) - Ancien code === 
        registered_users = user_repo.get_voice_registered_users() 
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error identifying voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))