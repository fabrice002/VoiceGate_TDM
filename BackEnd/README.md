# VoiceGate Backend

<div align="center">

![VoiceGate Logo](https://img.shields.io/badge/VoiceGate-AI%20Assistant-blueviolet)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green)

**Assistant vocal intelligent avec reconnaissance du locuteur en temps réel**

[Documentation API](http://127.0.0.1:8002/docs) • [Installation](#-installation) • [Utilisation](#-utilisation) • [API Reference](#-référence-api)

</div>

## 🌟 Présentation

VoiceGate est un assistant vocal intelligent capable de :
- 🎤 **Reconnaître les locuteurs** via empreintes vocales (ECAPA-TDNN)
- 📝 **Transcrire la parole** en texte avec Whisper
- 💬 **Dialoguer intelligemment** avec reconnaissance d'intention
- 🔊 **Répondre oralement** avec synthèse vocale multi-moteurs
- ⚡ **Fonctionner en temps réel** via WebSocket

**Architecture :** FastAPI + MongoDB + Whisper + WebSocket

## 🚀 Installation Rapide

### Prérequis
```bash
Python 3.9+ | FFmpeg | MongoDB (optionnel)
```

### 1. Cloner le projet
```bash
git clone https://github.com/fabrice002/VoiceGate_TDM.git
cd BackEnd
```

### 2. Configuration
```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer .env (optionnel)
nano .env
```

### 3. Installation avec Poetry (recommandé)
```bash
# Installer Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Installer les dépendances
poetry install

# Activer l'environnement virtuel
poetry shell
```

### 4. Installation avec pip
```bash
pip install -r requirements.txt
```

### 5. Lancer le serveur
```bash
# Mode développement
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8002

# Mode production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

## 📁 Structure du Projet

```
voicegate-backend/
├── app/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── core/
│   │   ├── config.py        # Configuration
│   │   └── database.py      # MongoDB + Mock DB
│   ├── api/routes/          # Routes API
│   ├── services/           # Services métier
│   ├── models/             # Modèles Pydantic
│   └── schemas/            # Schémas API
├── data/                   # Données persistantes
│   ├── mock_db/           # Base de données mock
│   ├── voice_embeddings/  # Empreintes vocales
│   └── audio_files/       # Fichiers audio 
```

## 🔧 Configuration

### Variables d'environnement (.env)
```env
# Application
APP_NAME=VoiceGate AI Assistant
DEBUG=True
PORT=8002

# Base de données
MONGODB_URI=mongodb://localhost:27017
MONGO_DB_NAME=voicegate_db
USE_MONGODB=False  # True pour MongoDB, False pour Mock DB

# Modèles AI
WHISPER_MODEL=base
WHISPER_LANGUAGE=fr
ECAPA_MODEL=speechbrain/spkrec-ecapa-voxceleb
HF_MODEL_NAME=microsoft/DialoGPT-small

# Paths
VOICE_DB_FOLDER=data/voice_embeddings
AUDIO_STORAGE_PATH=data/audio_files
```

## 🎯 Utilisation

### 1. Vérifier l'installation
```bash
curl http://127.0.0.1:8002/health
# Réponse: {"status": "healthy", "database": "Mock"}
```

### 2. Créer un utilisateur
```bash
curl -X POST http://127.0.0.1:8002/api/users/ \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@example.com"}'
```

### 3. Enregistrer une voix
```bash
curl -X POST http://127.0.0.1:8002/api/voice/register \
  -F "username=alice" \
  -F "file=@voix.wav"
```

### 4. Tester la reconnaissance vocale
```bash
curl -X POST http://127.0.0.1:8002/api/assistant/process \
  -F "file=@audio_test.wav"
```

### 5. Utiliser le pipeline complet
```python
import requests

# 1. Transcrire
transcription = requests.post(
    "http://127.0.0.1:8002/api/transcription/transcribe",
    files={"file": open("audio.wav", "rb")},
    data={"language": "fr"}
).json()

# 2. Générer réponse TTS
tts_response = requests.post(
    "http://127.0.0.1:8002/api/tts/generate",
    json={
        "text": f"Bonjour! Vous avez dit: {transcription['text']}",
        "language": "fr"
    }
).json()

print(f"Audio généré: {tts_response['audio_url']}")
```

## 📡 Référence API

### Endpoints Principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| **GET** | `/` | Page d'accueil avec statut |
| **GET** | `/docs` | Documentation Swagger |
| **GET** | `/health` | Santé du système |

### 👤 Gestion Utilisateurs
- `POST /api/users/` - Créer un utilisateur
- `GET /api/users/` - Lister les utilisateurs
- `GET /api/users/{username}` - Obtenir un utilisateur
- `DELETE /api/users/{username}` - Supprimer un utilisateur

### 🎤 Reconnaissance Vocale
- `POST /api/voice/register` - Enregistrer empreinte vocale
- `POST /api/voice/identify` - Identifier un locuteur

### 📝 Transcription
- `POST /api/transcription/transcribe` - Transcrire audio
- `POST /api/transcription/transcribe-base64` - Transcrire audio base64

### 💬 Conversation
- `POST /api/voice-conversation/voice-ask` - Pipeline complet voix→réponse
- `GET /api/voice-conversation/conversations/{user_id}/voice` - Historique

### 🔊 Text-to-Speech
- `POST /api/tts/generate` - Générer audio depuis texte
- `GET /api/tts/stream` - Stream audio en direct

### ⚡ WebSocket Temps Réel
- `WS /ws/ws/audio/{user_id}` - Streaming audio bidirectionnel
- `WS /ws/ws/logs` - Logs temps réel
- `WS /ws/ws/monitoring` - Métriques temps réel



## 🐛 Dépannage

### Problèmes courants

1. **Erreur "FFmpeg not found"**
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows (choco)
   choco install ffmpeg
   ```

2. **Port déjà utilisé**
   ```bash
   # Changer le port dans .env
   PORT=8003
   ```

3. **Base de données non connectée**
   ```bash
   # Vérifier MongoDB
   mongod --version
   
   # Ou utiliser Mock DB
   USE_MONGODB=False
   ```

4. **Modèles non téléchargés**
   ```bash
   # Les modèles se téléchargent automatiquement
   # Vérifier le dossier data/pretrained_models/
   ```

## 📈 Monitoring

### Dashboard intégré
- Accéder à: `http://127.0.0.1:8002/api/monitoring/metrics`
- Métriques en temps réel: latence, succès, performance

### Logs
```bash
# Mode debug
DEBUG=True

# Voir les logs
tail -f logs/app.log
```

## 🚀 Déploiement

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  voicegate:
    build: .
    ports:
      - "8002:8002"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017
      - USE_MONGODB=True
    depends_on:
      - mongodb

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

## 📚 Documentation Additionnelle

- [Guide Whisper](docs/whisper_guide.md)
- [API Swagger](http://127.0.0.1:8002/docs)
- [Schéma Base de Données](docs/database_schema.md)
- [Architecture](docs/architecture.md)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - Voir le fichier [LICENSE](LICENSE)

## 🙏 Remerciements

- [OpenAI Whisper](https://github.com/openai/whisper) pour la transcription
- [SpeechBrain](https://speechbrain.github.io/) pour ECAPA-TDNN
- [FastAPI](https://fastapi.tiangolo.com/) pour le backend
- [FFmpeg](https://ffmpeg.org/) pour le traitement audio



---

<div align="center">
  
**VoiceGate** - Votre assistant vocal intelligent

[⬆ Retour en haut](#voicegate-backend)

</div>