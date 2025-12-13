#  VoiceGate - Frontend

VoiceGate est une interface de chat vocal intelligente sécurisée par biométrie vocale.  
Ce frontend permet l'authentification par la voix, l'enregistrement audio et la communication avec une IA via une API Python.

---

## Fonctionnalités

- **Authentification Biométrique** : Connexion / inscription via empreinte vocale.
- **Chat Vocal & Texte** :
  - Enregistrement audio via microphone.
  - Upload de fichiers audio (WAV, MP3…)
  - Saisie textuelle classique.
- **Synthèse Vocale (TTS)** : Lecture automatique des réponses de l’IA.
- **Visualisation Audio** : Waveform en temps réel.
- **Historique** : Chargement et persistance des conversations.
- **Paramètres** : Suppression de compte, suppression de l’historique.

---

##  Stack Technique

- **Framework** : React (Vite)
- **État global** : Redux Toolkit
- **UI** : Tailwind CSS, Headless UI, React Icons
- **Audio** : react-media-recorder
- **HTTP** : Axios

---

##  Installation

### 1. Prérequis
Avoir **Node.js v16+**

### 2. Cloner le projet
```bash
git clone https://github.com/votre-repo/voicegate-frontend.git
cd voicegate-frontend
```

### 3. Installer les dépendances
```bash
npm install
```

### 4. Configuration `.env`
Créer un fichier `.env` :

```env
VITE_API_URL=http://localhost:8002/api
```

---

## ▶️ Démarrage

```bash
npm run dev
```

L’application démarre sur :
```
http://localhost:5173
```

---

##  Structure du Projet

```text
src/
├── components/
│   ├── audio/          # Visualisation audio
│   ├── auth/           # Connexion/inscription vocale
│   ├── chat/           # Interface du chat + upload audio
│   └── settings/       # Paramètres utilisateur
├── services/
│   └── api.js          # Axios + endpoints
├── store/
│   ├── slices/         # Redux slices (auth, chat)
│   └── store.js        # Store principal
├── App.jsx             # Layout principal
└── main.jsx            # Point d'entrée
```

---

##  Intégration Backend

Fonctionne avec le backend **VoiceGate (FastAPI)**, sur le port `8002` (modifiable dans `.env`).

Endpoints principaux utilisés :

- `POST /api/voice/register`
- `POST /api/voice/identify`
- `POST /api/voice-conversation/voice-ask`
- `GET /api/voice-conversation/conversations/{id}/last/messages`

---

##  Contribution

1. Forkez le projet  
2. Créez une branche (`git checkout -b feature/AmazingFeature`)  
3. Commit (`git commit -m "Add AmazingFeature"`)  
4. Push (`git push origin feature/AmazingFeature`)  
5. Ouvrez une Pull Request  

