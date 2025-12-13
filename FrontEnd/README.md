# VoiceGate Frontend

<div align="center">

![VoiceGate Logo](https://img.shields.io/badge/VoiceGate-Web%20Interface-blueviolet)
![React](https://img.shields.io/badge/React-18.2-blue)
![Vite](https://img.shields.io/badge/Vite-5.0-purple)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-sky)
![Redux Toolkit](https://img.shields.io/badge/Redux-Toolkit-purple)

**Interface web de chat vocal sécurisée par biométrie vocale**

[Voir le Backend](https://github.com/fabrice002/VoiceGate_TDM) • [Installation](#-installation-rapide) • [Utilisation](#-utilisation) • [Configuration](#-configuration)

</div>

##  Présentation

VoiceGate Frontend est l'interface utilisateur moderne conçue pour interagir avec l'assistant vocal VoiceGate. Elle offre une expérience fluide centrée sur la voix et la sécurité.

**Fonctionnalités principales :**
-  **Authentification Biométrique** : Connexion et inscription sécurisées par empreinte vocale.
-  **Chat Multimodal** :
  - Enregistrement vocal en temps réel.
  - **Upload de fichiers audio** (WAV, MP3, WebM).
  - Saisie textuelle classique.
-  **Synthèse Vocale (TTS)** : Lecture automatique des réponses de l'IA.
-  **Visualisation Audio** : Waveform dynamique lors de l'enregistrement.
-  **Historique Persistant** : Reprise des conversations précédentes.

**Architecture :** React + Vite + Redux Toolkit + Tailwind CSS

---

##  Installation Rapide

### Prérequis
```bash
Node.js v16+ | npm ou yarn


### 1\. Cloner le projet

```bash
git clone [https://github.com/votre-repo/voicegate-frontend.git](https://github.com/votre-repo/voicegate-frontend.git)
cd voicegate-frontend
```

### 2\. Installer les dépendances

```bash
npm install
# ou
yarn install
```

### 3\. Configuration

Créez un fichier `.env` à la racine du projet pour lier le frontend à votre API Backend.

```bash
cp .env.example .env
```

**Contenu du fichier `.env` :**

```env
# URL de l'API Backend (FastAPI)
VITE_API_URL=http://localhost:8002/api
```

### 4\. Lancer le serveur de développement

```bash
npm run dev
```

L'application sera accessible sur `http://localhost:5173`.



##  Structure du Projet

```text
voicegate-frontend/
├── src/
│   ├── components/
│   │   ├── audio/          # Visualiseurs et lecteurs audio
│   │   ├── auth/           # Modales de connexion/inscription vocale
│   │   ├── chat/           # Interface de chat, Micro et Upload
│   │   └── settings/       # Gestion de compte et préférences
│   ├── services/
│   │   └── api.js          # Configuration Axios et Endpoints
│   ├── store/
│   │   ├── slices/         # Reducers Redux (Auth, Chat)
│   │   └── store.js        # Configuration du Store global
│   ├── App.jsx             # Layout principal et Routing
│   └── main.jsx            # Point d'entrée React
├── public/                 # Assets statiques
└── package.json            # Dépendances et scripts
```



##  Utilisation

### 1\. Authentification

Au lancement, choisissez **"Connexion Vocale"** ou **"Inscription"**.

  - Cliquez sur le micro et prononcez votre phrase pass (ou votre nom).
  - Le système vérifie votre empreinte vocale via le backend.

### 2\. Conversation

Une fois connecté, accédez à l'interface de chat :

  - **Mode Vocal** : Maintenez le bouton Micro pour parler. Relâchez pour envoyer.
  - **Mode Fichier** : Cliquez sur le trombone 📎 pour uploader un fichier audio existant.
  - **Mode Texte** : Basculez sur le clavier pour écrire.

### 3\. Commandes Disponibles

| Action | Description |
|--------|-------------|
| **Microphone** | Enregistrement vocal direct (WebM) |
| **Upload** | Envoi de fichiers audio pré-enregistrés |
| **Settings** | Supprimer l'historique ou le compte utilisateur |



##  Intégration Backend

Ce frontend consomme les endpoints suivants du Backend VoiceGate :

| Méthode | Endpoint | Usage |
|---------|----------|-------|
| **POST** | `/voice/register` | Enrôlement d'une nouvelle voix |
| **POST** | `/voice/identify` | Authentification biométrique |
| **POST** | `/voice-conversation/voice-ask` | Traitement audio (Speech-to-Text + AI + TTS) |
| **POST** | `/voice-conversation/text-ask` | Chat textuel classique |
| **GET** | `/conversations/{id}/last/messages` | Récupération de l'historique |



##  Déploiement (Docker)

Pour déployer l'application en production avec Nginx :

```bash
# 1. Construire l'image
docker build -t voicegate-frontend .

# 2. Lancer le conteneur
docker run -p 80:80 voicegate-frontend
```

*Exemple de Dockerfile simple :*

```dockerfile
# Build Stage
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Serve Stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```



##  Contribution

1.  Forkez le projet
2.  Créez votre branche (`git checkout -b feature/AmazingFeature`)
3.  Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4.  Push vers la branche (`git push origin feature/AmazingFeature`)
5.  Ouvrez une Pull Request


