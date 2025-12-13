
#  VoiceGate - Full Stack AI Assistant

<div align="center">

![VoiceGate Logo](https://img.shields.io/badge/VoiceGate-AI%20Biometric%20Assistant-blueviolet?style=for-the-badge)

![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?style=flat&logo=mongodb&logoColor=white)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5.0-646CFF?style=flat&logo=vite&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-3.4-38B2AC?style=flat&logo=tailwindcss&logoColor=white)

**A secure, voice-activated AI assistant featuring biometric speaker recognition and real-time conversation.**

[Backend API Docs](http://localhost:8002/docs) • [Frontend Interface](http://localhost:5173) • [Installation](#-installation-guide)

</div>



##  Overview

**VoiceGate** is a complete solution for secure voice interaction. It combines a high-performance **FastAPI** backend handling AI inference (Whisper, Speaker Verification) with a modern **React** frontend providing a seamless user experience.

Unlike standard assistants, VoiceGate verifies **who** is speaking before processing commands, adding a layer of biometric security to voice interactions.

###  Key Features

| Category | Features |
|:---:|---|
| **Security** | **Biometric Authentication**: Log in and verify identity using your unique voiceprint (ECAPA-TDNN). |
| **Interaction** | **Multi-modal Input**: Real-time microphone recording, Audio File Upload (.wav, .mp3), or Text input. |
| **AI Core** | **Speech-to-Text**: Powered by OpenAI Whisper.<br>**Intent Recognition**: Context-aware dialogue.<br>**TTS**: Natural sounding Text-to-Speech responses. |
| **Interface** | **Real-time Visualization**: Dynamic audio waveforms.<br>**History**: Persistent conversation logs.<br>**Modern UI**: Dark mode, responsive design built with Tailwind. |



##  Project Architecture

The project consists of two main components communicating via REST API and WebSockets:

```text
VoiceGate/
├── backend/               # FastAPI Server (Python)
│   ├── app/               # API Routes, WebSocket, Logic
│   ├── services/          # AI Models (Whisper, TTS, Speaker Rec)
│   └── data/              # Voice embeddings & Audio storage
│
└── frontend/              # React Client (JavaScript/Vite)
    ├── src/components/    # Chat UI, Audio Visualizers
    ├── src/store/         # Redux State Management
    └── src/services/      # API Connectors
````



## Installation Guide

### Prerequisites

  * **System**: Python 3.9+, Node.js v16+, FFmpeg installed on system path.
  * **Database**: MongoDB (optional, can run with Mock DB).



### Part 1: Backend Setup 

1.  **Navigate to backend:**

    ```bash
    cd backend
    ```

2.  **Install Dependencies (using Poetry or Pip):**

    ```bash
    # Using Pip
    pip install -r requirements.txt

    # OR Using Poetry
    poetry install && poetry shell
    ```

3.  **Configure Environment:**

    ```bash
    cp .env.example .env
    ```

    *Edit `.env` if necessary (Default port: 8002).*

4.  **Start Server:**

    ```bash
    python -m uvicorn app.main:app --reload --port 8002
    ```

    *Backend is now running at `http://localhost:8002`*



### Part 2: Frontend Setup 

1.  **Open a new terminal and navigate to frontend:**

    ```bash
    cd frontend
    ```

2.  **Install Dependencies:**

    ```bash
    npm install
    ```

3.  **Configure Environment:**
    Create a `.env` file in the `frontend/` root:

    ```env
    VITE_API_URL=http://localhost:8002/api
    ```

4.  **Start Client:**

    ```bash
    npm run dev
    ```

    *Frontend is now running at `http://localhost:5173`*



##  Usage

1.  **Open the App**: Go to `http://localhost:5173`.
2.  **Register Voice**:
      * Click **"S'inscrire"**.
      * Enter a username.
      * Record a short phrase to create your voice biometric profile.
3.  **Login**: Use **"Connexion Vocale"** to verify your identity.
4.  **Chat**:
      *  **Voice**: Hold the microphone button to speak.
      *  **Upload**: Click the paperclip to analyze audio files.
      *  **Text**: Type naturally to interact with the AI.



## Tech Stack

### Backend (Python)

  * **Framework**: FastAPI, Uvicorn
  * **AI/ML**: OpenAI Whisper, SpeechBrain (ECAPA-TDNN), gTTS/pyttsx3
  * **Data**: MongoDB, Motor (Async driver)
  * **Utils**: FFmpeg, Librosa

### Frontend (JavaScript)

  * **Core**: React, Vite
  * **State**: Redux Toolkit
  * **Styling**: Tailwind CSS, Headless UI, React Icons
  * **Audio**: React Media Recorder



##  API Reference

Once the backend is running, full Swagger documentation is available at:
 **`http://localhost:8002/docs`**

**Key Endpoints:**

  * `POST /api/voice/register` - Enroll a new voice print.
  * `POST /api/voice/identify` - Authenticate user via audio.
  * `POST /api/voice-conversation/voice-ask` - Full pipeline (STT -\> AI -\> TTS).
  * `GET /api/conversations/{id}/history` - Retrieve chat history.



##  Docker Deployment

To launch the full stack using Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8002:8002"]
    environment:
      - USE_MONGODB=True
      - MONGODB_URI=mongodb://mongo:27017
    depends_on: [mongo]

  frontend:
    build: ./frontend
    ports: ["5173:80"] # Nginx container
    depends_on: [backend]

  mongo:
    image: mongo:latest
    ports: ["27017:27017"]
```

Run: `docker-compose up --build`



##  Contributing

Contributions are welcome\!

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit changes (`git commit -m 'Add NewFeature'`).
4.  Push to branch (`git push origin feature/NewFeature`).
5.  Open a Pull Request.


