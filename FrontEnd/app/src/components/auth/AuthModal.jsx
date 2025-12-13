import React, { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { FaMicrophone, FaStop, FaSpinner, FaUserCheck, FaUserPlus, FaFingerprint } from 'react-icons/fa';
import { useDispatch, useSelector } from 'react-redux';
import { loginSuccess } from '../../store/slices/authSlice';
import AudioVisualizer from '../audio/AudioVisualizer';
import { voiceApi } from '../../services/api';
// 1. Import de la bibliothèque stable
import { useReactMediaRecorder } from "react-media-recorder";

const AuthModal = () => {
  const [step, setStep] = useState(1);
  const [username, setUsername] = useState('');
  const [mode, setMode] = useState('register'); 
  const [targetUser, setTargetUser] = useState(null); 

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  
  const dispatch = useDispatch();
  const isOpen = useSelector(state => !state.auth.isAuthenticated);

  // --- HELPER : Nettoyage des erreurs ---
  const formatError = (err) => {
    if (err.message === "VOICE_MISMATCH") return "Voix non reconnue !";
    if (err.response && err.response.data) {
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) return detail.map(e => `${e.loc[e.loc.length - 1]}: ${e.msg}`).join(', ');
        if (typeof detail === 'string') return detail;
        if (typeof detail === 'object') return JSON.stringify(detail);
    }
    return err.message || "Erreur inconnue";
  };

  // --- LOGIQUE API (Exécutée quand l'enregistrement s'arrête) ---
  const handleAudioStop = async (blobUrl, blob) => {
    // Sécurité anti-vide
    if (!blob || blob.size < 1000) {
        console.warn("Audio trop court ignoré");
        return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    // Préparation du fichier WebM
    const audioFile = new File([blob], "voice_auth.webm", { type: "audio/webm" });
    const formData = new FormData();

    try {
      if (mode === 'register') {
        // === CAS 1: ENREGISTREMENT ===
        let userId;
        let currentUsername = username;

        // Création ou récupération de l'utilisateur
        if (targetUser && (targetUser.id || targetUser.user_id)) {
            console.log("Ajout empreinte pour utilisateur existant");
            userId = targetUser.id || targetUser.user_id;
            if (targetUser.username) currentUsername = targetUser.username;
        } else {
            console.log("Création nouvel utilisateur...");
            const userRes = await voiceApi.registerUser({ 
                username: username, 
                email: `${username}@voicegate.demo` 
            });
            userId = userRes.data.user_id || userRes.data.data?.user_id || userRes.data.id;
        }

        if (!userId) throw new Error("ID Utilisateur introuvable");

        formData.append('user_id', userId);
        formData.append('username', currentUsername); 
        formData.append('file', audioFile);
        
        await voiceApi.registerVoice(formData);
        
        completeLogin(userId);

      } else {
        // === CAS 2: VÉRIFICATION (LOGIN) ===
        console.log("Vérification identité pour:", targetUser?.username || username);
        
        formData.append('file', audioFile);
        // On envoie le username pour activer la vérification 1:1 (plus fiable)
        formData.append('username', targetUser?.username || username); 
        
        const identifyRes = await voiceApi.identifyVoice(formData);
        const data = identifyRes.data;

        // Vérification du succès
        if (data.identified === true) {
            // Sécurité supplémentaire sur l'ID
            const identifiedId = String(data.user_id);
            const targetId = String(targetUser.id || targetUser.user_id || targetUser._id);

            if (identifiedId === targetId) {
                completeLogin(targetId);
            } else {
                throw new Error("VOICE_MISMATCH");
            }
        } else {
            throw new Error("VOICE_MISMATCH");
        }
      }
    } catch (err) {
      console.error("Auth process failed:", err);
      const msg = formatError(err);
      setErrorMessage(msg);
      
      // Reset auto en cas d'échec d'auth
      if (msg === "Voix non reconnue !" || (err.response && err.response.status === 401)) {
         setTimeout(() => {
             setStep(1); 
             setUsername('');
             setErrorMessage(null);
         }, 2500);
      }
    } finally {
       setIsLoading(false);
    }
  };

  // --- HOOK MEDIA RECORDER ---
  const {
    status,
    startRecording,
    stopRecording,
    previewStream // Vital pour le visualiseur !
  } = useReactMediaRecorder({
    audio: true,
    blobPropertyBag: { type: "audio/webm" },
    onStop: handleAudioStop // Déclenche la logique API
  });

  const isRecording = status === "recording";

  // --- LOGIQUE AUTO-STOP ---
  useEffect(() => {
    let timeout;
    if (isRecording) {
        // 5s pour l'enregistrement (besoin de plus de données), 3s pour le login
        const duration = mode === 'register' ? 5000 : 3500;
        timeout = setTimeout(() => {
            stopRecording();
        }, duration);
    }
    return () => clearTimeout(timeout);
  }, [isRecording, mode, stopRecording]);


  // --- USER CHECK ---
  const handleCheckUsername = async () => {
    setErrorMessage(null);
    setIsLoading(true);

    try {
      const response = await voiceApi.getUser(username);
      const userData = response.data.data || response.data; 
      
      console.log("User found:", userData);
      setTargetUser(userData);

      if (userData.is_voice_registered) {
        setMode('login');
      } else {
        setMode('register');
      }
      setStep(2);

    } catch (error) {
      if (error.response && error.response.status === 404) {
        console.log("User not found -> Registration mode");
        setTargetUser(null);
        setMode('register');
        setStep(2);
      } else {
        setErrorMessage(formatError(error));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const completeLogin = (userId) => {
    setStep(3);
    setTimeout(() => {
      dispatch(loginSuccess({ username, id: userId }));
    }, 1500);
  };

  const getModeTitle = () => {
    if (step === 1) return "Identification";
    if (mode === 'login') return "Vérification Vocale";
    if (targetUser) return "Ajout d'empreinte vocale"; 
    return "Création de compte"; 
  };

  // Gestion du bouton micro
  const handleMicClick = () => {
      if (isRecording) {
          stopRecording(); // Arrêt manuel si l'utilisateur est pressé
      } else {
          startRecording();
      }
  };

  return (
    <Dialog open={isOpen} onClose={() => {}} className="relative z-50">
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-md rounded-2xl bg-surface border border-primary/20 p-8 text-gray-800 shadow-xl">
          
          <Dialog.Title className="text-2xl font-bold text-primary mb-2 flex items-center gap-2">
            {getModeTitle()}
          </Dialog.Title>

          {errorMessage && (
            <div className="mb-4 p-3 bg-red-500/20 border border-red-500 rounded text-red-100 text-sm animate-pulse whitespace-pre-wrap">
                ⚠️ {errorMessage}
            </div>
          )}

          {/* ÉTAPE 1 : USERNAME */}
          {step === 1 && (
            <div className="space-y-4">
              <p className="text-gray-400">Entrez votre nom d'utilisateur.</p>
              <input
                type="text"
                placeholder="Ex: jdoe"
                className="w-full bg-background border border-gray-700 rounded-lg p-3 text-gray-800 focus:border-primary outline-none placeholder-gray-500"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && username && handleCheckUsername()}
              />
              <button
                disabled={!username || isLoading}
                onClick={handleCheckUsername}
                className="w-full bg-primary hover:bg-indigo-600 py-3 rounded-lg font-semibold transition disabled:opacity-50 flex justify-center gap-2"
              >
                {isLoading ? <FaSpinner className="animate-spin" /> : "Continuer"}
              </button>
            </div>
          )}

          {/* ÉTAPE 2 : ENREGISTREMENT */}
          {step === 2 && (
            <div className="text-center space-y-6">
              <div className="bg-background/50 p-4 rounded-lg border border-gray-700">
                <p className="text-gray-300 text-sm mb-2 font-bold">
                  {mode === 'login' && <span className="text-secondary flex justify-center gap-2"><FaUserCheck/> Identité trouvée</span>}
                  {mode === 'register' && targetUser && <span className="text-accent flex justify-center gap-2"><FaFingerprint/> Enregistrement empreinte</span>}
                  {mode === 'register' && !targetUser && <span className="text-primary flex justify-center gap-2"><FaUserPlus/> Nouveau compte</span>}
                </p>
                <p className="text-gray-400">
                  {mode === 'register' 
                    ? "Pour sécuriser votre compte, lisez : " 
                    : "Prouvez que c'est vous : "}
                  <br/>
                  <span className="text-white italic text-lg">"Je confirme mon identité pour VoiceGate."</span>
                </p>
              </div>
              
              <div className="h-16 flex items-center justify-center">
                {isRecording ? (
                  /* Visualiseur branché sur le previewStream de la lib */
                  <AudioVisualizer stream={previewStream} isRecording={true} />
                ) : (
                  <div className="text-sm text-gray-500">
                      {isLoading ? "Analyse en cours..." : "Appuyez pour parler"}
                  </div>
                )}
              </div>

              <button
                onClick={handleMicClick}
                disabled={isLoading}
                className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-all transform ${
                  isRecording 
                    ? 'bg-red-500 scale-110 shadow-[0_0_20px_rgba(239,68,68,0.5)]' 
                    : isLoading 
                        ? 'bg-gray-600 cursor-not-allowed'
                        : 'bg-primary hover:scale-110 hover:shadow-[0_0_20px_rgba(99,102,241,0.5)]'
                }`}
              >
                {isLoading ? (
                    <FaSpinner className="animate-spin" size={24} />
                ) : isRecording ? (
                    <FaStop size={24} />
                ) : (
                    <FaMicrophone size={24} />
                )}
              </button>
            </div>
          )}

          {/* ÉTAPE 3 : SUCCÈS */}
          {step === 3 && (
            <div className="text-center text-secondary py-8 animate-bounce">
              <h3 className="text-2xl font-bold">Bienvenue !</h3>
            </div>
          )}
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default AuthModal;