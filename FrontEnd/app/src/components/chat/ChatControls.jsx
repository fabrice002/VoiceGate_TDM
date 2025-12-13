// src/components/chat/ChatControls.jsx

import React, { useState, useRef, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { FaMicrophone, FaPaperPlane, FaKeyboard, FaStop, FaPaperclip } from 'react-icons/fa';
import { addMessage, setTyping } from '../../store/slices/chatSlice';
import { voiceApi } from '../../services/api';
import { useReactMediaRecorder } from "react-media-recorder";

// --- HELPER: Detect Supported Browser MIME Type ---
// Browsers cannot record directly to MP3. We prioritize WebM (Chrome/Firefox) 
// and fallback to MP4/AAC (Safari) or WAV.
const getSupportedMimeType = () => {
  if (typeof window === 'undefined' || !window.MediaRecorder) return "";

  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4", // Safari (iOS/MacOS) preferred
    "audio/aac",
    "audio/ogg;codecs=opus",
    "audio/wav"
  ];

  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return ""; // Browser default
};

// --- HELPER: Get Correct Extension ---
const getExtensionFromMime = (mime) => {
  if (mime.includes("mp4") || mime.includes("aac")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("wav")) return "wav";
  return "webm"; // Default fallback
};

const ChatControls = () => {
  const [mode, setMode] = useState('voice');
  const [inputText, setInputText] = useState('');
  
  // Ref for hidden file input
  const fileInputRef = useRef(null);

  const { user } = useSelector(state => state.auth);
  const dispatch = useDispatch();

  const toggleMode = () => setMode(mode === 'voice' ? 'text' : 'voice');

  // Determine browser capability once on mount
  const mimeType = useMemo(() => getSupportedMimeType(), []);
  const fileExtension = useMemo(() => getExtensionFromMime(mimeType), [mimeType]);

  // --- 1. CENTRAL AUDIO SUBMISSION ---
  const handleAudioSubmission = async (audioFile, localUrl) => {
    // UI: Display User Message (Audio Player)
    dispatch(addMessage({
        id: Date.now(),
        text: "",
        sender: 'user',
        timestamp: new Date().toISOString(),
        audioUrl: localUrl,
        isAudio: true
    }));
    
    dispatch(setTyping(true));

    try {
        const formData = new FormData();
        formData.append('user_id', user?.id ? String(user.id) : "guest");
        formData.append('audio_file', audioFile);
        formData.append('language', 'fr');

        // API Call
        const response = await voiceApi.sendVoiceAsk(formData);

        dispatch(setTyping(false));

        const aiResponse = response.data.ai_response;

        // UI: Display AI Response
        dispatch(addMessage({
            id: Date.now() + 2,
            text: aiResponse,
            sender: 'ai',
            timestamp: new Date().toISOString(),
            audioUrl: response.data.audio_url
        }));

    } catch (error) {
        console.error("Error sending voice:", error);
        dispatch(setTyping(false));
        dispatch(addMessage({
            id: Date.now(),
            text: "Désolé, une erreur est survenue lors du traitement audio.",
            sender: 'ai',
            timestamp: new Date().toISOString(),
        }));
    }
  };

  // --- 2. MICROPHONE RECORDING MANAGEMENT ---
  const handleStopRecording = (blobUrl, blob) => {
    if (!blob || blob.size < 1000) {
        console.warn("Audio too short or empty, ignored.");
        return;
    }

    // Use dynamic extension and correct MIME type
    const fileName = `voice_record.${fileExtension}`;
    
    // Create the file using the actual MIME type the browser used
    const audioFile = new File([blob], fileName, { 
        type: mimeType || blob.type 
    });
    
    // Call central submission
    handleAudioSubmission(audioFile, blobUrl);
  };

  const { status, startRecording, stopRecording } = useReactMediaRecorder({
    audio: true, // Hardware constraints (echo cancellation, etc.)
    mediaRecorderOptions: {
        mimeType: mimeType // Use the detected supported type
    },
    onStop: async (blobUrl, blob) => {
        // console.log("Blob size:", blob.size, "Type:", blob.type);
        handleStopRecording(blobUrl, blob);
    }
  });

  const isRecording = status === "recording";

  // --- 3. FILE UPLOAD MANAGEMENT ---
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Create local URL for preview
    const localUrl = URL.createObjectURL(file);
    
    handleAudioSubmission(file, localUrl);

    // Reset input to allow re-uploading the same file
    e.target.value = null;
  };

  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  // --- 4. TEXT MANAGEMENT ---
  const handleSendText = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    dispatch(addMessage({
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date().toISOString()
    }));
    
    const textToSend = inputText;
    setInputText('');
    dispatch(setTyping(true));

    try {
      const payload = {
        user_id: user?.id ? String(user.id) : "guest",
        text: textToSend,
        language: "fr"
      };

      const response = await voiceApi.sendTextAsk(payload);
      dispatch(setTyping(false));
      
      const aiText = response.data.ai_response || response.data.message;
      dispatch(addMessage({
        id: Date.now() + 1,
        text: aiText,
        sender: 'ai',
        timestamp: new Date().toISOString(),
        audioUrl: response.data.audio_url
      }));

    } catch (error) {
      console.error("Error sending text:", error);
      dispatch(setTyping(false));
    }
  };

  return (
    <div className="bg-surface border-t border-gray-700 p-4">
      <div className="max-w-4xl mx-auto flex items-end gap-3">
        
        {/* Mode Toggle (Keyboard / Mic) */}
        <button 
          onClick={toggleMode}
          className="p-3 text-gray-400 hover:text-white bg-gray-800 rounded-full transition"
          title={mode === 'voice' ? "Passer au texte" : "Passer au vocal"}
        >
          {mode === 'voice' ? <FaKeyboard /> : <FaMicrophone />}
        </button>

        {/* Hidden Input for Upload */}
        <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            accept="audio/*" // Accepts all audio types
            className="hidden" 
        />

        {/* Central Area */}
        <div className="flex-1 bg-background rounded-2xl p-2 min-h-[50px] flex items-center relative overflow-hidden border border-gray-700 focus-within:border-primary transition-colors">
          
          {/* Upload Button */}
          <button 
            onClick={triggerFileUpload}
            className="p-2 text-gray-400 hover:text-white mr-2 transition"
            title="Uploader un fichier audio"
          >
            <FaPaperclip />
          </button>

          {mode === 'text' ? (
            <form onSubmit={handleSendText} className="w-full flex">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Écrivez un message..."
                className="w-full bg-transparent text-gray-800 placeholder-gray-500 px-2 outline-none"
              />
            </form>
          ) : (
            <div className="w-full flex items-center justify-center h-10">
               {isRecording ? (
                 <div className="flex items-center gap-3">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span className="text-red-400 font-medium text-sm animate-pulse">
                        Enregistrement...
                    </span>
                 </div>
               ) : (
                 <span className="text-gray-500 text-sm cursor-pointer select-none" onClick={startRecording}>
                   Maintenez le micro ou uploadez un fichier
                 </span>
               )}
            </div>
          )}
        </div>

        {/* Action Button (Send / Record) */}
        {mode === 'text' ? (
          <button 
            onClick={handleSendText}
            className="p-3 bg-primary text-white rounded-full hover:bg-indigo-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={!inputText.trim()}
          >
            <FaPaperPlane />
          </button>
        ) : (
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
            onMouseLeave={stopRecording}
            className={`p-3 rounded-full text-white transition-all transform ${
              isRecording ? 'bg-red-500 scale-110 shadow-lg shadow-red-500/50' : 'bg-secondary hover:bg-emerald-600'
            }`}
          >
            {isRecording ? <FaStop /> : <FaMicrophone />}
          </button>
        )}
      </div>
    </div>
  );
};

export default ChatControls;