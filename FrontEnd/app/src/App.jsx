// src/App.jsx
import React, { useEffect, useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { FaCog, FaSignOutAlt, FaWaveSquare } from 'react-icons/fa';
import AuthModal from './components/auth/AuthModal';
import ChatMessage from './components/chat/ChatMessage';
import ChatControls from './components/chat/ChatControls';
import { voiceApi } from './services/api';
import { setHistory, clearConversation } from './store/slices/chatSlice';
import { logout } from './store/slices/authSlice';
import SettingsModal from './components/settings/SettingsModal'

const App = () => {
  const { messages, isTyping } = useSelector(state => state.chat);
  const { user, isAuthenticated } = useSelector(state => state.auth);
  const dispatch = useDispatch();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  // Auto-scroll vers le bas
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages, isTyping]);

  // --- CHARGEMENT DE L'HISTORIQUE ---
  useEffect(() => {
    const loadHistory = async () => {
      // On ne charge que si l'utilisateur est authentifié et a un ID
      if (isAuthenticated && user?.id) {
        try {
          console.log("📥 Chargement de l'historique pour:", user.username);
          const response = await voiceApi.getLastConversationMessages(user.id);
          
          if (response.data && response.data.messages) {
            // Transformation des données Backend -> Frontend
            const history = response.data.messages.map(msg => ({
              id: msg._id || msg.id,
              text: msg.content,
              sender: msg.role === 'assistant' ? 'ai' : 'user',
              timestamp: msg.timestamp,
              // Reconstruction de l'URL audio si elle existe
              // Note: Assurez-vous que le port 8002 correspond à votre backend
              audioUrl: msg.audio_file_path 
                ? `http://localhost:8002/${msg.audio_file_path}` 
                : null,
              isAudio: msg.source === 'voice'
            }));

            // Mise à jour de Redux
            dispatch(setHistory({
              messages: history,
              conversationId: response.data.conversation_id
            }));
          }
        } catch (error) {
          console.error(" Erreur chargement historique:", error);
        }
      }
    };

    loadHistory();
  }, [isAuthenticated, user, dispatch]);

  // --- GESTION DÉCONNEXION ---
  const handleLogout = () => {
    dispatch(logout());
    dispatch(clearConversation());
    // Cela réouvrira automatiquement l'AuthModal car isAuthenticated passera à false
  };

  return (
    <div className="flex flex-col h-screen bg-background font-sans overflow-hidden">
      
      {/* 1. Header */}
      <header className="h-16 bg-surface border-b border-gray-700 flex items-center justify-between px-4 sm:px-6 shadow-lg z-10">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-tr from-primary to-secondary p-2 rounded-lg">
            <FaWaveSquare className="text-white text-xl" />
          </div>
          <h1 className="text-xl font-bold tracking-tight">VoiceGate</h1>
        </div>
        
        {isAuthenticated && (
          <div className="flex items-center gap-4">
             <span className="text-sm text-gray-400 hidden sm:block">
               {user.username}
             </span>
             <button 
               onClick={() => setIsSettingsOpen(true)} // Ouvre la modale
               className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-full transition"
               title="Paramètres"
               aria-label="Paramètres"
             >
               <FaCog />
             </button>
             <button 
               onClick={handleLogout}
               className="p-2 text-white hover:text-red-300 hover:bg-red-900/20 rounded-full transition"
               title="Se déconnecter"
               aria-label="Se déconnecter"
             >
               {/* <FaSignOutAlt className="text-red-500" /> */}
             </button>
          </div>
        )}
      </header>

      {/* 2. Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
        <div className="max-w-4xl mx-auto flex flex-col min-h-full justify-end">
          
          {/* Welcome Message Empty State */}
          {messages.length === 0 && (
             <div className="flex font-bold flex-col items-center justify-center text-gray-500 py-10 opacity-50">
                <FaWaveSquare size={48} className="mb-4" />
                <p>Aucune conversation. Commencez à parler !</p>
             </div>
          )}

          {/* Message List */}
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* Typing Indicator */}
          {isTyping && (
             <div className="flex justify-start w-full mb-6">
               <div className="bg-surface border border-gray-700 p-4 rounded-2xl rounded-tl-none flex gap-1">
                 <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                 <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                 <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
               </div>
             </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* 3. Controls */}
      <ChatControls />

      {/* Modals */}
      <AuthModal />
      {/* Ajout de la SettingsModal */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />
    </div>
  );
};

export default App;