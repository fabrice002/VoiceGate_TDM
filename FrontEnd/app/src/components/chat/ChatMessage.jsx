import React, { useState, useRef } from 'react';
import { FaRobot, FaUser, FaPlay, FaPause } from 'react-icons/fa';

const ChatMessage = ({ message }) => {
  const isAi = message.sender === 'ai';
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const toggleAudio = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Reset icon when audio ends
  const handleEnded = () => setIsPlaying(false);

  return (
    <div className={`flex w-full mb-6 ${isAi ? 'justify-start' : 'justify-end'}`}>
      <div className={`flex max-w-[80%] ${isAi ? 'flex-row' : 'flex-row-reverse'} gap-3`}>
        
        {/* Avatar */}
        <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
          isAi ? 'bg-primary text-white' : 'bg-secondary text-white'
        }`}>
          {isAi ? <FaRobot /> : <FaUser />}
        </div>

        {/* Bulle de message */}
        <div className={`p-4 rounded-2xl shadow-md min-w-[120px] ${
          isAi 
            ? 'bg-surface text-gray-100 rounded-tl-none border border-gray-700' 
            : 'bg-primary text-white rounded-tr-none'
        }`}>
          
          {/* LECTEUR AUDIO (WhatsApp style) */}
          {message.audioUrl ? (
            <div className="flex items-center gap-3">
              <button 
                onClick={toggleAudio}
                className="w-10 h-10 rounded-full bg-black/20 flex items-center justify-center hover:bg-black/30 transition flex-shrink-0"
              >
                {isPlaying ? <FaPause size={14} /> : <FaPlay size={14} className="ml-1" />}
              </button>
              
              <div className="flex flex-col justify-center flex-1 min-w-[100px]">
                 {/* Visualisation Waveform simplifiée (Barre) */}
                 <div className="h-1 bg-black/20 rounded-full w-full overflow-hidden">
                    <div className={`h-full bg-white/80 transition-all duration-300 ${isPlaying ? 'w-full animate-pulse' : 'w-0'}`} />
                 </div>
                 <span className="text-[10px] opacity-70 mt-1">Message Audio</span>
              </div>

              {/* Élément Audio caché */}
              <audio 
                ref={audioRef} 
                src={message.audioUrl} 
                onEnded={handleEnded} 
                className="hidden" 
              />
            </div>
          ) : (
            /* TEXTE NORMAL */
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
          )}
          
          <div className="mt-1 flex justify-end">
            <span className="text-[10px] opacity-60">
              {new Date(message.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;