import { useEffect, useRef } from 'react';
import { useDispatch } from 'react-redux';
import { addMessage, setTyping } from '../store/slices/chatSlice';

export const useWebSocket = (userId) => {
  const ws = useRef(null);
  const dispatch = useDispatch();

  useEffect(() => {
    if (!userId) return;

    const connect = () => {
      ws.current = new WebSocket(`ws://localhost:8002/ws/ws/audio/${userId}`);

      ws.current.onopen = () => {
        console.log('WS Connected');
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Gestion des événements backend
        if (data.type === 'ai_response_text') {
           dispatch(setTyping(false));
           dispatch(addMessage({
             id: Date.now(),
             text: data.payload,
             sender: 'ai',
             timestamp: new Date().toISOString()
           }));
        }
        // Ajouter d'autres cas (streaming audio, transcription partielle, etc.)
      };

      ws.current.onclose = () => {
        console.log('WS Disconnected, retrying...');
        setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      if (ws.current) ws.current.close();
    };
  }, [userId, dispatch]);

  const sendMessage = (msg) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(msg));
    }
  };

  return { sendMessage };
};