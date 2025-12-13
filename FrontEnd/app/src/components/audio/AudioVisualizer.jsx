// src/components/audio/AudioVisualizer.jsx

import React, { useEffect, useRef } from 'react';

const AudioVisualizer = ({ stream, isRecording }) => {
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const analyzerRef = useRef(null);
  const requestRef = useRef(null);

  useEffect(() => {
    // 1. Vérification de sécurité CRUCIALE
    if (!stream || !isRecording || stream.getAudioTracks().length === 0) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // 2. Initialisation AudioContext
    if (!audioContextRef.current) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContextRef.current = new AudioContext();
    }
    
    const audioContext = audioContextRef.current;

    // 3. Création Source & Analyseur
    try {
      // Nettoyage préventif
      if (sourceRef.current) sourceRef.current.disconnect();
      
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      sourceRef.current = source;
      analyzerRef.current = analyser;

      // 4. Boucle d'animation
      const draw = () => {
        if (!isRecording) return;
        requestRef.current = requestAnimationFrame(draw);
        
        analyser.getByteFrequencyData(dataArray);
        
        ctx.fillStyle = 'rgb(20, 20, 30)'; // Fond (couleur de votre thème)
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
          barHeight = dataArray[i] / 2;
          
          // Dégradé ou couleur simple
          ctx.fillStyle = `rgb(${barHeight + 100}, 50, 200)`;
          ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          
          x += barWidth + 1;
        }
      };
      
      draw();

    } catch (err) {
      console.error("Erreur visualisation audio:", err);
    }

    // 5. Cleanup
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (sourceRef.current) {
        try {
          sourceRef.current.disconnect();
        } catch (disconnectError) {
          console.error('Audio visualizer disconnect error:', disconnectError);
        }
      }
      // On ne ferme pas le contexte ici pour éviter de casser l'enregistrement parent
    };
  }, [stream, isRecording]);

  // Fallback UI si pas de stream actif
  if (!stream || !isRecording) {
      return <div className="h-full w-full flex items-center justify-center text-gray-600 text-xs">Micro inactif</div>;
  }

  return (
    <canvas 
      ref={canvasRef} 
      width={200} 
      height={60} 
      className="w-full h-full rounded-lg"
    />
  );
};

export default AudioVisualizer;