import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  messages: [], // On commence vide
  isRecording: false,
  isTyping: false,
  conversationId: null, // On stocke l'ID de la conversation active
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    addMessage: (state, action) => {
      state.messages.push(action.payload);
    },
    // --- NOUVELLE ACTION ---
    setHistory: (state, action) => {
      // Remplace tout l'historique actuel
      state.messages = action.payload.messages;
      state.conversationId = action.payload.conversationId;
    },
    setTyping: (state, action) => {
      state.isTyping = action.payload;
    },
    clearConversation: (state) => {
      state.messages = [];
      state.conversationId = null;
    },
  },
});

export const { addMessage, setHistory, setTyping, clearConversation } =
  chatSlice.actions;
export default chatSlice.reducer;
