import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  user: null,
  isAuthenticated: false,
  token: null,
  voiceEmbedding: null,
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginSuccess: (state, action) => {
      state.isAuthenticated = true;
      state.user = action.payload;
    },
    logout: (state) => {
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
    },
    setVoiceEmbedding: (state, action) => {
      state.voiceEmbedding = action.payload;
    },
  },
});

export const { loginSuccess, logout, setVoiceEmbedding } = authSlice.actions;
export default authSlice.reducer;
