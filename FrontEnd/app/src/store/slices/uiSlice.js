import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  isModalOpen: false,
  theme: "dark", // 'dark' | 'light'
};

const uiSlice = createSlice({
  name: "ui",
  initialState,
  reducers: {
    openModal: (state) => {
      state.isModalOpen = true;
    },
    closeModal: (state) => {
      state.isModalOpen = false;
    },
    toggleTheme: (state) => {
      state.theme = state.theme === "dark" ? "light" : "dark";
    },
  },
});

export const { openModal, closeModal, toggleTheme } = uiSlice.actions;
export default uiSlice.reducer;
