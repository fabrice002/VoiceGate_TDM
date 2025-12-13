import axios from "axios";

// Configuration centralisée
const API_URL = "http://localhost:8002/api";

// Instance Axios optimisée
const api = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30s timeout pour les fichiers audio lourds
  headers: {
    "Content-Type": "application/json",
  },
});

// Intercepteur de réponse (Senior touch: Gestion globale des erreurs)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Log l'erreur proprement pour le debugging
    console.error("API Error:", error.response?.data || error.message);

    // On pourrait ajouter ici une logique de redirection si 401 Unauthorized
    // if (error.response?.status === 401) { ... }

    return Promise.reject(error);
  }
);

export const voiceApi = {
  // ==========================================
  // 👤 GESTION UTILISATEURS
  // ==========================================

  /**
   * Crée un nouvel utilisateur
   * @param {Object} userData - { username: string, email?: string }
   */
  registerUser: (userData) => api.post("/users/", userData),

  /**
   * Récupère les infos d'un utilisateur
   * @param {string} username
   */
  getUser: (username) => api.get(`/users/${username}`),

  // ==========================================
  // 🎙️ GESTION EMPREINTE VOCALE (Enrollment)
  // ==========================================

  /**
   * Enregistre l'empreinte vocale (Nécessite FormData)
   * @param {FormData} formData - Doit contenir 'user_id' et 'file'
   */
  registerVoice: (formData) =>
    api.post("/voice/register", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),

  /**
   * Identifie un utilisateur par sa voix
   * @param {FormData} formData - Doit contenir 'file'
   */
  identifyVoice: (formData) =>
    api.post("/voice/identify", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),

  // ==========================================
  // 💬 CONVERSATION & INTELLIGENCE
  // ==========================================

  /**
   * Envoie une question vocale au backend
   * @param {FormData} formData - 'user_id', 'audio_file', 'language'
   */
  sendVoiceAsk: (formData) =>
    api.post("/voice-conversation/voice-ask", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),

  /**
   * Envoie une question texte
   * @param {Object} payload - { user_id, text, conversation_id? }
   */
  sendTextAsk: (payload) => api.post("/voice-conversation/text-ask", payload),

  /**
   * Récupère l'historique des conversations
   */
  getVoiceHistory: (userId) =>
    api.get(`/voice-conversation/conversations/${userId}/voice`),

  getLastConversationMessages: (userId) =>
    api.get(`/voice-conversation/conversations/${userId}/last/messages`),

  // --- GESTION DES SUPPRESSIONS ---

  // Supprimer tout l'historique d'un utilisateur
  deleteUserConversations: (userId) =>
    api.delete(`/voice-conversation/conversations/${userId}`),

  // Supprimer le compte utilisateur complet
  deleteUserAccount: (userId) => api.delete(`/users/${userId}`),

  // ==========================================
  // 🛠️ UTILITAIRES
  // ==========================================

  /**
   * Transcrit un fichier audio sans contexte conversationnel
   */
  transcribe: (formData) =>
    api.post("/transcription/transcribe", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),

  /**
   * Vérifie l'état de la base de données (Health Check)
   */
  checkHealth: () => api.get("/db_check/stats"), // Basé sur ton snippet code.py
};

export default api;
