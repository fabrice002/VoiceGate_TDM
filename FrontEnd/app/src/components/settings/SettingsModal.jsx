// src/components/settings/SettingsModal.jsx

import React, { useState } from 'react';
import { Dialog } from '@headlessui/react';


import { FaTrash, FaUserTimes, FaTimes, FaExclamationTriangle, FaSpinner } from 'react-icons/fa';
import { useDispatch, useSelector } from 'react-redux';
import { clearConversation } from '../../store/slices/chatSlice';
import { logout } from '../../store/slices/authSlice';
import { voiceApi } from '../../services/api';

const SettingsModal = ({ isOpen, onClose }) => {
  // 'menu' | 'confirm_history' | 'confirm_account'
  const [view, setView] = useState('menu'); 
  const [isLoading, setIsLoading] = useState(false);
  
  const { user } = useSelector(state => state.auth);
  const dispatch = useDispatch();

  const resetAndClose = () => {
    setView('menu');
    onClose();
  };

  // --- ACTIONS ---

  const handleDeleteHistory = async () => {
    if (!user?.id) return;
    setIsLoading(true);
    try {
      await voiceApi.deleteUserConversations(user.id);
      dispatch(clearConversation());
      resetAndClose();
      // Optionnel : Ajouter un Toast de succès ici
    } catch (error) {
      console.error("Erreur suppression historique:", error);
      alert("Erreur lors de la suppression.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (!user?.id) return;
    setIsLoading(true);
    try {
      await voiceApi.deleteUserAccount(user.id);
      dispatch(logout()); // Déconnecte et renvoie à l'écran d'accueil
      dispatch(clearConversation());
      resetAndClose();
    } catch (error) {
      console.error("Erreur suppression compte:", error);
      alert("Impossible de supprimer le compte.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onClose={resetAndClose} className="relative z-50">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm" aria-hidden="true" />

      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-md rounded-2xl bg-surface border border-gray-700 p-6 text-white shadow-2xl">
          
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <Dialog.Title className="text-xl font-bold flex items-center gap-2">
              {view === 'menu' ? 'Paramètres' : 'Confirmation requise'}
            </Dialog.Title>
            <button onClick={resetAndClose} className="text-gray-400 hover:text-white">
              <FaTimes />
            </button>
          </div>

          {/* VUE 1 : MENU PRINCIPAL */}
          {view === 'menu' && (
            <div className="space-y-4">
              <div className="p-4 bg-background rounded-lg border border-gray-700 mb-6">
                <p className="text-sm text-gray-400">Compte connecté :</p>
                <p className="text-lg font-semibold text-primary">{user?.username}</p>
              </div>

              <button 
                onClick={() => setView('confirm_history')}
                className="w-full flex items-center justify-between p-4 bg-gray-800 hover:bg-gray-700 rounded-xl transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-700 rounded-lg group-hover:bg-gray-600">
                    <FaTrash className="text-gray-300" />
                  </div>
                  <div className="text-left">
                    <p className="font-medium">Effacer l'historique</p>
                    <p className="text-xs text-gray-400">Supprime toutes les conversations</p>
                  </div>
                </div>
              </button>

              <button 
                onClick={() => setView('confirm_account')}
                className="w-full flex items-center justify-between p-4 bg-red-900/20 hover:bg-red-900/30 border border-red-900/50 rounded-xl transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-red-900/40 rounded-lg text-red-400">
                    <FaUserTimes />
                  </div>
                  <div className="text-left">
                    <p className="font-medium text-red-200">Supprimer mon compte</p>
                    <p className="text-xs text-red-400/70">Action irréversible</p>
                  </div>
                </div>
              </button>
            </div>
          )}

          {/* VUE 2 : CONFIRMATION (Alerte) */}
          {(view === 'confirm_history' || view === 'confirm_account') && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-300">
              <div className="flex flex-col items-center text-center mb-6">
                <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mb-4 text-red-500">
                  <FaExclamationTriangle size={32} />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Êtes-vous absolument sûr ?</h3>
                <p className="text-gray-400 text-sm">
                  {view === 'confirm_history' 
                    ? "Cette action effacera définitivement tous vos échanges avec l'IA. Vous ne pourrez pas les récupérer."
                    : "Cette action supprimera votre compte et vos données vocales de nos serveurs. Cette action est irréversible."}
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setView('menu')}
                  className="flex-1 py-3 px-4 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition"
                  disabled={isLoading}
                >
                  Annuler
                </button>
                <button
                  onClick={view === 'confirm_history' ? handleDeleteHistory : handleDeleteAccount}
                  disabled={isLoading}
                  className="flex-1 py-3 px-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-bold transition flex justify-center items-center gap-2"
                >
                  {isLoading ? <FaSpinner className="animate-spin" /> : "Confirmer"}
                </button>
              </div>
            </div>
          )}

        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default SettingsModal;