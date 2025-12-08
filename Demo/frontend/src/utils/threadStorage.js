/**
 * Utility functions for managing chat thread UUIDs in localStorage
 */

const THREAD_KEY = 'current_thread_uuid';

/**
 * Save a thread UUID to localStorage
 * @param {string} uuid - The thread UUID to save
 */
export const saveThreadUUID = (uuid) => {
  if (uuid) {
    localStorage.setItem(THREAD_KEY, uuid);
  }
};

/**
 * Get the current thread UUID from localStorage
 * @returns {string|null} The thread UUID or null if not found
 */
export const getThreadUUID = () => {
  return localStorage.getItem(THREAD_KEY);
};

/**
 * Clear the current thread UUID from localStorage
 */
export const clearThreadUUID = () => {
  localStorage.removeItem(THREAD_KEY);
};

/**
 * Check if a thread UUID exists in localStorage
 * @returns {boolean} True if a thread UUID exists, false otherwise
 */
export const hasThreadUUID = () => {
  return !!getThreadUUID();
};

export default {
  saveThreadUUID,
  getThreadUUID,
  clearThreadUUID,
  hasThreadUUID
};
