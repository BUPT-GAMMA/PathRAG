import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';
import { userAPI } from '../services/api';

// Define theme color schemes
export const themes = {
  blue: {
    name: 'Ocean Blue',
    primary: '#1E88E5',
    'primary-rgb': '30, 136, 229',
    secondary: '#0D47A1',
    accent: '#90CAF9',
    background: '#F5F9FF',
    contentBg: '#FFFFFF',
    sidebar: '#1E3A5F',
    sidebarHover: '#0D47A1',  /* Same as navActive for consistency */
    navActive: '#0D47A1',
    text: '#2A3747',
    textLight: '#FFFFFF',
    textMuted: '#A4B8D3',
    border: '#E2E8F0',
    cardBg: '#FFFFFF',
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    info: '#2196F3',
    headerBg: '#1E3A5F',
    headerText: '#FFFFFF',
    panelBg: '#FFFFFF',
    inputBg: '#F8FAFC',
    dropdownBg: '#FFFFFF',
    dropdownHover: '#F0F7FF',
    shadow: 'rgba(0, 0, 0, 0.1)',
  },
  red: {
    name: 'Ruby Red',
    primary: '#E53935',
    'primary-rgb': '229, 57, 53',
    secondary: '#B71C1C',
    accent: '#FFCDD2',
    background: '#FFF5F5',
    contentBg: '#FFFFFF',
    sidebar: '#8B2E2E',
    sidebarHover: '#B71C1C',  /* Same as navActive for consistency */
    navActive: '#B71C1C',
    text: '#3A2A2A',
    textLight: '#FFFFFF',
    textMuted: '#D4A6A6',
    border: '#EFE0E0',
    cardBg: '#FFFFFF',
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    info: '#2196F3',
    headerBg: '#8B2E2E',
    headerText: '#FFFFFF',
    panelBg: '#FFFFFF',
    inputBg: '#FFF8F8',
    dropdownBg: '#FFFFFF',
    dropdownHover: '#FFF0F0',
    shadow: 'rgba(0, 0, 0, 0.1)',
  },
  violet: {
    name: 'Royal Purple',
    primary: '#7E57C2',
    'primary-rgb': '126, 87, 194',
    secondary: '#4527A0',
    accent: '#D1C4E9',
    background: '#F8F5FF',
    contentBg: '#FFFFFF',
    sidebar: '#503D8B',  /* Matches the screenshot purple */
    sidebarHover: '#3A2D66',  /* Same as navActive for consistency */
    navActive: '#3A2D66',  /* Darker purple for active items */
    text: '#2D2A3A',
    textLight: '#FFFFFF',
    textMuted: '#B9B0D6',
    border: '#E6E0F0',
    cardBg: '#FFFFFF',
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    info: '#2196F3',
    headerBg: '#503D8B',
    headerText: '#FFFFFF',
    panelBg: '#FFFFFF',
    inputBg: '#F9F7FC',
    dropdownBg: '#FFFFFF',
    dropdownHover: '#F3EEFF',
    shadow: 'rgba(0, 0, 0, 0.1)',
  }
};

// Create the context
const ThemeContext = createContext();

// Create the provider component
export const ThemeProvider = ({ children }) => {
  const { currentUser, isAuthenticated } = useAuth();

  // Get theme from user preferences, localStorage, or use default
  const [currentTheme, setCurrentTheme] = useState(() => {
    // First try to get from localStorage
    const savedTheme = localStorage.getItem('theme');
    return savedTheme && themes[savedTheme] ? savedTheme : 'blue';
  });

  // Update theme when user logs in
  useEffect(() => {
    if (isAuthenticated && currentUser?.theme && themes[currentUser.theme]) {
      setCurrentTheme(currentUser.theme);
    }
  }, [isAuthenticated, currentUser]);

  // Apply theme to document root for CSS variables
  useEffect(() => {
    const theme = themes[currentTheme];
    const root = document.documentElement;

    // Set CSS variables
    Object.entries(theme).forEach(([key, value]) => {
      if (key !== 'name') {
        root.style.setProperty(`--${key}`, value);
      }
    });

    // Save to localStorage
    localStorage.setItem('theme', currentTheme);
  }, [currentTheme]);

  // Change theme function
  const changeTheme = async (themeName) => {
    if (themes[themeName]) {
      setCurrentTheme(themeName);

      // If user is authenticated, save theme preference to backend
      if (isAuthenticated) {
        try {
          // Update user theme preference in the backend
          await userAPI.updateTheme(themeName);
        } catch (error) {
          console.error('Failed to save theme preference:', error);
        }
      }
    }
  };

  return (
    <ThemeContext.Provider value={{
      currentTheme,
      changeTheme,
      themes,
      theme: themes[currentTheme]
    }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook to use the theme context
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
