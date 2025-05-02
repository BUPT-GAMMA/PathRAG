import React, { createContext, useState, useContext, useEffect } from 'react';
import { authAPI } from '../services/api';
import jwt_decode from 'jwt-decode';

// Create auth context
const AuthContext = createContext();

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Check if token is valid
  const isTokenValid = (token) => {
    if (!token) return false;

    try {
      const decoded = jwt_decode(token);
      const currentTime = Date.now() / 1000;

      return decoded.exp > currentTime;
    } catch (error) {
      return false;
    }
  };

  // Load user from token
  const loadUser = async () => {
    setIsLoading(true);

    const token = localStorage.getItem('token');

    if (token && isTokenValid(token)) {
      try {
        const response = await authAPI.getCurrentUser();
        setCurrentUser(response.data);
        setIsAuthenticated(true);
      } catch (error) {
        localStorage.removeItem('token');
        setCurrentUser(null);
        setIsAuthenticated(false);
        setError('Session expired. Please login again.');
      }
    } else {
      localStorage.removeItem('token');
      setCurrentUser(null);
      setIsAuthenticated(false);
    }

    setIsLoading(false);
  };

  // Login user
  const login = async (username, password) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await authAPI.login(username, password);
      const { access_token } = response.data;

      localStorage.setItem('token', access_token);

      await loadUser();

      return true;
    } catch (error) {
      setError(error.response?.data?.detail || 'Login failed. Please try again.');
      setIsLoading(false);
      return false;
    }
  };

  // Register user
  const register = async (userData) => {
    setIsLoading(true);
    setError(null);

    try {
      await authAPI.register(userData);

      // Login after successful registration
      const loginSuccess = await login(userData.username, userData.password);

      return loginSuccess;
    } catch (error) {
      setError(error.response?.data?.detail || 'Registration failed. Please try again.');
      setIsLoading(false);
      return false;
    }
  };

  // Logout user
  const logout = () => {
    localStorage.removeItem('token');
    setCurrentUser(null);
    setIsAuthenticated(false);
  };

  // Load user on initial render
  useEffect(() => {
    loadUser();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Context value
  const value = {
    currentUser,
    isAuthenticated,
    isLoading,
    error,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
};
