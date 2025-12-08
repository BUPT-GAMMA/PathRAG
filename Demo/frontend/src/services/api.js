import axios from 'axios';

// Get API URL from environment variable or use default
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle 401 Unauthorized errors
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    return api.post('/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  register: (userData) => api.post('/register', userData),
  getCurrentUser: () => api.get('/users/me'),
};

// User API
export const userAPI = {
  updateTheme: (theme) => api.post('/users/theme', { theme }),
  getSettings: () => api.get('/users/settings'),
  updateSettings: (settings) => api.put('/users/settings', settings),
};

// Chat API
export const chatAPI = {
  // Thread endpoints
  getThreads: () => api.get('/chats/threads'),
  getRecentThreads: () => api.get('/chats/recent'),
  createThread: (title) => api.post('/chats/threads', { title }),
  getThread: (uuid) => api.get(`/chats/threads/${uuid}`),
  updateThread: (uuid, title) => api.put(`/chats/threads/${uuid}`, { title }),
  deleteThread: (uuid) => api.delete(`/chats/threads/${uuid}`),

  // Chat endpoints
  getChats: () => api.get('/chats'),
  createChat: (data) => {
    // Handle both string and object formats
    if (typeof data === 'string') {
      return api.post('/chats', { message: data });
    }
    return api.post('/chats', data);
  },
  sendChatToThread: (threadUuid, searchContext, message) => {
    return api.post(`/chats/chat/${threadUuid}`, {
      message,
      search_context: searchContext,
      thread_uuid: threadUuid
    });
  },
  getChat: (id) => api.get(`/chats/${id}`),
};

// Document API
export const documentAPI = {
  getDocuments: () => api.get('/documents'),
  uploadDocument: (formData, onUploadProgress) => {
    // For backward compatibility, handle both the new /upload endpoint and the old endpoint
    return api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    }).catch(error => {
      // If the old endpoint fails, try the new one
      throw error;
    });
  },
  getDocument: (id) => api.get(`/documents/${id}`),
  getDocumentStatus: (id) => api.get(`/documents/${id}/status`),
  deleteDocument: (id) => api.delete(`/documents/${id}`),
  reloadDocuments: () => api.post('/documents/reload'),
};

// Knowledge Graph API
export const knowledgeGraphAPI = {
  getGraph: () => api.get('/knowledge-graph'),
  queryGraph: (query) => api.post('/knowledge-graph/query', { query }),
};

export default api;
