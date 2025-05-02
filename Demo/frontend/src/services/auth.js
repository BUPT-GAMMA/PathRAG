// Authentication service

// Get the authentication token from localStorage
export const getToken = () => {
  return localStorage.getItem('token');
};

// Set the authentication token in localStorage
export const setToken = (token) => {
  localStorage.setItem('token', token);
};

// Remove the authentication token from localStorage
export const removeToken = () => {
  localStorage.removeItem('token');
};

// Check if the user is authenticated
export const isAuthenticated = () => {
  return !!getToken();
};

// Parse JWT token to get user information
export const parseToken = (token) => {
  if (!token) return null;
  
  try {
    // JWT tokens are in the format: header.payload.signature
    // We only need the payload part
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    
    return JSON.parse(jsonPayload);
  } catch (error) {
    console.error('Error parsing token:', error);
    return null;
  }
};

// Get user information from the token
export const getUserInfo = () => {
  const token = getToken();
  if (!token) return null;
  
  return parseToken(token);
};
