import React, { useState, useEffect } from 'react';
import { Container, Header, Content, Sidebar, Sidenav, Dropdown, Nav } from 'rsuite';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { chatAPI } from '../services/api';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTachometerAlt,
  faComments,
  faProjectDiagram,
  faFileAlt,
  faSignOutAlt,
  faUser,
  faPalette,
  faBars,
  faCog,
  faPlus
} from '@fortawesome/free-solid-svg-icons';

const Layout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { currentUser, logout, isAuthenticated } = useAuth();
  const { currentTheme, changeTheme, themes } = useTheme();
  const [expanded, setExpanded] = useState(() => {
    // Get sidebar state from localStorage or default to true
    const savedState = localStorage.getItem('sidebarExpanded');
    return savedState !== null ? JSON.parse(savedState) : true;
  });
  const [mobile, setMobile] = useState(false);
  const [recentChats, setRecentChats] = useState([]);
  const [isLoadingChats, setIsLoadingChats] = useState(false);

  // Check if mobile view on window resize
  useEffect(() => {
    const checkMobile = () => {
      const isMobile = window.innerWidth < 768;
      setMobile(isMobile);
      if (isMobile && expanded) {
        setExpanded(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);

    return () => {
      window.removeEventListener('resize', checkMobile);
    };
  }, [expanded]);

  // Save sidebar state to localStorage
  useEffect(() => {
    localStorage.setItem('sidebarExpanded', JSON.stringify(expanded));
  }, [expanded]);

  // Fetch recent chat threads
  useEffect(() => {
    const fetchRecentThreads = async () => {
      if (isAuthenticated) {
        try {
          setIsLoadingChats(true);
          const response = await chatAPI.getRecentThreads();
          if (response && response.data && response.data.threads) {
            setRecentChats(response.data.threads);
          }
        } catch (error) {
          // Silently handle the error - don't show error in console
          setRecentChats([]); // Set empty array on error
        } finally {
          setIsLoadingChats(false);
        }
      }
    };

    fetchRecentThreads();

    // Re-fetch threads when authentication status changes
  }, [isAuthenticated]);

  const handleSelect = (eventKey) => {
    if (eventKey === 'logout') {
      logout();
      navigate('/login');
    } else {
      navigate(eventKey);
    }

    // Auto-collapse sidebar on mobile after navigation
    if (mobile) {
      setExpanded(false);
    }
  };

  const toggleSidebar = () => {
    setExpanded(!expanded);
  };

  const handleThreadSelect = (threadUuid) => {
    // Navigate to the specific thread
    navigate(`/chat/${threadUuid}`);

    // Auto-collapse sidebar on mobile after selection
    if (mobile) {
      setExpanded(false);
    }
  };

  // Handle theme change
  const handleThemeChange = (themeName) => {
    changeTheme(themeName);
    // TODO: Add API call to save theme preference to user profile
  };

  return (
    <Container>
      {/* Sidebar toggle button - visible on all screen sizes */}
      <div
        className="sidebar-toggle"
        onClick={toggleSidebar}
        style={{
          top: 10,
          left: expanded ? (mobile ? 220 : 240) : 10,
          transform: expanded ? 'rotate(0deg)' : 'rotate(180deg)'
        }}
      >
        <FontAwesomeIcon icon={faBars} />
      </div>

      {/* Responsive Sidebar */}
      <Sidebar
        style={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--sidebar)',
          color: 'var(--textLight)',
          width: expanded ? (mobile ? '240px' : '260px') : '0px',
          transition: 'all 0.3s ease',
          position: 'fixed',
          height: '100vh',
          zIndex: 999,
          overflow: expanded ? 'auto' : 'hidden',
          boxShadow: expanded ? '2px 0 10px rgba(0, 0, 0, 0.2)' : 'none',
          transform: expanded ? 'translateX(0)' : 'translateX(-100%)'
        }}
        width={expanded ? (mobile ? 240 : 260) : 0}
        collapsible
      >
        <Sidenav.Header>
          <div style={{
            padding: '18px 16px',
            height: 56,
            color: 'var(--textLight)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <span>PathRAG</span>
          </div>
        </Sidenav.Header>

        <Sidenav expanded appearance="subtle" defaultOpenKeys={['chats', 'documents']} style={{ color: 'white' }}>
          <Sidenav.Body style={{ color: 'white' }}>
            <Nav activeKey={location.pathname} onSelect={handleSelect} className="custom-nav">
              <Nav.Item
                eventKey="/"
                icon={<FontAwesomeIcon icon={faTachometerAlt} className="nav-icon" />}
              >
                Dashboard
              </Nav.Item>

              <Nav.Item
                eventKey="/chat"
                icon={<FontAwesomeIcon icon={faComments} className="nav-icon" />}
                onClick={() => {
                  // Check if there's a thread UUID in localStorage
                  const threadId = localStorage.getItem('current_thread_uuid');
                  if (threadId) {
                    // Navigate to the existing thread
                    navigate(`/chat/${threadId}`);
                  } else {
                    // Create a new thread
                    (async () => {
                      try {
                        const response = await chatAPI.createThread("New Chat");
                        if (response && response.data && response.data.uuid) {
                          // Save the thread UUID to localStorage
                          localStorage.setItem('current_thread_uuid', response.data.uuid);
                          // Navigate to the new thread
                          navigate(`/chat/${response.data.uuid}`);
                        }
                      } catch (error) {
                        console.error("Failed to create new chat thread", error);
                        // Fallback to the base chat route
                        navigate('/chat');
                      }
                    })();
                  }
                }}
              >
                Chats
              </Nav.Item>

              <Nav.Item
                eventKey="/knowledge-graph"
                icon={<FontAwesomeIcon icon={faProjectDiagram} className="nav-icon" />}
              >
                Knowledge Graph
              </Nav.Item>

              <Nav.Item
                eventKey="/documents"
                icon={<FontAwesomeIcon icon={faFileAlt} className="nav-icon" />}
              >
                Documents
              </Nav.Item>
            </Nav>

            {/* Chat Section */}
            <div style={{
              marginTop: 30,
              padding: '0 16px',
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              paddingTop: 16
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 12
              }}>
                <div style={{
                  color: 'var(--textMuted)',
                  fontSize: '0.85rem',
                  fontWeight: 'bold',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Recent Chats
                </div>
                <button
                  onClick={async () => {
                    try {
                      // Create a new thread
                      const response = await chatAPI.createThread("New Chat");
                      if (response && response.data && response.data.uuid) {
                        // Save the thread UUID to localStorage
                        localStorage.setItem('current_thread_uuid', response.data.uuid);
                        // Navigate to the thread
                        handleThreadSelect(response.data.uuid);
                      }
                    } catch (error) {
                      console.error("Failed to create new chat thread", error);
                    }
                  }}
                  style={{
                    backgroundColor: 'var(--primary)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    fontSize: '0.8rem',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <FontAwesomeIcon icon={faPlus} size="xs" />
                  New Chat
                </button>
              </div>

                {isLoadingChats ? (
                  <div style={{ color: 'var(--textMuted)', fontSize: '0.9rem', padding: '8px 0' }}>
                    Loading...
                  </div>
                ) : (
                  recentChats.map(thread => (
                    <div
                      key={thread.id}
                      onClick={() => handleThreadSelect(thread.uuid)}
                      style={{
                        padding: '10px 12px',
                        borderRadius: '4px',
                        marginBottom: 8,
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        display: 'flex',
                        flexDirection: 'column'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = 'var(--sidebarHover)';
                        e.currentTarget.style.transform = 'translateX(2px)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                        e.currentTarget.style.transform = 'translateX(0)';
                      }}
                    >
                      <div style={{
                        color: 'white',
                        fontWeight: 'medium',
                        fontSize: '0.9rem',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {thread.title || 'New Chat'}
                      </div>
                      <div style={{
                        color: 'var(--textMuted)',
                        fontSize: '0.75rem',
                        marginTop: 2
                      }}>
                        {thread.updated_at ? new Date(thread.updated_at).toLocaleDateString() : 'Recent'}
                      </div>
                    </div>
                  ))
                )}
              </div>
          </Sidenav.Body>
        </Sidenav>

        <div style={{ flex: 1 }}></div>
      </Sidebar>

      {/* Main content container */}
      <Container style={{
        marginLeft: expanded && !mobile ? '260px' : 0,
        transition: 'all 0.3s ease',
        width: expanded && !mobile ? 'calc(100% - 260px)' : '100%'
      }}>
        <Header style={{
          position: 'fixed',
          top: 0,
          width: expanded && !mobile ? 'calc(100% - 260px)' : '100%',
          right: 0,
          zIndex: 900,
          transition: 'all 0.3s ease'
        }}>
          <div style={{
            padding: '0 20px',
            height: 56,
            lineHeight: '56px',
            backgroundColor: 'white',
            color: 'var(--text)',
            borderBottom: '1px solid #E8EDF3',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)'
          }}>
            <div style={{
              marginLeft: expanded ? 0 : 50,
              fontWeight: 600,
              fontSize: '1.1rem',
              color: '#333'
            }}>
              {location.pathname === '/' && 'Dashboard'}
              {location.pathname === '/chat' && 'Chats'}
              {location.pathname === '/knowledge-graph' && 'Knowledge Graph'}
              {location.pathname === '/documents' && 'Documents'}
            </div>

            <div style={{ display: 'flex', alignItems: 'center' }}>
              <Dropdown
                title={
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: '1px solid #E8EDF3',
                    backgroundColor: '#F8FAFC',
                    color: '#555'
                  }}>
                    <FontAwesomeIcon icon={faPalette} style={{
                      marginRight: 8,
                      color: 'var(--primary)'
                    }} />
                    {!mobile && 'Theme'}
                  </div>
                }
                placement="bottomEnd"
              >
                {Object.keys(themes).map(themeName => (
                  <Dropdown.Item
                    key={themeName}
                    active={currentTheme === themeName}
                    onClick={() => handleThemeChange(themeName)}
                  >
                    <div className="theme-option">
                      <div
                        className="theme-color-preview"
                        style={{
                          backgroundColor: themes[themeName].primary,
                          width: 20,
                          height: 20,
                          boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                        }}
                      ></div>
                      {themes[themeName].name}
                    </div>
                  </Dropdown.Item>
                ))}
              </Dropdown>

              {/* User Menu */}
              <Dropdown
                title={
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '6px 12px',
                    marginLeft: 10,
                    borderRadius: '4px',
                    border: '1px solid #E8EDF3',
                    backgroundColor: '#F8FAFC',
                    color: '#555'
                  }}>
                    <FontAwesomeIcon icon={faUser} style={{
                      marginRight: 8,
                      color: 'var(--primary)'
                    }} />
                    {!mobile && currentUser?.username}
                  </div>
                }
                placement="bottomEnd"
              >
                <Dropdown.Item icon={<FontAwesomeIcon icon={faCog} />}>
                  Settings
                </Dropdown.Item>
                <Dropdown.Item
                  icon={<FontAwesomeIcon icon={faSignOutAlt} />}
                  onClick={() => {
                    logout();
                    navigate('/login');
                  }}
                >
                  Logout
                </Dropdown.Item>
              </Dropdown>
            </div>
          </div>
        </Header>

        <Content style={{
          padding: mobile ? 10 : 20,
          backgroundColor: '#F5F7FA',
          minHeight: 'calc(100vh - 56px)',
          marginTop: 56, /* Add margin to account for fixed header */
          overflow: 'auto'
        }}>
          {children}
        </Content>
      </Container>
    </Container>
  );
};

export default Layout;
