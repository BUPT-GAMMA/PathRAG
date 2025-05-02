import React, { useState, useEffect, useRef } from 'react';
import { Loader } from 'rsuite';
import { useParams, useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import ChatMessage from '../components/chat/ChatMessage';
import ChatInput from '../components/chat/ChatInput';
import { chatAPI } from '../services/api';
import { saveThreadUUID, getThreadUUID } from '../utils/threadStorage';
import '../styles/chat.css';

const ChatPage = () => {
  const { threadId } = useParams();
  const navigate = useNavigate();
  const [currentThreadId, setCurrentThreadId] = useState(threadId);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [searchContext, setSearchContext] = useState('hybrid');
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Check for thread ID or create a new one
  useEffect(() => {
    const initializeThread = async () => {
      // If we have a thread ID in the URL, use it
      if (threadId) {
        setCurrentThreadId(threadId);
        saveThreadUUID(threadId);
        return;
      }

      // If we have a thread ID in localStorage, redirect to it
      const storedThreadId = getThreadUUID();
      if (storedThreadId) {
        navigate(`/chat/${storedThreadId}`, { replace: true });
        return;
      }

      // Otherwise, create a new thread
      try {
        const response = await chatAPI.createThread("New Chat");
        if (response && response.data && response.data.uuid) {
          const newThreadId = response.data.uuid;
          saveThreadUUID(newThreadId);
          navigate(`/chat/${newThreadId}`, { replace: true });
        }
      } catch (error) {
        console.error("Failed to create new chat thread", error);
      }
    };

    initializeThread();
  }, [threadId, navigate]);

  // Fetch chat history for the current thread
  useEffect(() => {
    const fetchThreadChats = async () => {
      if (!currentThreadId) return;

      setLoading(true);

      try {
        const response = await chatAPI.getThread(currentThreadId);

        if (response && response.data && response.data.chats) {
          // Format messages from the thread
          const formattedMessages = response.data.chats.map((chat) => ({
            id: `${chat.id}`,
            content: chat.message,
            isUser: chat.role === 'user',
          }));

          setMessages(formattedMessages);
        }
      } catch (error) {
        console.error('Error fetching thread chats:', error);
      } finally {
        setLoading(false);
      }
    };

    if (currentThreadId) {
      fetchThreadChats();
    }
  }, [currentThreadId]);

  // Function to scroll to bottom
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
      });
    }

    // Alternative method using the container
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, sending]);

  // Send message
  const handleSendMessage = async (message) => {
    if (!message.trim() || sending || !currentThreadId) return;

    // Add user message to state
    const userMessage = {
      id: `temp-${Date.now()}-user`,
      content: message,
      isUser: true,
    };

    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setSending(true);

    try {
      // Send message to API with thread UUID and search mode
      console.log(`Sending message to thread ${currentThreadId} with search mode: ${searchContext}`);
      const response = await chatAPI.sendChatToThread(currentThreadId, searchContext, message);

      if (response && response.data) {
        // Refresh the thread to get both user message and AI response
        const threadResponse = await chatAPI.getThread(currentThreadId);

        if (threadResponse && threadResponse.data && threadResponse.data.chats) {
          // Format messages from the thread
          const formattedMessages = threadResponse.data.chats.map((chat) => ({
            id: `${chat.id}`,
            content: chat.message,
            isUser: chat.role === 'user',
          }));

          setMessages(formattedMessages);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message
      const errorMessage = {
        id: `error-${Date.now()}`,
        content: 'Sorry, there was an error processing your request.',
        isUser: false,
      };

      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setSending(false);
    }
  };


  return (
    <Layout>
      <div className="chat-page">
        <div className="chat-content">
          <div className="messages-container" id="chat-messages" ref={messagesContainerRef}>
            {loading ? (
              <div className="loading-message">
                <Loader content="Loading chat history..." vertical />
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-chat">
                <p>No messages yet. Start a conversation!</p>
              </div>
            ) : (
              <div className="conversation-thread">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message-wrapper ${message.isUser ? 'user-message-wrapper' : 'ai-message-wrapper'}`}
                  >
                    <ChatMessage
                      message={message.content}
                      isUser={message.isUser}
                    />
                  </div>
                ))}

                {sending && (
                  <div className="message-wrapper ai-message-wrapper">
                    <div className="loading-message">
                      <Loader size="md" content="Thinking..." vertical />
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} style={{ height: 1, clear: 'both' }} />
              </div>
            )}
          </div>
        </div>

        <div className="chat-input-container">
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={sending}
            searchContext={searchContext}
            setSearchContext={setSearchContext}
          />
        </div>
      </div>
    </Layout>
  );
};

export default ChatPage;
