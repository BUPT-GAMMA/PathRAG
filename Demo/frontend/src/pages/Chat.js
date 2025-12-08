import React, { useState, useEffect, useRef } from 'react';
import { Container, Panel, InputGroup, Button, Loader, SelectPicker, Divider } from 'rsuite';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPaperPlane, faSpinner } from '@fortawesome/free-solid-svg-icons';
import { chatAPI } from '../services/api';
import ChatMessage from '../components/chat/ChatMessage';
import Layout from '../components/Layout';
import '../styles/chat.css';

// Search context options
const contextOptions = [
  { label: 'All Documents', value: 'all' },
  { label: 'Recent Documents', value: 'recent' },
  { label: 'Selected Documents', value: 'selected' },
  { label: 'Knowledge Graph', value: 'knowledge_graph' }
];

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [searchContext, setSearchContext] = useState('all');
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      content: input,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Send message to API with search context
      const response = await chatAPI.createChat({
        message: input,
        context: searchContext
      });

      // Add bot response to chat
      if (response && response.data) {
        const botMessage = {
          id: Date.now() + 1,
          content: response.data.response || 'Sorry, I could not process your request.',
          sender: 'bot',
          timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        content: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'bot',
        error: true,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Mock response for testing
  const handleMockResponse = () => {
    const mockResponses = [
      `Here's a simple markdown example:

# Heading 1
## Heading 2

- List item 1
- List item 2

**Bold text** and *italic text*

\`\`\`python
def hello_world():
    print("Hello, world!")
\`\`\`
      `,
      `Here's a mermaid diagram example:

\`\`\`mermaid
graph TD
    A[Client] --> B[Load Balancer]
    B --> C[Server1]
    B --> D[Server2]
\`\`\`

And some more text after the diagram.
      `,
      `Let me explain the architecture:

\`\`\`mermaid
sequenceDiagram
    participant User
    participant API
    participant Database

    User->>API: Request data
    API->>Database: Query
    Database-->>API: Return results
    API-->>User: Response
\`\`\`

This shows the basic flow of information in our system.
      `
    ];

    const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];

    const botMessage = {
      id: Date.now() + 1,
      content: randomResponse,
      sender: 'bot',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, botMessage]);
  };

  return (
    <Layout>
      <Container className="chat-container">
        <Panel bordered className="chat-panel">
          <div className="chat-header">
            <h2>Chat</h2>
            <SelectPicker
              data={contextOptions}
              value={searchContext}
              onChange={setSearchContext}
              cleanable={false}
              searchable={false}
              className="context-selector"
              placeholder="Search Context"
            />
          </div>

          <Divider />

          <div className="messages-container" ref={chatContainerRef}>
            {messages.length === 0 ? (
              <div className="empty-chat">
                <p>No messages yet. Start a conversation!</p>
              </div>
            ) : (
              messages.map(message => (
                <ChatMessage
                  key={message.id}
                  message={message.content}
                  isUser={message.sender === 'user'}
                />
              ))
            )}

            {isLoading && (
              <div className="loading-message">
                <Loader size="md" content="Thinking..." vertical />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-container">
            <InputGroup inside>
              <input
                className="rs-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                disabled={isLoading}
              />
              <InputGroup.Button
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
                className="send-button"
              >
                {isLoading ? (
                  <FontAwesomeIcon icon={faSpinner} spin />
                ) : (
                  <FontAwesomeIcon icon={faPaperPlane} />
                )}
              </InputGroup.Button>
            </InputGroup>

            {/* Test button for mock responses - remove in production */}
            <Button
              appearance="link"
              onClick={handleMockResponse}
              style={{ marginTop: 10 }}
            >
              Test Response
            </Button>
          </div>
        </Panel>
      </Container>
    </Layout>
  );
};

export default Chat;
