import React, { useEffect } from 'react';
import { Avatar, IconButton } from 'rsuite';
import MoreIcon from '@rsuite/icons/More';
import { useAuth } from '../../context/AuthContext';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import mermaid from 'mermaid';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: 'monospace'
});

const ChatMessage = ({ message, isUser }) => {
  const { currentUser } = useAuth();

  useEffect(() => {
    // Process any mermaid diagrams after component mounts
    setTimeout(() => {
      try {
        mermaid.contentLoaded();
      } catch (error) {
        console.error('Mermaid processing error:', error);
      }
    }, 0);
  }, [message]);

  // Function to process mermaid code blocks
  const processMermaidBlocks = (content) => {
    if (typeof content !== 'string') return '';

    const mermaidRegex = /```mermaid\n([\s\S]*?)```/g;
    let lastIndex = 0;
    const parts = [];
    let match;

    while ((match = mermaidRegex.exec(content)) !== null) {
      // Add text before the mermaid block
      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }

      // Add mermaid diagram with unique ID
      const diagramId = `mermaid-diagram-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      parts.push(
        `<div class="mermaid-diagram">
          <div class="mermaid" id="${diagramId}">
            ${match[1]}
          </div>
        </div>`
      );

      lastIndex = match.index + match[0].length;
    }

    // Add any remaining text
    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return parts.join('');
  };

  // Process content for mermaid diagrams if it's a bot message
  const processedContent = !isUser ? processMermaidBlocks(message) : message;

  return (
    <div className={`chat-message ${isUser ? 'user-message' : 'ai-message'}`}>
      {!isUser && (
        <div className="message-avatar">
          <Avatar circle src="/logo192.png">
            AI
          </Avatar>
        </div>
      )}

      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">{isUser ? currentUser?.username : 'Enterprise KG AI'}</span>
          <span className="message-time">{new Date().toLocaleTimeString()}</span>
        </div>

        <div className="message-bubble">
          {!isUser ? (
            <div className="markdown-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {processedContent}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="text-content">{message}</div>
          )}
        </div>

        <div className="message-actions">
          <IconButton icon={<MoreIcon />} circle size="xs" />
        </div>
      </div>

      {isUser && (
        <div className="message-avatar">
          <Avatar circle>
            {currentUser?.username?.charAt(0).toUpperCase()}
          </Avatar>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
