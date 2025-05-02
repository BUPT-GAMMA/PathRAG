import React, { useState } from 'react';
import { Input, InputGroup, Loader, IconButton, Tooltip, Whisper, SelectPicker } from 'rsuite';
import SendIcon from '@rsuite/icons/Send';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faMicrophone,
  faPaperclip
} from '@fortawesome/free-solid-svg-icons';

// Search modes for dropdown
const searchModeOptions = [
  { label: 'Local', value: 'local' },
  { label: 'Hybrid', value: 'hybrid' },
  { label: 'Global', value: 'global' }
];

const ChatInput = ({ onSendMessage, isLoading, searchContext, setSearchContext }) => {
  const [message, setMessage] = useState('');

  // Handle file attachment
  const handleAttachment = () => {
    // This would typically open a file picker
    console.log('Attachment clicked');
  };

  // Handle voice input
  const handleVoiceInput = () => {
    // This would typically start voice recording
    console.log('Voice input clicked');
  };

  const handleSend = () => {
    if (message.trim() && !isLoading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input">
      <div className="chat-input-main">
        <div className="input-with-dropdown">
          <SelectPicker
            data={searchModeOptions}
            value={searchContext}
            onChange={setSearchContext}
            cleanable={false}
            searchable={false}
            // className="search-mode-dropdown"
            renderValue={(value, item) => {
              return item ? item.label.replace('Search Mode: ', '') : value;
            }}
          />

          <InputGroup inside className="message-input-group">
            <Whisper
              placement="top"
              trigger="hover"
              speaker={<Tooltip>Attach file</Tooltip>}
            >
              <IconButton
                icon={<FontAwesomeIcon icon={faPaperclip} />}
                onClick={handleAttachment}
                // className="input-icon-button"
                
                appearance="subtle"
                disabled={isLoading}
              />
            </Whisper>

            <Input
              as="textarea"
              rows={1}
              placeholder="How can I help you?"
              value={message}
              onChange={setMessage}
              onKeyDown={handleKeyPress}
              disabled={isLoading}
              className="chat-text-input"
            />

            <Whisper
              placement="top"
              trigger="hover"
              speaker={<Tooltip>Voice input</Tooltip>}
            >
              <IconButton
                icon={<FontAwesomeIcon icon={faMicrophone} />}
                onClick={handleVoiceInput}
                // className="input-icon-button"
                appearance="subtle"
                disabled={isLoading}
              />
            </Whisper>

            <InputGroup.Button
              onClick={handleSend}
              disabled={isLoading || !message.trim()}
              // className="send-button"
            >
              {isLoading ? <Loader size="xs" /> : <SendIcon />}
            </InputGroup.Button>
          </InputGroup>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
