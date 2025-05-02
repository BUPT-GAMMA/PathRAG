import React, { useState, useEffect } from 'react';
import { Progress, Panel, Tag, Loader } from 'rsuite';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheckCircle, faExclamationTriangle, faSpinner } from '@fortawesome/free-solid-svg-icons';
import { documentAPI } from '../../services/api';

const UploadProgress = ({ document }) => {
  const [status, setStatus] = useState({
    progress: document.status === 'completed' ? 100 : document.status === 'processing' ? 50 : 0,
    status: document.status || 'uploading',
    message: document.status === 'completed'
      ? 'Document processed successfully'
      : document.status === 'processing'
        ? 'Processing document...'
        : 'Initializing upload...'
  });

  const [intervalId, setIntervalId] = useState(null);

  useEffect(() => {
    // Function to fetch document status
    const fetchDocumentStatus = async () => {
      try {
        const response = await documentAPI.getDocumentStatus(document.id);
        if (response && response.data) {
          const documentStatus = response.data.status;

          setStatus({
            progress: documentStatus === 'completed' ? 100 : documentStatus === 'processing' ? 75 : 25,
            status: documentStatus || 'uploading',
            message: documentStatus === 'completed'
              ? 'Document processed successfully'
              : documentStatus === 'failed'
                ? response.data.error_message || 'Processing failed'
                : 'Processing document...'
          });

          // If document is completed or failed, clear the interval
          if (documentStatus === 'completed' || documentStatus === 'failed') {
            if (intervalId) {
              clearInterval(intervalId);
              setIntervalId(null);
            }
          }
        }
      } catch (error) {
        console.error('Error fetching document status:', error);
      }
    };

    // Fetch status immediately
    fetchDocumentStatus();

    // Set up interval to check status every 15 seconds
    const id = setInterval(fetchDocumentStatus, 15000);
    setIntervalId(id);

    // Clean up interval on unmount
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [document.id, intervalId]);

  // Determine status color and icon
  const getStatusInfo = () => {
    switch (status.status) {
      case 'completed':
        return {
          color: 'green',
          icon: <FontAwesomeIcon icon={faCheckCircle} />,
          tagColor: 'green'
        };
      case 'failed':
        return {
          color: 'red',
          icon: <FontAwesomeIcon icon={faExclamationTriangle} />,
          tagColor: 'red'
        };
      case 'processing':
        return {
          color: 'blue',
          icon: <FontAwesomeIcon icon={faSpinner} spin />,
          tagColor: 'blue'
        };
      default:
        return {
          color: 'orange',
          icon: <FontAwesomeIcon icon={faSpinner} spin />,
          tagColor: 'orange'
        };
    }
  };

  const statusInfo = getStatusInfo();

  return (
    <Panel bordered className="upload-progress-panel">
      <div className="upload-progress-header">
        <h4>{document.filename}</h4>
        <Tag color={statusInfo.tagColor}>
          {statusInfo.icon} {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
        </Tag>
      </div>

      <div className="upload-progress-bar">
        <Progress.Line
          percent={status.progress}
          strokeColor={statusInfo.color}
          showInfo={false}
        />
        <span className="progress-percentage">{Math.round(status.progress)}%</span>
      </div>

      <div className="upload-progress-message">
        {status.message}
      </div>
    </Panel>
  );
};

export default UploadProgress;
