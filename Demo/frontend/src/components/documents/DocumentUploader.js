import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Panel, Loader, Progress, Message, Button, Notification, toaster } from 'rsuite';
import { documentAPI } from '../../services/api';

const DocumentUploader = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [uploadProgress, setUploadProgress] = useState(0); // Used in the onUploadProgress callback
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState({
    status: '',
    message: '',
    progress: 0
  });
  const [, setIsPolling] = useState(false);
  const [isReloading, setIsReloading] = useState(false);
  const pollingIntervalRef = useRef(null);
  const documentIdRef = useRef(null);

  // Status polling effect
  useEffect(() => {
    // Clean up polling interval on component unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Function to start polling for document status
  const startStatusPolling = (documentId) => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    documentIdRef.current = documentId;
    setIsPolling(true);

    // Poll every 15 seconds
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const response = await documentAPI.getDocumentStatus(documentId);

        if (response && response.data) {
          const documentStatus = response.data.status;

          setUploadStatus({
            status: documentStatus || 'processing',
            message: documentStatus === 'completed'
              ? 'Document processed successfully'
              : documentStatus === 'failed'
                ? response.data.error_message || 'Processing failed'
                : 'Processing document...',
            progress: documentStatus === 'completed' ? 100 : 75
          });

          // If document is completed, stop polling and trigger reload
          if (documentStatus === 'completed') {
            stopPollingAndReload();
          } else if (documentStatus === 'failed') {
            stopPolling();
            setError(response.data.error_message || 'Document processing failed');
          }
        }
      } catch (error) {
        console.error('Error polling document status:', error);
      }
    }, 15000); // 15 seconds
  };

  // Function to stop polling
  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    setIsPolling(false);
  };

  // Function to stop polling and trigger reload
  const stopPollingAndReload = async () => {
    stopPolling();

    try {
      setIsReloading(true);
      // Call the reload endpoint
      const reloadResponse = await documentAPI.reloadDocuments();

      if (reloadResponse && reloadResponse.data && reloadResponse.data.success) {
        console.log('Documents reloaded successfully:', reloadResponse.data.message);

        // Update status to show reload completed
        setUploadStatus(prev => ({
          ...prev,
          status: 'completed',
          message: 'Document processed and indexed successfully'
        }));

        // Show success notification
        toaster.push(
          <Notification type="success" header="Document Ready" closable>
            <p>Document has been processed and is now available for search.</p>
          </Notification>,
          { placement: 'topEnd' }
        );

        // Set uploading to false to show success message
        setUploading(false);

        // Notify parent component
        if (onUploadComplete && uploadedFile) {
          onUploadComplete(uploadedFile);
        }
      }
    } catch (error) {
      console.error('Error reloading documents:', error);
    } finally {
      setIsReloading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];

    if (!file) return;

    // Check file type
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/markdown',
      'text/plain',
      'text/html'
    ];

    // Also check by extension for cases where MIME type is not reliable
    const allowedExtensions = ['.pdf', '.docx', '.md', '.txt', '.html', '.htm'];
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      setError('Unsupported file type. Please upload PDF, DOCX, MD, TXT, or HTML files.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);
    setUploadStatus({
      status: 'uploading',
      message: 'Preparing to upload...',
      progress: 0
    });

    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Upload the document
      const response = await documentAPI.uploadDocument(formData, (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
        setUploadStatus({
          status: 'uploading',
          message: `Uploading file: ${percentCompleted}%`,
          progress: percentCompleted / 2 // Scale to 50% of total process
        });
      });

      setUploadedFile(response.data);

      // Set status to processing
      setUploadStatus({
        status: 'processing',
        message: 'Document uploaded, processing started...',
        progress: 50
      });

      // Start polling for document status
      if (response.data && response.data.id) {
        console.log(`Starting status polling for document ID: ${response.data.id}`);
        startStatusPolling(response.data.id);
      } else {
        // If no document ID, just complete the process
        setUploadStatus({
          status: 'completed',
          message: 'Document uploaded successfully',
          progress: 100
        });
        setUploading(false);
      }

      // Don't notify parent component yet - we'll do that after processing is complete
      // We'll keep the uploading state true until processing is complete

    } catch (error) {
      setUploading(false);
      setError(error.response?.data?.detail || 'Failed to upload document');
    }
  }, [startStatusPolling, onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/markdown': ['.md'],
      'text/plain': ['.txt'],
      'text/html': ['.html', '.htm'],
    },
    disabled: uploading,
    multiple: false,
  });

  // Determine status color based on current status
  const getStatusColor = () => {
    switch (uploadStatus.status) {
      case 'completed':
        return 'green';
      case 'failed':
        return 'red';
      case 'processing':
        return 'blue';
      default:
        return 'orange';
    }
  };

  // We use getStatusColor() directly in the Progress.Line component

  return (
    <Panel bordered className="document-uploader">
      {error && <Message type="error" closable>{error}</Message>}

      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${uploading ? 'disabled' : ''}`}
      >
        <input {...getInputProps()} />

        {uploading ? (
          <div className="upload-progress">
            <div className="upload-status-header">
              <Loader content={uploadStatus.status === 'uploading' ? 'Uploading...' : 'Processing...'} />
              <span className={`status-label ${uploadStatus.status}`}>
                {uploadStatus.status.charAt(0).toUpperCase() + uploadStatus.status.slice(1)}
              </span>
            </div>

            <Progress.Line
              percent={uploadStatus.progress}
              status={uploadStatus.status === 'failed' ? 'fail' : 'active'}
              strokeColor={getStatusColor()}
            />

            <p className="status-message">{uploadStatus.message || 'Processing document...'}</p>
          </div>
        ) : (
          <div className="upload-prompt">
            <p>Drag & drop a document here, or click to select</p>
            <p className="upload-note">Supported formats: PDF, DOCX, MD, TXT, HTML</p>
          </div>
        )}
      </div>

      {uploadedFile && !uploading && uploadStatus.status === 'completed' && (
        <div className="upload-success">
          <Message type="success" closable>
            Document uploaded, processed, and indexed successfully: {uploadedFile.filename}
          </Message>
          <Button appearance="primary" onClick={() => onUploadComplete(uploadedFile)}>
            View Documents
          </Button>
        </div>
      )}

      {isReloading && (
        <div className="upload-reloading">
          <Message type="info">
            <Loader size="sm" style={{ marginRight: '10px' }} />
            Reloading document index to make the document available for search...
          </Message>
        </div>
      )}
    </Panel>
  );
};

export default DocumentUploader;
