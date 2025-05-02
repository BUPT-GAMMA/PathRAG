import React from 'react';
import { Table, Panel } from 'rsuite';
const { Column, HeaderCell, Cell } = Table;

const DocumentList = ({ documents, loading }) => {
  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Format status
  const formatStatus = (status) => {
    if (!status) return 'Completed';

    switch (status.toLowerCase()) {
      case 'processing':
        return 'Processing';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      case 'uploading':
        return 'Uploading';
      default:
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };

  return (
    <Panel bordered className="document-list">
      <Table
        height={400}
        data={documents}
        loading={loading}
        hover={true}
        autoHeight
        style={{ width: '100%' }}
      >
        <Column flexGrow={3} align="left">
          <HeaderCell>Filename</HeaderCell>
          <Cell dataKey="filename" />
        </Column>

        <Column flexGrow={2} align="left">
          <HeaderCell>Type</HeaderCell>
          <Cell dataKey="content_type" />
        </Column>

        <Column flexGrow={1} align="right">
          <HeaderCell>Size</HeaderCell>
          <Cell>
            {(rowData) => formatFileSize(rowData.file_size)}
          </Cell>
        </Column>

        <Column flexGrow={2} align="left">
          <HeaderCell>Uploaded</HeaderCell>
          <Cell>
            {(rowData) => formatDate(rowData.uploaded_at)}
          </Cell>
        </Column>

        <Column flexGrow={1} align="center">
          <HeaderCell>Status</HeaderCell>
          <Cell>
            {(rowData) => {
              const status = rowData.status ? rowData.status.toLowerCase() : 'completed';
              return (
                <div className={`document-status-badge ${status}`}>
                  {formatStatus(rowData.status)}
                </div>
              );
            }}
          </Cell>
        </Column>
      </Table>
    </Panel>
  );
};

export default DocumentList;
