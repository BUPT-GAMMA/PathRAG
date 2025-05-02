import React, { useState, useEffect } from 'react';
import { Panel, Loader, Message } from 'rsuite';
import Layout from '../components/Layout';
import Graph from '../components/knowledge-graph/Graph';
import QueryForm from '../components/knowledge-graph/QueryForm';
import { knowledgeGraphAPI } from '../services/api';
import '../styles/knowledge-graph.css';

const KnowledgeGraphPage = () => {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState(null);

  // Fetch initial graph data
  useEffect(() => {
    const fetchGraph = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await knowledgeGraphAPI.getGraph();
        setGraphData(response.data);
      } catch (error) {
        console.error('Error fetching graph:', error);
        setError('Failed to load knowledge graph');
      } finally {
        setLoading(false);
      }
    };

    fetchGraph();
  }, []);

  // Handle query submission
  const handleQuerySubmit = async (query) => {
    if (!query.trim() || querying) return;

    setQuerying(true);
    setError(null);

    try {
      const response = await knowledgeGraphAPI.queryGraph(query);
      setGraphData(response.data);
    } catch (error) {
      console.error('Error querying graph:', error);
      setError('Failed to query knowledge graph');
    } finally {
      setQuerying(false);
    }
  };

  return (
    <Layout>
      <QueryForm onSubmit={handleQuerySubmit} isLoading={querying} />

      {error && <Message type="error">{error}</Message>}

      <Panel bordered className="graph-panel">
        {loading ? (
          <div className="graph-loading">
            <Loader content="Loading knowledge graph..." />
          </div>
        ) : !graphData || graphData.nodes.length === 0 ? (
          <div className="graph-empty">
            <p>No knowledge graph data available. Try uploading some documents first.</p>
          </div>
        ) : (
          <Graph data={graphData} />
        )}
      </Panel>
    </Layout>
  );
};

export default KnowledgeGraphPage;
