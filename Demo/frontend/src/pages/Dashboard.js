import React, { useState, useEffect } from 'react';
import { Panel, Grid, Row, Col, Placeholder } from 'rsuite';
import Layout from '../components/Layout';
import { chatAPI, documentAPI, knowledgeGraphAPI } from '../services/api';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    chats: 0,
    documents: 0,
    graphNodes: 0,
    graphEdges: 0,
  });

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);

      try {
        // Fetch chats
        const chatsResponse = await chatAPI.getChats();

        // Fetch documents
        const documentsResponse = await documentAPI.getDocuments();

        // Fetch knowledge graph
        const graphResponse = await knowledgeGraphAPI.getGraph();

        // Update stats
        setStats({
          chats: chatsResponse.data.chats.length,
          documents: documentsResponse.data.documents.length,
          graphNodes: graphResponse.data.nodes.length,
          graphEdges: graphResponse.data.edges.length,
        });
      } catch (error) {
        console.error('Error fetching stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  return (
    <Layout>
      <h2>Dashboard</h2>

      {loading ? (
        <Placeholder.Grid rows={2} columns={2} active />
      ) : (
        <Grid fluid>
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Panel header="Chats" bordered>
                <h3>{stats.chats}</h3>
                <p>Total chat conversations</p>
              </Panel>
            </Col>

            <Col xs={24} md={12}>
              <Panel header="Documents" bordered>
                <h3>{stats.documents}</h3>
                <p>Uploaded documents</p>
              </Panel>
            </Col>

            <Col xs={24} md={12}>
              <Panel header="Knowledge Graph Nodes" bordered>
                <h3>{stats.graphNodes}</h3>
                <p>Entities in knowledge graph</p>
              </Panel>
            </Col>

            <Col xs={24} md={12}>
              <Panel header="Knowledge Graph Edges" bordered>
                <h3>{stats.graphEdges}</h3>
                <p>Relationships in knowledge graph</p>
              </Panel>
            </Col>
          </Row>
        </Grid>
      )}
    </Layout>
  );
};

export default Dashboard;
