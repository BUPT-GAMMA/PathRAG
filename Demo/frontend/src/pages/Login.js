import React from 'react';
import { Container, Content, Panel, FlexboxGrid, Button } from 'rsuite';
import { Link, Navigate } from 'react-router-dom';
import LoginForm from '../components/auth/LoginForm';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const { isAuthenticated } = useAuth();
  
  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/" />;
  }
  
  return (
    <Container className="login-page">
      <Content>
        <FlexboxGrid justify="center" align="middle" style={{ height: '100vh' }}>
          <FlexboxGrid.Item colspan={12} md={8} lg={6}>
            <Panel header={<h2 className="text-center">PathRAG - Enterprise Knowledge Graph</h2>} bordered>
              <LoginForm />
              
              <div className="text-center mt-3">
                <p>Don't have an account?</p>
                <Link to="/register">
                  <Button appearance="link">Register</Button>
                </Link>
              </div>
            </Panel>
          </FlexboxGrid.Item>
        </FlexboxGrid>
      </Content>
    </Container>
  );
};

export default Login;
