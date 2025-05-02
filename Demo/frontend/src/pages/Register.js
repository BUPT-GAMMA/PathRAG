import React from 'react';
import { Container, Content, Panel, FlexboxGrid, Button } from 'rsuite';
import { Link, Navigate } from 'react-router-dom';
import RegisterForm from '../components/auth/RegisterForm';
import { useAuth } from '../context/AuthContext';

const Register = () => {
  const { isAuthenticated } = useAuth();
  
  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/" />;
  }
  
  return (
    <Container className="register-page">
      <Content>
        <FlexboxGrid justify="center" align="middle" style={{ height: '100vh' }}>
          <FlexboxGrid.Item colspan={12} md={8} lg={6}>
            <Panel header={<h2 className="text-center">PathRAG</h2>} bordered>
              <RegisterForm />
              
              <div className="text-center mt-3">
                <p>Already have an account?</p>
                <Link to="/login">
                  <Button appearance="link">Login</Button>
                </Link>
              </div>
            </Panel>
          </FlexboxGrid.Item>
        </FlexboxGrid>
      </Content>
    </Container>
  );
};

export default Register;
