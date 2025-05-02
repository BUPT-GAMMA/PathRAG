import React, { useState } from 'react';
import { Form, Button, Panel, Message, Loader } from 'rsuite';
import { useAuth } from '../../context/AuthContext';

const LoginForm = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { login, isLoading, error } = useAuth();

  const handleSubmit = async () => {
    if (!username || !password) {
      return;
    }

    await login(username, password);
  };

  return (
    <Panel header={<div className="panel-title">Login</div>} bordered className="login-form">
      {error && <Message type="error">{error}</Message>}

      <Form fluid onSubmit={handleSubmit}>
        <Form.Group>
          <Form.ControlLabel>Username</Form.ControlLabel>
          <Form.Control
            name="username"
            value={username}
            onChange={setUsername}
            required
          />
        </Form.Group>

        <Form.Group>
          <Form.ControlLabel>Password</Form.ControlLabel>
          <Form.Control
            name="password"
            type="password"
            value={password}
            onChange={setPassword}
            required
          />
        </Form.Group>

        <Form.Group>
          <Button appearance="primary" type="submit" block disabled={isLoading}>
            {isLoading ? <Loader content="Logging in..." /> : 'Login'}
          </Button>
        </Form.Group>
      </Form>
    </Panel>
  );
};

export default LoginForm;
