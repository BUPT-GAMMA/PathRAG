import React, { useState } from 'react';
import { Form, Button, Panel, Message, Loader } from 'rsuite';
import { useAuth } from '../../context/AuthContext';

const RegisterForm = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [formError, setFormError] = useState('');
  
  const { register, isLoading, error } = useAuth();

  const handleSubmit = async () => {
    // Reset form error
    setFormError('');
    
    // Validate form
    if (!username || !email || !password || !confirmPassword) {
      setFormError('All fields are required');
      return;
    }
    
    if (password !== confirmPassword) {
      setFormError('Passwords do not match');
      return;
    }
    
    // Register user
    await register({ username, email, password });
  };

  return (
    <Panel header={<h3>Register</h3>} bordered className="register-form">
      {(error || formError) && (
        <Message type="error">{formError || error}</Message>
      )}
      
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
          <Form.ControlLabel>Email</Form.ControlLabel>
          <Form.Control
            name="email"
            type="email"
            value={email}
            onChange={setEmail}
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
          <Form.ControlLabel>Confirm Password</Form.ControlLabel>
          <Form.Control
            name="confirmPassword"
            type="password"
            value={confirmPassword}
            onChange={setConfirmPassword}
            required
          />
        </Form.Group>
        
        <Form.Group>
          <Button appearance="primary" type="submit" block disabled={isLoading}>
            {isLoading ? <Loader content="Registering..." /> : 'Register'}
          </Button>
        </Form.Group>
      </Form>
    </Panel>
  );
};

export default RegisterForm;
