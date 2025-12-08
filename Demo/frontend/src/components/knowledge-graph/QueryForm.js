import React, { useState } from 'react';
import { Form, InputGroup, Input, Loader } from 'rsuite';
import SearchIcon from '@rsuite/icons/Search';

const QueryForm = ({ onSubmit, isLoading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = () => {
    if (query.trim() && !isLoading) {
      onSubmit(query);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Form fluid className="query-form">
      <InputGroup>
        <Input
          placeholder="Enter your query..."
          value={query}
          onChange={setQuery}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <InputGroup.Button onClick={handleSubmit} disabled={isLoading || !query.trim()}>
          {isLoading ? <Loader size="xs" /> : <SearchIcon />}
        </InputGroup.Button>
      </InputGroup>
    </Form>
  );
};

export default QueryForm;
