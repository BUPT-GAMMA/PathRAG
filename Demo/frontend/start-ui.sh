#!/bin/bash
# Script to start only the UI on port 3000

echo "Starting PathRAG UI on port 3000..."
# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
    echo "Frontend dependencies installed."
fi

# Set environment variables
export PORT=3000
export REACT_APP_API_URL=http://localhost:8000

# Start the UI
echo "Starting React application on port 3000..."
npx cross-env PORT=3000 npm start

echo "UI server stopped."
