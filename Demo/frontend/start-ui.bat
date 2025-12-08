@echo off
REM Script to start only the UI on port 3000

echo Starting PathRAG UI on port 3000...

REM Install dependencies if needed
if not exist node_modules (
    echo Installing frontend dependencies...
    call npm install
    echo Frontend dependencies installed.
)

REM Set environment variables
set PORT=3000
set REACT_APP_API_URL=http://localhost:8000

REM Start the UI
echo Starting React application on port 3000...
npx cross-env PORT=3000 npm start

echo UI server stopped.
