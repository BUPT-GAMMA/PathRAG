@echo off
REM Start script for PathRAG API only

echo Starting PathRAG API...
cd backend
REM Check if Python virtual environment exists
if not exist .venv (
    echo Python virtual environment not found. Creating one...
    python -m venv .venv
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating Python virtual environment...
call .venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt
echo Backend dependencies installed.

REM Start backend API
echo Starting backend API on port 8000...
uvicorn main:app --host 0.0.0.0 --port 8000

echo API server stopped.
