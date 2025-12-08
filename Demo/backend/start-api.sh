#!/bin/bash
# Start script for PathRAG API only

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PathRAG API...${NC}"
cd backend
# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Python virtual environment not found. Creating one...${NC}"
    python -m venv .venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating Python virtual environment...${NC}"
source .venv/bin/activate

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}Backend dependencies installed.${NC}"

# Start backend API
echo -e "${BLUE}Starting backend API on port 8000...${NC}"
cd "$(dirname "$0")"
uvicorn main:app --host 0.0.0.0 --port 8000

echo -e "${GREEN}API server stopped.${NC}"
