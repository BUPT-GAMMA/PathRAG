# Installation Guide

This guide provides detailed instructions for setting up the PathRAG application for development and production environments.

## Development Environment Setup

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pathrag
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv .venv

   # Activate on Windows
   .venv\Scripts\activate

   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```
   # JWT Settings (Required)
   SECRET_KEY=your_secret_key_here  # Use a strong random string, e.g., openssl rand -hex 32
   ACCESS_TOKEN_EXPIRE_MINUTES=30   # Token expiration time in minutes

   # Database Settings (Optional - defaults to SQLite)
   DATABASE_URL=sqlite:///./pathrag.db  # SQLite database path

   # Application Settings (Required)
   WORKING_DIR=./data                   # Directory for storing PathRAG data
   UPLOAD_DIR=./uploads                 # Directory for storing uploaded documents

   # Azure OpenAI Settings (Required if using Azure OpenAI)
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_DEPLOYMENT=gpt-4o       # Your Azure OpenAI deployment name
   AZURE_OPENAI_API_KEY=your_azure_key  # Your Azure OpenAI API key
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com

   AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large  # Your Azure embedding model deployment
   AZURE_EMBEDDING_API_VERSION=2023-05-15

   # OpenAI Settings (Required if using OpenAI directly instead of Azure)
   OPENAI_API_KEY=your_openai_key       # Your OpenAI API key
   OPENAI_API_BASE=https://api.openai.com/v1

   # PathRAG Settings (Optional - advanced configuration)
   CHUNK_SIZE=1200                      # Size of text chunks for processing
   CHUNK_OVERLAP=100                    # Overlap between chunks
   MAX_TOKENS=32768                     # Maximum tokens for LLM context
   TEMPERATURE=0.7                      # LLM temperature setting
   TOP_K=40                             # Number of top results to retrieve

   # CORS Settings (Optional - for production)
   CORS_ORIGINS=http://localhost:3000,https://your-domain.com  # Allowed origins
   ```

   > **Important**: Never commit your `.env` file to version control. Add it to your `.gitignore` file.

5. **Initialize the database**
   The database will be automatically created when you start the application for the first time.

6. **Run the development server**

   There are multiple ways to start the server:

   **Option 1: Using the start-api script (Recommended)**
   ```bash
   # On Unix/Linux/macOS
   chmod +x start-api.sh
   ./start-api.sh

   # On Windows
   start-api.bat
   ```

   **Option 2: Using the main.py script**
   ```bash
   # Basic start with default settings
   python main.py

   # With environment variables override
   HOST=127.0.0.1 PORT=8080 DEBUG=true python main.py
   ```

   **Option 3: Using uvicorn directly**
   ```bash
   # Basic start
   uvicorn main:app --host 0.0.0.0 --port 8000

   # With reload for development (auto-restart on file changes)
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

   # With log level configuration
   uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug

   # With SSL/TLS for HTTPS (requires key and cert files)
   uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem
   ```

   **Option 4: Using uvicorn with module syntax**
   ```bash
   # If your app is in a module structure
   uvicorn pathrag.main:app --host 0.0.0.0 --port 8000
   ```

   The API will be available at the configured host and port (default: http://localhost:8000)

### Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd pathrag-ui
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure API endpoint**
   If your backend is running on a different URL, update the `baseURL` in `src/services/api.js`.

4. **Run the development server**
   ```bash
   npm start
   ```
   The application will be available at http://localhost:3000

5. **Build for production**
   ```bash
   npm run build
   ```
   The build files will be in the `build` directory.

## Production Deployment

### Backend Deployment

1. **Set up a production server**
   - Install Python 3.8+
   - Install required dependencies
   - Set up a reverse proxy (Nginx, Apache, etc.)

2. **Deploy the application**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd pathrag

   # Create a virtual environment
   python -m venv .venv
   source .venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with production settings. For production, consider these additional security measures:

   ```
   # Use a stronger secret key and longer expiration for production
   SECRET_KEY=your_very_strong_production_secret_key  # Use openssl rand -hex 32
   ACCESS_TOKEN_EXPIRE_MINUTES=60

   # Database settings - consider using a more robust database for production
   DATABASE_URL=sqlite:///./pathrag_prod.db

   # Set restrictive CORS settings
   CORS_ORIGINS=https://your-production-domain.com

   # Set production-specific paths
   WORKING_DIR=/var/lib/pathrag/data
   UPLOAD_DIR=/var/lib/pathrag/uploads

   # Other settings as in development, but with production API keys
   ```

   For enhanced security in production:
   - Consider using environment variables directly instead of a .env file
   - Use a secrets management service if available (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Ensure file permissions are restricted for sensitive files

4. **Run with a production ASGI server**

   For production deployments, you should use a production-grade ASGI server rather than the development server.

   **Option 1: Using Gunicorn with Uvicorn workers (Recommended for Unix/Linux)**
   ```bash
   # Install Gunicorn
   pip install gunicorn

   # Basic run with 4 worker processes
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

   # With bind address configuration
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 main:app

   # With timeout and worker configuration
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker --timeout 120 --graceful-timeout 60 main:app

   # With SSL/TLS for HTTPS
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker --keyfile ./key.pem --certfile ./cert.pem main:app
   ```

   **Option 2: Using Uvicorn directly (Works on all platforms)**
   ```bash
   # Production mode (no auto-reload)
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

   # With process manager like Supervisor or PM2
   # Example supervisor config:
   # [program:pathrag]
   # command=/path/to/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   # directory=/path/to/pathrag
   # user=www-data
   # autostart=true
   # autorestart=true
   ```

   **Option 3: Using Hypercorn (Alternative ASGI server)**
   ```bash
   # Install Hypercorn
   pip install hypercorn

   # Run with Hypercorn
   hypercorn main:app --bind 0.0.0.0:8000 --workers 4
   ```

   **Number of workers calculation:**
   - A common formula is `(2 x number_of_cores) + 1`
   - For a 4-core server: 9 workers
   - For a 2-core server: 5 workers
   - Adjust based on your server's available memory and CPU resources

5. **Set up a systemd service (optional)**

   For automatic startup and process management, create a systemd service.

   **Option 1: Using Gunicorn with Uvicorn workers**

   Create a file `/etc/systemd/system/pathrag.service`:
   ```ini
   [Unit]
   Description=PathRAG API
   After=network.target

   [Service]
   User=your_user
   Group=your_group
   WorkingDirectory=/path/to/pathrag
   Environment="PATH=/path/to/pathrag/.venv/bin"
   EnvironmentFile=/path/to/pathrag/.env
   ExecStart=/path/to/pathrag/.venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 120 main:app
   Restart=always
   RestartSec=5
   StartLimitIntervalSec=0

   # Security options
   PrivateTmp=true
   ProtectSystem=full
   NoNewPrivileges=true

   [Install]
   WantedBy=multi-user.target
   ```

   **Option 2: Using Uvicorn directly**

   Create a file `/etc/systemd/system/pathrag.service`:
   ```ini
   [Unit]
   Description=PathRAG API
   After=network.target

   [Service]
   User=your_user
   Group=your_group
   WorkingDirectory=/path/to/pathrag
   Environment="PATH=/path/to/pathrag/.venv/bin"
   EnvironmentFile=/path/to/pathrag/.env
   ExecStart=/path/to/pathrag/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   Restart=always
   RestartSec=5
   StartLimitIntervalSec=0

   # Security options
   PrivateTmp=true
   ProtectSystem=full
   NoNewPrivileges=true

   [Install]
   WantedBy=multi-user.target
   ```

   **Enable and start the service:**
   ```bash
   # Reload systemd to recognize the new service
   sudo systemctl daemon-reload

   # Enable the service to start on boot
   sudo systemctl enable pathrag

   # Start the service
   sudo systemctl start pathrag

   # Check the status
   sudo systemctl status pathrag

   # View logs
   sudo journalctl -u pathrag -f
   ```

### Frontend Deployment

1. **Build the frontend**
   ```bash
   cd pathrag-ui
   npm install
   npm run build
   ```

2. **Deploy the build files**
   - Copy the contents of the `build` directory to your web server
   - Configure your web server to serve the static files
   - Set up proper caching headers for static assets

3. **Nginx configuration example**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       root /path/to/pathrag-ui/build;
       index index.html;

       location / {
           try_files $uri $uri/ /index.html;
       }

       location /static/ {
           expires 1y;
           add_header Cache-Control "public, max-age=31536000";
       }

       location /api/ {
           proxy_pass http://localhost:8000/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## Docker Deployment (Optional)

### Backend Dockerfile

Create a `Dockerfile` in the root directory:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data uploads

# Expose port
EXPOSE 8000

# Run the application with uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS:-1} --log-level ${LOG_LEVEL:-info}
```

For a more production-ready setup, you can use a multi-stage build:

```dockerfile
# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data uploads

# Create non-root user for security
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application with uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS:-1} --log-level ${LOG_LEVEL:-info}
```

### Frontend Dockerfile

Create a `Dockerfile` in the `pathrag-ui` directory:
```dockerfile
FROM node:16-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Create an `nginx.conf` file:
```nginx
server {
    listen 80;

    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Compose

Create a `docker-compose.yml` file in the root directory:
```yaml
version: '3'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    env_file:
      - .env
    environment:
      # You can override .env file settings here
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=sqlite:///./pathrag.db
      - WORKING_DIR=/app/data
      - UPLOAD_DIR=/app/uploads
      # Add other environment variables as needed
    restart: unless-stopped

  frontend:
    build: ./pathrag-ui
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - NODE_ENV=production
    restart: unless-stopped

volumes:
  data:
  uploads:
```

For a more secure production setup, consider using Docker secrets:
```yaml
version: '3.8'

services:
  backend:
    # ... other settings as above
    secrets:
      - jwt_secret
      - openai_api_key
    environment:
      - SECRET_KEY_FILE=/run/secrets/jwt_secret
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
      # ... other environment variables

secrets:
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  openai_api_key:
    file: ./secrets/openai_api_key.txt
```

Run with Docker Compose:
```bash
# Basic run
docker-compose up -d

# With environment variable overrides
SECRET_KEY=$(openssl rand -hex 32) docker-compose up -d
```

## Troubleshooting

### Backend Issues

1. **Database errors**
   - Check if the SQLite database file has proper permissions
   - Try deleting the database file and restarting the application

2. **API connection errors**
   - Verify that the backend server is running
   - Check for CORS issues in the browser console
   - Ensure the correct API URL is configured in the frontend

3. **Authentication issues**
   - Check if the JWT secret key is properly set
   - Verify that tokens are being correctly stored and sent

### Frontend Issues

1. **Build errors**
   - Clear the node_modules directory and reinstall dependencies
   - Check for version conflicts in package.json

2. **Runtime errors**
   - Check the browser console for JavaScript errors
   - Verify that all API endpoints are correctly configured

3. **PWA issues**
   - Ensure the service worker is properly registered
   - Check for HTTPS requirements in production
