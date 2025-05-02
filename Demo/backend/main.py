import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pathrag")

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    logger.info(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    logger.warning("No .env file found. Using environment variables from the system.")

# Check for required environment variables
required_vars = ['SECRET_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.warning("Please set these variables in your .env file or environment.")
    logger.info("See sample.env for an example configuration.")

# Import models and routes
from models.database import create_tables, User, SessionLocal
from api.auth.jwt_handler import get_password_hash
from api.auth.routes import router as auth_router
from api.features.users.routes import router as users_router
from api.features.chats.routes import router as chats_router
from api.features.documents.routes import router as documents_router
from api.features.knowledge_graph.routes import router as knowledge_graph_router

# Create default users
def create_default_users():
    db = SessionLocal()
    try:
        # Check if users already exist
        if db.query(User).count() == 0:
            # Create 3 default users
            default_users = [
                {"username": "user1", "email": "user1@example.com", "password": "Pass@123"},
                {"username": "user2", "email": "user2@example.com", "password": "Pass@123"},
                {"username": "user3", "email": "user3@example.com", "password": "Pass@123"}
            ]

            for user_data in default_users:
                hashed_password = get_password_hash(user_data["password"])
                user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    hashed_password=hashed_password
                )
                db.add(user)

            db.commit()
            logger.info("Default users created successfully")
    finally:
        db.close()

# Already imported at the top

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: Run before the application starts
    logger.info("Starting PathRAG API...")

    # Create tables
    create_tables()
    logger.info("Database tables created")

    # Create default users
    create_default_users()

    # Create directories
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    working_dir = os.getenv("WORKING_DIR", "./data")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)
    logger.info(f"Directories created: {upload_dir}, {working_dir}")

    # Yield control to the application
    yield

    # Shutdown: Run when the application is shutting down
    logger.info("Shutting down PathRAG API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="PathRAG API",
    description="API for PathRAG application with authentication, chat, document management, and knowledge graph features",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "*")
if cors_origins == "*":
    # Allow all origins in development
    origins = ["*"]
    logger.warning("CORS is configured to allow all origins. This is not recommended for production.")
else:
    # Use specific origins from environment variable in production
    origins = [origin.strip() for origin in cors_origins.split(",")]
    logger.info(f"CORS is configured to allow specific origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(chats_router)
app.include_router(documents_router)
app.include_router(knowledge_graph_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to PathRAG API",
        "docs": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # Log startup configuration
    logger.info(f"Starting server on {host}:{port} (reload={reload})")

    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
