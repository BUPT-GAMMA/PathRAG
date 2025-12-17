# PathRAG - Knowledge Graph-based RAG System

This PathRAG Demo App is a comprehensive knowledge graph-based Retrieval-Augmented Generation (RAG) system that combines document processing, chat functionality, and knowledge graph visualization in a single application.

## What is PathRAG?

PathRAG (Path-based Retrieval Augmented Generation) is an advanced approach to knowledge retrieval and generation that combines the power of knowledge graphs with large language models (LLMs).

### Core Concepts

#### Knowledge Graph Integration
PathRAG builds and maintains a knowledge graph from your documents, where:
- **Nodes** represent entities (people, organizations, concepts, locations, etc.)
- **Edges** represent relationships between these entities
- **Properties** store additional information about entities and relationships

#### Path-based Retrieval
Unlike traditional RAG systems that rely solely on vector similarity:
1. PathRAG identifies relevant paths through the knowledge graph
2. These paths provide contextual connections between entities
3. The system can follow logical relationships to find information not directly mentioned

#### Hybrid Search
PathRAG combines multiple search strategies:
- **Vector search** for semantic similarity
- **Graph traversal** for relationship-based connections
- **Entity-centric retrieval** for focused information about specific entities

#### Advantages Over Traditional RAG
- **Relational understanding**: Captures relationships between concepts, not just similarity
- **Explainability**: Provides clear paths showing how information is connected
- **Reduced hallucinations**: Grounds responses in explicit knowledge connections
- **Complex reasoning**: Can answer multi-hop questions requiring several logical steps

### How PathRAG Works

1. **Document Processing**:
   - Documents are chunked into manageable pieces
   - Entities and relationships are extracted using NLP techniques
   - A knowledge graph is constructed connecting these entities

2. **Query Processing**:
   - User queries are analyzed to identify key entities and intents
   - The system identifies relevant paths in the knowledge graph
   - Both vector similarity and graph structure are used to retrieve information

3. **Response Generation**:
   - Retrieved context from multiple paths is synthesized
   - The LLM generates responses grounded in this structured knowledge
   - Responses include information from across the knowledge graph

## Features

### Core Functionality
- **Document Management**: Upload, process, and manage documents (PDF, DOCX, MD, TXT, HTML, etc.)
- **Chat Interface**: Thread-based chat system with context-aware responses
- **Knowledge Graph**: Visualize and query the knowledge graph built from your documents
- **User Management**: User authentication and personalization

### Technical Features
- **React Frontend**: Modern UI built with React and RSuite components
- **FastAPI Backend**: High-performance Python API with async support
- **SQLite Database**: Lightweight database for storing user data, chat threads, and document metadata
- **Thread-Based Chat**: Persistent chat threads with unique IDs
- **Document Processing**: Automatic extraction of text and entities from various document formats
- **Knowledge Graph Visualization**: Interactive visualization using D3.js
- **Theme Customization**: Customizable UI themes (blue, red, violet)
- **Automatic Document Reloading**: System checks document status every 15 seconds and automatically reloads when processing completes

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLite**: Lightweight database for storing users, chats, and documents
- **JWT**: JSON Web Tokens for authentication
- **PathRAG**: Path-based Retrieval Augmented Generation for knowledge graph and chat functionality
- **NetworkX**: Graph data structure and algorithms (for development/demo)
- **NanoVectorDB**: Local file-based vector storage (for development/demo)

### Frontend
- **React**: JavaScript library for building user interfaces
- **RSuite**: UI component library with responsive design
- **D3.js**: Data visualization library for knowledge graph
- **React Router**: Navigation and routing
- **Axios**: HTTP client for API requests
- **React Dropzone**: Drag-and-drop file upload
- **Font Awesome**: Icon library

## Project Structure

### Backend
```
/api
  /auth - Authentication module
    - jwt_handler.py - JWT token handling
    - routes.py - Authentication endpoints
    - schemas.py - Authentication data models
  /features - Feature modules
    /users - User management
    /chats - Chat functionality
    /documents - Document management
    /knowledge_graph - Knowledge graph functionality
/models - Database models
  - database.py - SQLite database setup and models
main.py - Main application entry point
```

### Frontend
```
/pathrag-ui
  /public - Static files
  /src
    /components - Reusable components
      /auth - Authentication components
      /chat - Chat components
      /documents - Document components
      /knowledge-graph - Knowledge graph components
    /context - React context providers
    /pages - Application pages
    /services - API services
    /utils - Utility functions
    App.js - Main application component
    index.js - Application entry point
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Quick Start (Recommended)
#### Command Line (Without GUI)
Python Version: Python 3.10.18
```bash
cd PathRAG
pip install -e .
```
- Use OpenAI API key
- You can quickly experience this project in the `v1_test.py` file.
- Set OpenAI API key in environment if using OpenAI models: `api_key="sk-...".` in the `v1_test.py` and `llm.py` file.
- Prepare your retrieval document "text.txt".
- Use the following Python snippet in the `v1_test.py` file to initialize PathRAG and perform queries.
  
```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./your_working_dir"
api_key="your_api_key"
os.environ["OPENAI_API_KEY"] = api_key
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)

data_file="./text.txt"
question="your_question"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
```
##### Quick Start with models from different sources
- You can use the model from huggingface, ollama, modelscope, local and vllm.
- You can quickly experience this project in the `rag_test.py` file.
- Select your model source—— hf / vllm / ollama / ms / local.
- Prepare your llm_model，embedding_model and  retrieval document "your data file" . 
- Use the following Python snippet in the `rag_test.py` file to use models from different sources.
- Detailed examples can be referred to `examples.txt`.

```Python
import os
import asyncio
import torch
from PathRAG.RAGRunner import RAGRunner
if __name__ == "__main__":
	backend = "your_model_source" # hf / vllm / ollama / ms / local
	working_dir = f"your_working_dir" 
	llm_model_name = "Qwen/Qwen3-0.6B" # model name or path of model
	embedding_model_name = "iic/nlp_corom_sentence-embedding_english-base" # embedding model name or path ofembedding model
	# ollama additional parameters
	llm_model_kwargs = {
	"host": "http://localhost:11434",
	"options": {"num_ctx": 8192},
	"timeout": 300,
	} if backend == "ollama" else {}
	
	runner = RAGRunner(
		backend=backend,
		working_dir=working_dir,
		llm_model_name=llm_model_name,
		embedding_model_name=embedding_model_name,
		llm_model_max_token_size=8192,
		llm_model_kwargs=llm_model_kwargs,
		embedding_dim=768,
		embedding_max_token_size=5000,
	)
	data_file = "your_data_file"
	question = "your_question"
	with open(data_file, "r", encoding="utf-8") as f:
		runner.insert_text(f.read())
	answer = runner.query(question, mode="hybrid")
	print("question:", question)
	print("answer:", answer)
```

##### Parameter modification
You can adjust the relevant parameters in the `base.py` and `operate.py` files.
- You can change the hyperparameter `top_k` in `base.py`, where `top_k` represents the number of nodes retrieved
- You can change the hyperparameters `alpha` and `threshold` in `operate.py`, where `alpha` represents the decay rate of information propagation along the edges, and `threshold` is the pruning threshold.
##### Batch Insert
```python
import os
folder_path = "your_folder_path"  

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        rag.insert(file.read())
```

#### Evaluation

##### Dataset
The dataset used in PathRAG can be downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).
##### Eval Metrics
<details>
<summary> Prompt </summary>
  
```python
You will evaluate two answers to the same question based on five criteria: **Comprehensiveness**, **Diversity**, **logicality**, **Coherence**, **Relevance**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **logicality**: How logically does the answer respond to all parts of the question?
- **Coherence**: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?
- **Relevance**: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.

Here is the question:
{query}

Here are the two answers:
**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the five criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Diversity": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "logicality": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Coherence": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Relevance": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }}
}}
```
</details>

#### With GUI
##### Starting the API

Use our start script to set up and run the API:

**For Unix/Linux/macOS:**
```bash
# Make the script executable (first time only)
chmod +x start-api.sh

# Run the API
./start-api.sh
```

**For Windows:**
```bash
# Run the API
start-api.bat
```

These scripts will:
1. Create a Python virtual environment named `.venv` if it doesn't exist
2. Install all backend dependencies
3. Start the backend API on port 8000

The API will be available at:
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

##### Starting the UI

Navigate to the UI directory and start the React application:

```bash
# Navigate to the UI directory
cd ui

# Install dependencies (first time only)
npm install

# Start the UI
npm start
```

The UI will be available at:
- Frontend UI: http://localhost:3000

### Manual Setup

If you prefer to set up and run the components separately, follow these instructions:

#### Backend Setup
1. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate on Windows
   .venv\Scripts\activate

   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Copy the sample environment file and modify it with your settings:
   ```bash
   cp sample.env .env
   # Edit .env with your preferred text editor
   ```

   Key environment variables include:
   ```
   # JWT Authentication
   SECRET_KEY=your_secret_key_here  # Generate with: openssl rand -hex 32
   ACCESS_TOKEN_EXPIRE_MINUTES=30

   # Application Directories
   WORKING_DIR=./data
   UPLOAD_DIR=./uploads

   # Database Configuration
   DATABASE_URL=sqlite:///./pathrag.db

   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   DEBUG=False
   LOG_LEVEL=info
   CORS_ORIGINS=http://localhost:3000

   # AI Model Settings (choose one option)
   # Option 1: Azure OpenAI
   AZURE_OPENAI_API_KEY=your_azure_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

   # Option 2: OpenAI direct
   OPENAI_API_KEY=your_openai_key
   OPENAI_API_BASE=https://api.openai.com/v1

   # PathRAG Configuration
   CHUNK_SIZE=1200
   CHUNK_OVERLAP=100
   MAX_TOKENS=32768
   TEMPERATURE=0.7
   TOP_K=40
   ```

   See [INSTALLATION.md](INSTALLATION.md) for detailed environment variable configuration and [sample.env](sample.env) for a complete example.

4. Start the backend server:
   ```bash
   python main.py
   ```

   The API will be available at http://localhost:8000

   For more advanced options, see [INSTALLATION.md](INSTALLATION.md).

5. API Documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

#### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The application will be available at http://localhost:3000

## Usage

### Authentication
- Use the default credentials to log in:
  - Username: user1, Password: Pass@123
  - Username: user2, Password: Pass@123
  - Username: user3, Password: Pass@123
- Or register a new account using the registration form

### Chat Threads
1. Navigate to the Chat page
2. Click "New Chat" to start a new thread
3. Type your message in the input field
4. Press Enter or click the send button
5. View the AI response
6. Your chat threads are saved and can be accessed from the sidebar
7. Each thread has a unique ID and maintains its own conversation history
8. Thread titles are automatically updated based on the first message

### Knowledge Graph
1. Navigate to the Knowledge Graph page
2. Enter a query in the search field to filter the graph
3. Interact with the graph by dragging nodes
4. Zoom in/out using the mouse wheel
5. Click on nodes to see entity details

### Document Management
1. Navigate to the Documents page
2. Click "Upload Document" button
3. Drag and drop a file or click to select a file
4. Monitor the upload progress
5. The system automatically checks document status every 15 seconds
6. When processing completes, the system automatically reloads the PathRAG instance
7. You can also manually reload the PathRAG instance by clicking the "Reload Documents" button
8. View the uploaded documents in the list with their processing status

## Data Models

### User
- `id`: Integer (Primary Key)
- `username`: String (Unique)
- `email`: String (Unique)
- `hashed_password`: String
- `created_at`: DateTime
- `theme`: String (Default: "blue")

### Thread
- `id`: Integer (Primary Key)
- `uuid`: String (Unique)
- `user_id`: Integer (Foreign Key to User)
- `title`: String
- `created_at`: DateTime
- `updated_at`: DateTime
- `is_deleted`: Boolean (Default: False)

### Chat
- `id`: Integer (Primary Key)
- `user_id`: Integer (Foreign Key to User)
- `thread_id`: Integer (Foreign Key to Thread)
- `role`: String ('user' or 'system')
- `message`: Text
- `created_at`: DateTime

### Document
- `id`: Integer (Primary Key)
- `user_id`: Integer (Foreign Key to User)
- `filename`: String
- `content_type`: String
- `file_path`: String
- `file_size`: Integer
- `uploaded_at`: DateTime
- `status`: String
- `processed_at`: DateTime (Nullable)
- `error_message`: Text (Nullable)

## API Endpoints

### Authentication
- `POST /token`: Login and get access token
- `POST /register`: Register a new user
- `GET /users/me`: Get current user information

### Users
- `GET /users/`: Get all users
- `POST /users/theme`: Update user theme

### Chat Threads
- `GET /chats/threads`: Get all chat threads
- `POST /chats/threads`: Create a new chat thread
- `GET /chats/threads/{thread_uuid}`: Get a specific thread with all its chats
- `PUT /chats/threads/{thread_uuid}`: Update a thread's title
- `DELETE /chats/threads/{thread_uuid}`: Mark a thread as deleted

### Chats
- `GET /chats/`: Get all chats
- `GET /chats/recent`: Get the 5 most recent chat threads
- `POST /chats/chat/{thread_uuid}`: Create a new chat message in a thread

### Documents
- `GET /documents/`: Get all documents
- `POST /documents/upload`: Upload a document
- `GET /documents/{document_id}`: Get a specific document
- `GET /documents/{document_id}/status`: Get document processing status
- `POST /documents/reload`: Reload the PathRAG instance to recognize new documents

### Knowledge Graph
- `GET /knowledge-graph/`: Get the knowledge graph
- `POST /knowledge-graph/query`: Query the knowledge graph

## Storage Options

### Demo/Development (Default)
- **Vector Storage**: NanoVectorDB (local file-based vector store)
- **Graph Storage**: NetworkX (local in-memory graph)
- **Key-Value Storage**: JsonKVStorage (local file-based storage)

> **Note**: These storage options are suitable for demonstration and development purposes only. They are not recommended for production use with large datasets or high traffic.

### Production Options
For production environments, consider using these alternatives:
- **Vector Databases**: PostgreSQL (pgvector), Pinecone, DataStax, Azure Cognitive Search, Azure SQL(Preview)
- **Graph Databases**: Neo4j, ArangoDB, Apache AGE (PostgreSQL extension), CosmosDB, Azure SQL
- **Document Databases**: MongoDB, Cassandra, CosmosDB

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Use Cases

PathRAG is particularly effective for:

### Knowledge-Intensive Applications
- **Research assistance**: Connecting findings across multiple papers and sources
- **Legal document analysis**: Identifying relationships between cases, statutes, and legal concepts
- **Medical knowledge systems**: Connecting symptoms, conditions, treatments, and research

### Complex Information Retrieval
- **Multi-hop question answering**: "What treatments were developed based on research by scientists who studied under Marie Curie?"
- **Contextual understanding**: Understanding how different parts of a document relate to each other
- **Exploratory research**: Discovering unexpected connections between concepts

### Enterprise Knowledge Management
- **Corporate knowledge bases**: Connecting information across departments and documents
- **Compliance and regulation**: Tracking relationships between policies, regulations, and procedures
- **Institutional memory**: Preserving and accessing organizational knowledge

## Limitations and Considerations

- **Knowledge graph quality**: The system's effectiveness depends on the quality of entity and relationship extraction
- **Computational complexity**: Graph operations can be more resource-intensive than simple vector searches
- **Domain specificity**: May require domain-specific entity extraction for specialized fields
- **Storage limitations**: The default storage options (NanoVectorDB, NetworkX) are not suitable for large-scale production use

## Authors & Contributors

### PathRAG Core Logic Research Team
- Boyu Chen¹, Zirui Guo¹,², Zidan Yang¹,³, Yuluo Chen¹, Junze Chen¹, Zhenghao Liu³, Chuan Shi¹, Cheng Yang¹
  1. Beijing University of Posts and Telecommunications
  2. University of Hong Kong
  3. Northeastern University

  Contact: chenbys4@bupt.edu.cn, yangcheng@bupt.edu.cn

### Demo Application Contributor
- Robert Dennyson, Solution Architect, UK
- Contact: robertdennyson@live.in

## Acknowledgements
- PathRAG for the knowledge graph and retrieval augmented generation capabilities
- RSuite for the UI components
- D3.js for the knowledge graph visualization
