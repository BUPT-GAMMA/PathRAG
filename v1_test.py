import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WORKING_DIR = "./PathRAG"

# Define storage paths
BASE_DIR = Path(__file__).resolve().parent
GRAPH_FILE_PATH = os.path.join(BASE_DIR, 'graph')
KV_STORE_FILE_PATH = os.path.join(BASE_DIR, 'kv_store')
VDB_FILE_PATH = os.path.join(BASE_DIR, 'vdb')

# Ensure directories exist
for path in [GRAPH_FILE_PATH, KV_STORE_FILE_PATH, VDB_FILE_PATH]:
    os.makedirs(path, exist_ok=True)

# Ensure necessary JSON files exist
for file_name in ["kv_store_full_docs.json", "kv_store_text_chunks.json", "kv_store_llm_response_cache.json"]:
    file_path = os.path.join(KV_STORE_FILE_PATH, file_name)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("{}")  # Initialize with empty JSON object

# Set up API keys
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
base_url = os.getenv("BASE_URL")
os.environ["OPENAI_API_BASE"] = base_url



if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)

data_file="text.txt"
question="what is this document all about?"

with open(data_file, "r", encoding="utf-8") as f:
    file_content = f.read().strip()  

if not file_content:
    raise ValueError("The input file is empty. Please provide valid content.")

rag.insert(file_content)

print(rag.query(question, param=QueryParam(mode="hybrid")))














