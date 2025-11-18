from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_FILE_PATH = os.path.join(BASE_DIR,'graph')
KV_STORE_FILE_PATH = os.path.join(BASE_DIR,'kv_store')
VDB_FILE__PATH = os.path.join(BASE_DIR,'vdb')