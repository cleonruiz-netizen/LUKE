# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- API Keys (loaded from environment variables for security) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Pinecone Settings ---
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "legal-docs-peru"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# --- Data & Model Paths ---
PDF_FOLDER_PATH = "pdfs_check"
LOCAL_INDEX_PATH = "local_bm25_index.json" # Stores data for the sparse retriever
# --- AI & Model Settings ---
USE_OPENAI_EMBEDDINGS = True
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4-turbo"  # LLM for synthesizing the final answer
RERANKER_MODEL = 'rerank-multilingual-v2.0'
HYBRID_SEARCH_ALPHA = 0.55  # Weight for dense vs. sparse search (70% dense)

# --- API Metadata ---
API_TITLE = "LUKE - Legal Research API"
API_VERSION = "1.0.0"


EMAIL_SENDER_ADDRESS = os.getenv("EMAIL_SENDER_ADDRESS")
EMAIL_SENDER_PASSWORD = os.getenv("EMAIL_SENDER_PASSWORD") 
SMTP_SERVER_HOST= os.getenv("SMTP_SERVER_HOST")
SMTP_SERVER_PORT= os.getenv("SMTP_SERVER_PORT")

CHAT_SESSIONS_DIR = "chat_sessions"
CLAUSE_BASELINES_PATH = "baselines/clause_baselines.json"

MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING")