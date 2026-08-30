"""Configuration module for IntelliAssist AI.

Centralizes default settings and configuration values for document processing,
Hugging Face embeddings, FAISS vector store, and semantic search.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
NORMALIZE_EMBEDDINGS: bool = (
    os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() in ("true", "1", "yes")
)

# Search & Retrieval Configuration
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 4))
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10

# Document Ingestion Configuration
DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 25))
MAX_FILE_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set[str] = {"pdf", "txt", "docx"}
