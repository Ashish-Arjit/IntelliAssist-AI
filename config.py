"""Configuration module for IntelliAssist AI.

Centralizes default settings and configuration values for document processing,
Hugging Face embeddings, FAISS vector store, and semantic search.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def get_config_value(key: str, default: str = "") -> str:
    """Retrieve configuration from environment variable or Streamlit secrets with fallback."""
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# Embedding Configuration
DEFAULT_EMBEDDING_MODEL: str = get_config_value(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE: str = get_config_value("EMBEDDING_DEVICE", "cpu")
NORMALIZE_EMBEDDINGS: bool = (
    get_config_value("NORMALIZE_EMBEDDINGS", "True").lower() in ("true", "1", "yes")
)

# Search & Retrieval Configuration
DEFAULT_TOP_K: int = int(get_config_value("DEFAULT_TOP_K", "4"))
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10

# Document Ingestion Configuration
DEFAULT_CHUNK_SIZE: int = int(get_config_value("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP: int = int(get_config_value("DEFAULT_CHUNK_OVERLAP", "200"))
MAX_UPLOAD_SIZE_MB: int = int(get_config_value("MAX_UPLOAD_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set[str] = {"pdf", "txt", "docx"}

# LLM & RAG Pipeline Configuration
DEFAULT_LLM_PROVIDER: str = get_config_value("LLM_PROVIDER", "gemini")
DEFAULT_LLM_MODEL: str = get_config_value("LLM_MODEL_NAME", "gemini-1.5-flash")
DEFAULT_TEMPERATURE: float = float(get_config_value("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_OUTPUT_TOKENS: int = int(get_config_value("LLM_MAX_OUTPUT_TOKENS", "1024"))
DEFAULT_MIN_SIMILARITY_THRESHOLD: float = float(get_config_value("MIN_SIMILARITY_THRESHOLD", "0.20"))


# Grounded RAG Messages & Fallbacks
NO_CONTEXT_FOUND_MESSAGE: str = (
    "I couldn't find this information in the uploaded documents."
)
NO_DOCUMENTS_MESSAGE: str = (
    "Please upload and process at least one document before asking questions."
)
MISSING_API_KEY_MESSAGE: str = (
    "LLM API key not detected. Please configure GOOGLE_API_KEY in your environment, Streamlit secrets, or sidebar input."
)

# Summarization Configuration
DEFAULT_SUMMARY_STYLE: str = "Executive Summary"
SUMMARY_STYLES: list[str] = [
    "Executive Summary",
    "Key Points / Bullet Points",
    "Comprehensive Overview",
]
MAX_DIRECT_SUMMARY_CHARS: int = int(os.getenv("MAX_DIRECT_SUMMARY_CHARS", 12000))
SUMMARY_CHUNK_SIZE: int = int(os.getenv("SUMMARY_CHUNK_SIZE", 4000))

# Sentiment Analysis Configuration
DEFAULT_SENTIMENT_MODEL: str = os.getenv(
    "SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Intent Analysis Configuration
INTENT_CATEGORIES: list[str] = [
    "Question",
    "Summary Request",
    "Information Search",
    "Explanation Request",
]

