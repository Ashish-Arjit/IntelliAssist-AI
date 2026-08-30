"""Embedding module for generating vector representations of text and document chunks.

Uses Hugging Face Sentence Transformers via LangChain HuggingFaceEmbeddings to generate
consistent, high-quality vector embeddings for document chunks and user search queries.
"""

from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings

from config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DEVICE, NORMALIZE_EMBEDDINGS


class EmbeddingManager:
    """Manager class for initializing and accessing Hugging Face embedding models."""

    _instance_cache: dict[str, HuggingFaceEmbeddings] = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
    ):
        """Initialize the EmbeddingManager.

        Args:
            model_name: Hugging Face model repository ID or path. Defaults to config value.
            device: Device to run embedding model on ('cpu', 'cuda', etc.).
            normalize_embeddings: Whether to normalize output embeddings to unit length.
        """
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.device = device or EMBEDDING_DEVICE
        self.normalize_embeddings = (
            normalize_embeddings if normalize_embeddings is not None else NORMALIZE_EMBEDDINGS
        )

        self._embeddings = self._get_or_create_embeddings()
        self._dimension: Optional[int] = None

    def _get_or_create_embeddings(self) -> HuggingFaceEmbeddings:
        """Create or retrieve cached HuggingFaceEmbeddings instance."""
        cache_key = f"{self.model_name}_{self.device}_{self.normalize_embeddings}"

        if cache_key not in self._instance_cache:
            try:
                model_kwargs = {"device": self.device}
                encode_kwargs = {"normalize_embeddings": self.normalize_embeddings}
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
                self._instance_cache[cache_key] = embeddings
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Hugging Face embedding model '{self.model_name}': {str(e)}"
                ) from e

        return self._instance_cache[cache_key]

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the underlying LangChain HuggingFaceEmbeddings instance."""
        return self._embeddings

    @property
    def dimension(self) -> int:
        """Get the vector embedding dimensionality."""
        if self._dimension is None:
            sample_embedding = self.embed_query("test dimension probe")
            self._dimension = len(sample_embedding)
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute vector embeddings for a list of document chunk strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of vector embeddings (each a list of floats).
        """
        if not texts:
            return []

        # Filter out empty or whitespace-only strings safely
        safe_texts = [t if t and t.strip() else " " for t in texts]
        return self._embeddings.embed_documents(safe_texts)

    def embed_query(self, text: str) -> List[float]:
        """Compute vector embedding for a single user query string.

        Args:
            text: Query string.

        Returns:
            Vector embedding as a list of floats.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty or whitespace-only query string.")

        return self._embeddings.embed_query(text.strip())
