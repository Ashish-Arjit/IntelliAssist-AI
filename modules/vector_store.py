"""FAISS Vector Store module for indexing and semantic similarity search over document chunks.

Encapsulates FAISS index creation, chunk embedding storage, metadata preservation,
and semantic similarity search with normalized relevance score calculations.
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import DEFAULT_TOP_K
from modules.embeddings import EmbeddingManager


class VectorStoreManager:
    """Manager class for creating and querying FAISS vector indexes."""

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[FAISS] = None,
    ):
        """Initialize VectorStoreManager.

        Args:
            embedding_manager: EmbeddingManager instance. If None, creates a default instance.
            vector_store: Optional existing LangChain FAISS instance.
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store: Optional[FAISS] = vector_store

    @property
    def vector_store(self) -> Optional[FAISS]:
        """Get the underlying LangChain FAISS vector store."""
        return self._vector_store

    @property
    def is_initialized(self) -> bool:
        """Check whether the vector store contains a valid FAISS index."""
        return self._vector_store is not None

    @property
    def total_vectors(self) -> int:
        """Get total number of vectors stored in the FAISS index."""
        if self._vector_store is None or self._vector_store.index is None:
            return 0
        return self._vector_store.index.ntotal

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """Create a new FAISS vector store index from a list of LangChain Document chunks.

        Preserves all document content and rich metadata attached to chunks.

        Args:
            documents: List of chunked Document objects.

        Returns:
            The created FAISS vector store instance.
        """
        if not documents:
            raise ValueError("Cannot create FAISS vector store from an empty list of documents.")

        try:
            self._vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_manager.embeddings,
            )
            return self._vector_store
        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS vector store: {str(e)}") from e

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add additional document chunks to an existing FAISS index.

        Args:
            documents: List of Document chunks to insert.

        Returns:
            List of IDs for the added documents.
        """
        if not documents:
            return []

        if not self.is_initialized:
            self.create_from_documents(documents)
            return [str(i) for i in range(len(documents))]

        try:
            return self._vector_store.add_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to FAISS index: {str(e)}") from e

    @staticmethod
    def calculate_similarity_score(distance: float) -> float:
        """Convert FAISS L2 distance on normalized vectors to a similarity score [0.0, 1.0].

        For unit-normalized vectors:
        Cosine Similarity = 1 - (L2_distance^2 / 2)
        Similarity score clamped between 0.0 and 1.0.

        Args:
            distance: Raw L2 distance returned by FAISS.

        Returns:
            Similarity score as a float in range [0.0, 1.0].
        """
        # Cosine similarity calculation from Euclidean distance on unit sphere
        cosine_sim = 1.0 - (distance**2 / 2.0)
        # Clamp to [0.0, 1.0] for friendly display
        return max(0.0, min(1.0, float(cosine_sim)))

    def similarity_search_with_score(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float, float]]:
        """Perform semantic similarity search on the FAISS index for a user query.

        Args:
            query: User search query text.
            top_k: Number of most relevant document chunks to retrieve.
            score_threshold: Optional minimum similarity score filter [0.0 - 1.0].

        Returns:
            List of tuples: (Document, raw_distance, similarity_score_0_to_1).
        """
        if not query or not query.strip():
            return []

        if not self.is_initialized or self.total_vectors == 0:
            return []

        try:
            # Query FAISS vector store
            k = max(1, min(top_k, self.total_vectors))
            raw_results = self._vector_store.similarity_search_with_score(
                query=query.strip(), k=k
            )

            processed_results: List[Tuple[Document, float, float]] = []
            for doc, distance in raw_results:
                dist_float = float(distance)
                similarity_score = self.calculate_similarity_score(dist_float)
                if score_threshold is not None and similarity_score < score_threshold:
                    continue
                processed_results.append((doc, dist_float, similarity_score))


            # Sort descending by similarity score
            processed_results.sort(key=lambda item: item[2], reverse=True)
            return processed_results

        except Exception as e:
            raise RuntimeError(f"Semantic search query failed: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Document]:
        """Perform similarity search returning only the retrieved Document chunks.

        Args:
            query: Search query text.
            top_k: Number of top documents to return.

        Returns:
            List of retrieved Document objects.
        """
        results = self.similarity_search_with_score(query=query, top_k=top_k)
        return [doc for doc, _, _ in results]

    def clear(self) -> None:
        """Reset and clear the current FAISS vector store."""
        self._vector_store = None
