"""Document chunking module using LangChain RecursiveCharacterTextSplitter.

Splits document text into manageable, semantically coherent chunks while
strictly preserving document-level metadata and attaching chunk-level indices.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Provides methods for splitting documents into configurable chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """Initialize chunker with configurable size and overlap.

        Args:
            chunk_size: Maximum character count per chunk.
            chunk_overlap: Overlapping character count between chunks.
            separators: Optional custom separator hierarchy for splitting.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of LangChain Documents into smaller chunks preserving metadata.

        Args:
            documents: List of input LangChain Documents.

        Returns:
            List of chunked LangChain Documents with enriched metadata.
        """
        if not documents:
            return []

        raw_chunks = self._splitter.split_documents(documents)
        total_chunks = len(raw_chunks)
        enriched_chunks: List[Document] = []

        for idx, chunk in enumerate(raw_chunks, start=1):
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["chunk_index"] = idx
            meta["total_chunks"] = total_chunks
            meta["chunk_character_count"] = len(chunk.page_content)
            meta["chunk_word_count"] = len(chunk.page_content.split())
            enriched_chunks.append(Document(page_content=chunk.page_content, metadata=meta))

        return enriched_chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split raw text into chunks and package into Document objects.

        Args:
            text: Text to split.
            metadata: Optional base metadata dict.

        Returns:
            List of chunked Document objects.
        """
        if not text or not text.strip():
            return []

        doc = Document(page_content=text, metadata=metadata or {})
        return self.split_documents([doc])
