"""Unit tests for EmbeddingManager."""

import unittest
from modules.embeddings import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    """Test suite for EmbeddingManager."""

    @classmethod
    def setUpClass(cls):
        """Initialize EmbeddingManager once for the test suite."""
        cls.manager = EmbeddingManager()

    def test_embedding_initialization_and_dimension(self):
        """Verify model initialization and vector dimension (384 for all-MiniLM-L6-v2)."""
        self.assertEqual(self.manager.dimension, 384)
        self.assertIsNotNone(self.manager.embeddings)

    def test_embed_query(self):
        """Verify embedding generation for a single query."""
        vec = self.manager.embed_query("Artificial Intelligence document search")
        self.assertEqual(len(vec), 384)
        self.assertIsInstance(vec, list)
        self.assertIsInstance(vec[0], float)

    def test_embed_documents(self):
        """Verify batch embedding generation for multiple text snippets."""
        texts = [
            "Natural language processing is a branch of computer science.",
            "FAISS is a library for efficient similarity search.",
            "Document preprocessing removes irregular whitespaces.",
        ]
        vectors = self.manager.embed_documents(texts)
        self.assertEqual(len(vectors), 3)
        for vec in vectors:
            self.assertEqual(len(vec), 384)

    def test_embed_empty_query_raises_error(self):
        """Verify error handling when query is empty."""
        with self.assertRaises(ValueError):
            self.manager.embed_query("")
        with self.assertRaises(ValueError):
            self.manager.embed_query("   ")

    def test_embed_empty_documents_list(self):
        """Verify empty documents list returns empty list."""
        self.assertEqual(self.manager.embed_documents([]), [])


if __name__ == "__main__":
    unittest.main()
