"""Unit tests for VectorStoreManager."""

import unittest
from langchain_core.documents import Document

from modules.embeddings import EmbeddingManager
from modules.vector_store import VectorStoreManager


class TestVectorStoreManager(unittest.TestCase):
    """Test suite for VectorStoreManager."""

    @classmethod
    def setUpClass(cls):
        cls.embedding_manager = EmbeddingManager()
        cls.docs = [
            Document(
                page_content="Python is an interpreted, high-level, general-purpose programming language.",
                metadata={"file_name": "python_overview.txt", "page": 1, "chunk_index": 1},
            ),
            Document(
                page_content="FAISS is a library developed by Meta for fast nearest neighbor search in high-dimensional spaces.",
                metadata={"file_name": "vector_db.txt", "page": 1, "chunk_index": 1},
            ),
            Document(
                page_content="Photosynthesis is the biological process by which plants convert light energy into chemical energy.",
                metadata={"file_name": "biology.pdf", "page": 3, "chunk_index": 2},
            ),
        ]

    def test_create_and_count_vectors(self):
        """Test index initialization and vector count."""
        v_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        self.assertFalse(v_manager.is_initialized)
        self.assertEqual(v_manager.total_vectors, 0)

        v_manager.create_from_documents(self.docs)
        self.assertTrue(v_manager.is_initialized)
        self.assertEqual(v_manager.total_vectors, 3)

    def test_similarity_search_semantic_matching(self):
        """Test semantic query retrieval accuracy and metadata retention."""
        v_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_manager.create_from_documents(self.docs)

        # Search for vector database topic
        results = v_manager.similarity_search_with_score("similarity search indexing vector database", top_k=2)
        self.assertGreaterEqual(len(results), 1)

        top_doc, distance, similarity_score = results[0]
        self.assertIn("FAISS", top_doc.page_content)
        self.assertEqual(top_doc.metadata["file_name"], "vector_db.txt")
        self.assertGreater(similarity_score, 0.4)
        self.assertLessEqual(similarity_score, 1.0)
        self.assertIsInstance(distance, float)

    def test_search_biology_topic(self):
        """Test semantic query for biology topic."""
        v_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_manager.create_from_documents(self.docs)

        results = v_manager.similarity_search_with_score("How do plants turn sunlight into energy?", top_k=1)
        self.assertEqual(len(results), 1)
        top_doc = results[0][0]
        self.assertIn("Photosynthesis", top_doc.page_content)
        self.assertEqual(top_doc.metadata["file_name"], "biology.pdf")
        self.assertEqual(top_doc.metadata["page"], 3)

    def test_add_documents(self):
        """Test incrementally adding documents to existing index."""
        v_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_manager.create_from_documents(self.docs[:2])
        self.assertEqual(v_manager.total_vectors, 2)

        v_manager.add_documents([self.docs[2]])
        self.assertEqual(v_manager.total_vectors, 3)

        results = v_manager.similarity_search("plants photosynthesis", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Photosynthesis", results[0].page_content)

    def test_empty_query_and_edge_cases(self):
        """Test empty queries, uninitialized store, and clear functionality."""
        v_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        
        # Uninitialized store search
        self.assertEqual(v_manager.similarity_search_with_score("query"), [])

        # Create store and test empty query
        v_manager.create_from_documents(self.docs)
        self.assertEqual(v_manager.similarity_search_with_score(""), [])
        self.assertEqual(v_manager.similarity_search_with_score("   "), [])

        # Clear store
        v_manager.clear()
        self.assertFalse(v_manager.is_initialized)
        self.assertEqual(v_manager.total_vectors, 0)

    def test_create_with_empty_list_raises(self):
        """Test creating index with empty list raises ValueError."""
        v_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        with self.assertRaises(ValueError):
            v_manager.create_from_documents([])


if __name__ == "__main__":
    unittest.main()
