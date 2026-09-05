import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from config import (

    DEFAULT_LLM_MODEL,
    NO_CONTEXT_FOUND_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    MISSING_API_KEY_MESSAGE,
)
from modules.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test suite for RAGPipeline methods, formatting, citations, and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vector_store = MagicMock()
        self.rag = RAGPipeline(
            vector_store_manager=self.mock_vector_store,
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.0,
            api_key="test_api_key_12345",
        )

        self.sample_docs = [
            (
                Document(
                    page_content="IntelliAssist AI is a modular RAG platform supporting PDF, TXT, and DOCX files.",
                    metadata={"file_name": "architecture.pdf", "page": 1, "chunk_index": 1, "total_chunks": 4},
                ),
                0.15,
                0.96,
            ),
            (
                Document(
                    page_content="Google Gemini LLM provides grounded answers using FAISS retrieved context.",
                    metadata={"file_name": "overview.docx", "page": None, "chunk_index": 2, "total_chunks": 5},
                ),
                0.35,
                0.85,
            ),
        ]

    def test_initialization(self):
        """Test proper initialization of RAGPipeline."""
        self.assertEqual(self.rag.model_name, DEFAULT_LLM_MODEL)
        self.assertEqual(self.rag.temperature, 0.0)
        self.assertTrue(self.rag.is_api_key_configured())
        self.assertIsNotNone(self.rag.prompt_template)

    def test_api_key_detection(self):
        """Test API key validation logic."""
        rag_no_key = RAGPipeline(vector_store_manager=self.mock_vector_store, api_key="")
        self.assertFalse(rag_no_key.is_api_key_configured())

        rag_placeholder_key = RAGPipeline(
            vector_store_manager=self.mock_vector_store,
            api_key="your_google_api_key_here",
        )
        self.assertFalse(rag_placeholder_key.is_api_key_configured())

        rag_valid_key = RAGPipeline(
            vector_store_manager=self.mock_vector_store,
            api_key="AIzaSyValidGeminiKey",
        )
        self.assertTrue(rag_valid_key.is_api_key_configured())

    def test_prompt_template_contains_grounding_rules(self):
        """Test that system prompt enforces strict document grounding."""
        messages = self.rag.prompt_template.messages
        self.assertEqual(len(messages), 2)
        system_prompt = messages[0].prompt.template
        self.assertIn("IntelliAssist AI", system_prompt)
        self.assertIn(NO_CONTEXT_FOUND_MESSAGE, system_prompt)
        self.assertIn("Strict Grounding Rules", system_prompt)

    def test_format_context_from_documents(self):
        """Test formatting of retrieved documents with metadata headers."""
        context_str = self.rag.format_context_from_documents(self.sample_docs)
        self.assertIn("[Source #1: architecture.pdf, Page 1, Chunk 1 | Relevance: 96.0%]", context_str)
        self.assertIn("[Source #2: overview.docx, Chunk 2 | Relevance: 85.0%]", context_str)
        self.assertIn("IntelliAssist AI is a modular RAG platform", context_str)
        self.assertIn("Google Gemini LLM provides grounded answers", context_str)

    def test_format_context_empty_documents(self):
        """Test context formatting with empty list."""
        context_str = self.rag.format_context_from_documents([])
        self.assertEqual(context_str, "No relevant document context found.")

    def test_extract_citations(self):
        """Test structured citation extraction from document tuples."""
        citations = self.rag.extract_citations(self.sample_docs)
        self.assertEqual(len(citations), 2)

        # First citation
        c1 = citations[0]
        self.assertEqual(c1["rank"], 1)
        self.assertEqual(c1["file_name"], "architecture.pdf")
        self.assertEqual(c1["page"], 1)
        self.assertEqual(c1["chunk_index"], 1)
        self.assertEqual(c1["total_chunks"], 4)
        self.assertAlmostEqual(c1["similarity_score"], 0.96)
        self.assertIn("modular RAG platform", c1["snippet"])

        # Second citation
        c2 = citations[1]
        self.assertEqual(c2["rank"], 2)
        self.assertEqual(c2["file_name"], "overview.docx")
        self.assertIsNone(c2["page"])
        self.assertEqual(c2["chunk_index"], 2)
        self.assertEqual(c2["total_chunks"], 5)
        self.assertAlmostEqual(c2["similarity_score"], 0.85)

    def test_answer_empty_question(self):
        """Test handling of empty or whitespace-only questions."""
        res = self.rag.answer_question("")
        self.assertEqual(res["status"], "warning")
        self.assertEqual(len(res["citations"]), 0)
        self.assertIn("Please enter a valid question", res["answer"])

        res_spaces = self.rag.answer_question("   \n\t  ")
        self.assertEqual(res_spaces["status"], "warning")

    def test_answer_no_vector_store(self):
        """Test handling when vector store is empty / uninitialized."""
        self.mock_vector_store.is_initialized = False
        self.mock_vector_store.total_vectors = 0

        res = self.rag.answer_question("What is IntelliAssist?")
        self.assertEqual(res["status"], "warning")
        self.assertEqual(res["answer"], NO_DOCUMENTS_MESSAGE)
        self.assertEqual(len(res["citations"]), 0)

    def test_answer_missing_api_key(self):
        """Test handling when LLM API key is not provided."""
        self.mock_vector_store.is_initialized = True
        self.mock_vector_store.total_vectors = 5
        rag_no_key = RAGPipeline(vector_store_manager=self.mock_vector_store, api_key="")

        res = rag_no_key.answer_question("What formats are supported?")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"], "MISSING_API_KEY")
        self.assertEqual(res["answer"], MISSING_API_KEY_MESSAGE)

    def test_answer_no_relevant_context_found(self):
        """Test handling when FAISS similarity search returns no matching chunks."""
        self.mock_vector_store.is_initialized = True
        self.mock_vector_store.total_vectors = 5
        self.mock_vector_store.similarity_search_with_score.return_value = []

        res = self.rag.answer_question("What is the quantum state of Jupiter?")
        self.assertEqual(res["status"], "no_context")
        self.assertEqual(res["answer"], NO_CONTEXT_FOUND_MESSAGE)
        self.assertEqual(len(res["citations"]), 0)

    @patch("modules.rag_pipeline.ChatGoogleGenerativeAI")
    def test_answer_successful_llm_generation(self, mock_gemini_cls):
        """Test complete end-to-end question answering flow with mocked LLM response."""
        self.mock_vector_store.is_initialized = True
        self.mock_vector_store.total_vectors = 2
        self.mock_vector_store.similarity_search_with_score.return_value = self.sample_docs

        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "IntelliAssist AI supports PDF, TXT, and DOCX document formats."
        mock_llm_instance.invoke.return_value = mock_response
        mock_gemini_cls.return_value = mock_llm_instance

        res = self.rag.answer_question("What file formats are supported?", top_k=2)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["answer"], "IntelliAssist AI supports PDF, TXT, and DOCX document formats.")
        self.assertEqual(len(res["citations"]), 2)
        self.assertEqual(res["retrieved_chunks_count"], 2)
        self.assertEqual(res["citations"][0]["file_name"], "architecture.pdf")

    @patch("modules.rag_pipeline.ChatGoogleGenerativeAI")
    def test_answer_llm_api_failure_handling(self, mock_gemini_cls):
        """Test graceful exception handling when LLM invocation raises an error."""
        self.mock_vector_store.is_initialized = True
        self.mock_vector_store.total_vectors = 2
        self.mock_vector_store.similarity_search_with_score.return_value = self.sample_docs

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = RuntimeError("API_KEY_INVALID: The provided API key is expired.")
        mock_gemini_cls.return_value = mock_llm_instance

        res = self.rag.answer_question("What is IntelliAssist AI?")

        self.assertEqual(res["status"], "error")
        self.assertIn("Invalid Google Gemini API key", res["answer"])
        self.assertEqual(len(res["citations"]), 2)


if __name__ == "__main__":
    unittest.main()
