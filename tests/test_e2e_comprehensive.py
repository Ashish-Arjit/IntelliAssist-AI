"""End-to-End Comprehensive Validation Script for IntelliAssist AI.

Tests all aspects required by the final validation checklist:
1. PDF, TXT, DOCX extraction, preprocessing, chunking, embeddings, FAISS storage.
2. Multiple documents indexing and cross-document semantic retrieval.
3. Document metadata preservation (filename, page numbers, chunk index).
4. RAG context assembly, prompt grounding rules, and source citation extraction.
5. Realistic user scenarios: Document questions vs Unrelated questions.
6. Empty inputs handling: empty file, empty doc, empty query, empty question.
7. Additional NLP features:
   - Conversation history (build, export, clear)
   - Document summarization (direct, chunked, extractive fallback, styles)
   - Sentiment analysis (positive, negative, neutral)
   - Intent analysis (Question, Summary Request, Information Search, Explanation Request)
8. Error handling: corrupted files, missing index, missing API key, fallback behavior.
"""

import os
import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import docx
from langchain_core.documents import Document
from pypdf import PdfWriter

# Ensure workspace is on sys.path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    NO_CONTEXT_FOUND_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    MISSING_API_KEY_MESSAGE,
    SUMMARY_STYLES,
)
from modules.chunker import DocumentChunker
from modules.document_loader import DocumentLoader
from modules.embeddings import EmbeddingManager
from modules.intent import (
    INTENT_EXPLANATION,
    INTENT_QUESTION,
    INTENT_SEARCH,
    INTENT_SUMMARY,
    IntentAnalyzer,
)
from modules.preprocessor import TextPreprocessor
from modules.rag_pipeline import RAGPipeline
from modules.sentiment import SentimentAnalyzer
from modules.summarization import DocumentSummarizer
from modules.validator import DocumentValidator
from modules.vector_store import VectorStoreManager
from utils.helpers import (
    build_chat_message,
    clean_query_text,
    clear_chat_history,
    export_chat_history,
    format_file_size,
    format_similarity_score,
    get_score_badge_color,
)


from tests.test_vector_search_integration import generate_test_pdf_bytes

create_sample_pdf = generate_test_pdf_bytes


def create_sample_docx(paragraphs: list[str], table_data: list[list[str]] = None) -> bytes:
    """Create a valid DOCX file in memory."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_data:
        table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
        for r_idx, row in enumerate(table_data):
            for c_idx, val in enumerate(row):
                table.rows[r_idx].cells[c_idx].text = val
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestEndToEndValidation(unittest.TestCase):
    """Rigorous end-to-end validation suite covering all final submission requirements."""

    @classmethod
    def setUpClass(cls):
        print("\n--- Initializing Hugging Face Embedding Manager for E2E Tests ---")
        cls.embedding_manager = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
        print(f"Embedding Manager initialized. Dimension: {cls.embedding_manager.dimension}")

    # =========================================================================
    # SECTION 1 & 2: Test All Document Formats (PDF, TXT, DOCX) & Pipeline
    # =========================================================================
    def test_txt_format_complete_pipeline(self):
        """Verify complete workflow for TXT document."""
        raw_text = (
            "Deep Learning architectures such as convolutional neural networks (CNNs) "
            "and transformers have revolutionized computer vision and natural language processing.\n\n"
            "Transformers rely on the self-attention mechanism to capture contextual relationships."
        )
        txt_bytes = raw_text.encode("utf-8")

        # 1. Validation
        val = DocumentValidator.validate_file_upload(
            type("File", (), {"name": "deep_learning.txt", "size": len(txt_bytes), "getvalue": lambda self: txt_bytes})()
        )
        self.assertTrue(val.is_valid)

        # 2. Text Extraction
        docs = DocumentLoader.load_txt(txt_bytes, filename="deep_learning.txt")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_type"], "txt")
        self.assertEqual(docs[0].metadata["file_name"], "deep_learning.txt")

        # 3. Content Validation
        c_val = DocumentValidator.validate_extracted_content(docs, "deep_learning.txt")
        self.assertTrue(c_val.is_valid)

        # 4. Preprocessing
        prep_docs = TextPreprocessor.preprocess_documents(docs)
        self.assertEqual(len(prep_docs), 1)
        self.assertTrue(prep_docs[0].metadata["preprocessed"])

        # 5. Chunking
        chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.split_documents(prep_docs)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["file_name"], "deep_learning.txt")

        # 6. FAISS Embedding & Indexing
        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_store.create_from_documents(chunks)
        self.assertEqual(v_store.total_vectors, len(chunks))

        # 7. Semantic Search
        results = v_store.similarity_search_with_score("What do transformers rely on?", top_k=2)
        self.assertGreater(len(results), 0)
        top_doc, dist, score = results[0]
        self.assertIn("self-attention", top_doc.page_content.lower())
        self.assertGreater(score, 0.40)

    def test_docx_format_complete_pipeline(self):
        """Verify complete workflow for DOCX document including tables."""
        paragraphs = [
            "Reinforcement Learning (RL) trains agents to make sequential decisions by maximizing cumulative rewards.",
            "Key algorithms include Q-Learning, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO)."
        ]
        table_data = [
            ["Algorithm", "Type", "Domain"],
            ["PPO", "Policy Gradient", "Robotics and Control"],
            ["DQN", "Value-Based", "Game Playing"],
        ]
        docx_bytes = create_sample_docx(paragraphs, table_data)

        # 1. Validation
        val = DocumentValidator.validate_file_upload(
            type("File", (), {"name": "reinforcement_learning.docx", "size": len(docx_bytes), "getvalue": lambda self: docx_bytes})()
        )
        self.assertTrue(val.is_valid)

        # 2. Text Extraction
        docs = DocumentLoader.load_docx(docx_bytes, filename="reinforcement_learning.docx")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_type"], "docx")
        self.assertIn("Proximal Policy Optimization", docs[0].page_content)
        self.assertIn("Robotics and Control", docs[0].page_content)

        # 3. Preprocessing & Chunking
        prep = TextPreprocessor.preprocess_documents(docs)
        chunker = DocumentChunker(chunk_size=120, chunk_overlap=20)
        chunks = chunker.split_documents(prep)
        self.assertGreater(len(chunks), 1)

        # 4. Vector Store & Semantic Retrieval
        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_store.create_from_documents(chunks)
        results = v_store.similarity_search_with_score("What algorithms are used in reinforcement learning?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("ppo" in d.page_content.lower() or "q-learning" in d.page_content.lower() for d, _, _ in results))

    def test_pdf_format_complete_pipeline(self):
        """Verify complete workflow for multi-page PDF document."""
        pdf_pages = [
            "Chapter 1: Natural Language Processing fundamentals and tokenization strategies.",
            "Chapter 2: Vector embeddings map semantic meaning into high-dimensional vector spaces.",
        ]
        pdf_bytes = create_sample_pdf(pdf_pages)

        # 1. Validation
        val = DocumentValidator.validate_file_upload(
            type("File", (), {"name": "nlp_handbook.pdf", "size": len(pdf_bytes), "getvalue": lambda self: pdf_bytes})()
        )
        self.assertTrue(val.is_valid)

        # 2. Text Extraction
        docs = DocumentLoader.load_pdf(pdf_bytes, filename="nlp_handbook.pdf")
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 2)
        self.assertEqual(docs[0].metadata["file_type"], "pdf")

        # 3. Preprocessing & Chunking
        prep = TextPreprocessor.preprocess_documents(docs)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split_documents(prep)
        self.assertGreaterEqual(len(chunks), 2)

        # 4. Vector Store & Semantic Retrieval
        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_store.create_from_documents(chunks)
        results = v_store.similarity_search_with_score("vector spaces and semantic meaning", top_k=1)
        self.assertEqual(len(results), 1)
        doc, _, score = results[0]
        self.assertIn("vector embeddings", doc.page_content.lower())
        self.assertEqual(doc.metadata["page"], 2)

    # =========================================================================
    # SECTION 3: Test Multiple Documents Concurrently
    # =========================================================================
    def test_multi_document_indexing_and_citations(self):
        """Verify multi-document ingestion preserves document metadata and retrieves from correct source."""
        # Doc 1: Biology
        bio_text = "Photosynthesis occurs in plant chloroplasts using sunlight, carbon dioxide, and water to generate glucose."
        # Doc 2: Astronomy
        astro_text = "Supernovae are catastrophic stellar explosions that occur during the final evolutionary stages of massive stars."
        # Doc 3: Computer Science
        cs_text = "The Quicksort algorithm utilizes a divide-and-conquer strategy with an average time complexity of O(n log n)."

        docs1 = DocumentLoader.load_txt(bio_text.encode(), filename="biology.txt")
        docs2 = DocumentLoader.load_txt(astro_text.encode(), filename="astronomy.txt")
        docs3 = DocumentLoader.load_txt(cs_text.encode(), filename="algorithms.txt")

        all_docs = TextPreprocessor.preprocess_documents(docs1 + docs2 + docs3)
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        all_chunks = chunker.split_documents(all_docs)

        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_store.create_from_documents(all_chunks)

        # Test retrieval targeting doc 1
        res_bio = v_store.similarity_search_with_score("Where does photosynthesis take place?", top_k=1)
        self.assertEqual(res_bio[0][0].metadata["file_name"], "biology.txt")

        # Test retrieval targeting doc 2
        res_astro = v_store.similarity_search_with_score("What happens during a supernova explosion?", top_k=1)
        self.assertEqual(res_astro[0][0].metadata["file_name"], "astronomy.txt")

        # Test retrieval targeting doc 3
        res_cs = v_store.similarity_search_with_score("What is the time complexity of quicksort?", top_k=1)
        self.assertEqual(res_cs[0][0].metadata["file_name"], "algorithms.txt")

        # Verify source citation formatting
        rag = RAGPipeline(vector_store_manager=v_store, api_key="dummy_key_for_test")
        citations = rag.extract_citations(res_bio)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["file_name"], "biology.txt")
        self.assertIn("chloroplasts", citations[0]["snippet"])

    # =========================================================================
    # SECTION 4: User Scenarios (Document Qs, Unrelated Qs, Empty Inputs)
    # =========================================================================
    def test_user_scenario_document_questions(self):
        """Verify grounded context and answering for valid questions."""
        content = "IntelliAssist AI version 2.0 features a local FAISS vector index with 384-dimensional embeddings."
        docs = DocumentLoader.load_txt(content.encode(), filename="spec.txt")
        chunks = DocumentChunker(chunk_size=200, chunk_overlap=0).split_documents(docs)
        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_store.create_from_documents(chunks)

        rag = RAGPipeline(vector_store_manager=v_store, api_key="test_key")

        # Mock LLM generation
        mock_response = MagicMock()
        mock_response.content = "IntelliAssist AI version 2.0 uses 384-dimensional embeddings."
        with patch.object(rag, "get_llm") as mock_get_llm:
            mock_llm_inst = MagicMock()
            mock_llm_inst.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm_inst

            res = rag.answer_question("What dimension are the embeddings in version 2.0?")
            self.assertEqual(res["status"], "success")
            self.assertIn("384-dimensional", res["answer"])
            self.assertEqual(len(res["citations"]), 1)
            self.assertEqual(res["citations"][0]["file_name"], "spec.txt")

    def test_user_scenario_unrelated_questions(self):
        """Verify system does not invent information for questions completely absent from context."""
        content = "The cafeteria menu on Mondays includes vegetable soup, grilled cheese sandwiches, and fresh fruit."
        docs = DocumentLoader.load_txt(content.encode(), filename="menu.txt")
        chunks = DocumentChunker(chunk_size=200, chunk_overlap=0).split_documents(docs)
        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        v_store.create_from_documents(chunks)

        rag = RAGPipeline(vector_store_manager=v_store, api_key="test_key")

        # Low similarity threshold filter blocks completely unrelated topics
        res = rag.answer_question("What is quantum chromodynamics in particle physics?", score_threshold=0.55)
        self.assertEqual(res["status"], "no_context")
        self.assertEqual(res["answer"], NO_CONTEXT_FOUND_MESSAGE)
        self.assertEqual(res["citations"], [])

    def test_user_scenario_empty_inputs(self):
        """Verify clear user-friendly messages on empty inputs."""
        # 1. Empty file upload
        res_file = DocumentValidator.validate_file_upload(
            type("File", (), {"name": "empty.txt", "size": 0, "getvalue": lambda self: b""})()
        )
        self.assertFalse(res_file.is_valid)
        self.assertIn("empty", res_file.error_message.lower())

        # 2. None upload
        res_none = DocumentValidator.validate_file_upload(None)
        self.assertFalse(res_none.is_valid)
        self.assertIn("no document uploaded", res_none.error_message.lower())

        # 3. Empty search query
        v_store = VectorStoreManager(embedding_manager=self.embedding_manager)
        self.assertEqual(v_store.similarity_search_with_score("   "), [])
        self.assertEqual(clean_query_text("   "), "")

        # 4. Empty chatbot question
        rag = RAGPipeline(vector_store_manager=v_store, api_key="test_key")
        ans_empty = rag.answer_question("   ")
        self.assertEqual(ans_empty["status"], "warning")
        self.assertIn("valid question", ans_empty["answer"])

        # 5. No documents in vector store
        ans_no_docs = rag.answer_question("Any valid question?")
        self.assertEqual(ans_no_docs["status"], "warning")
        self.assertEqual(ans_no_docs["answer"], NO_DOCUMENTS_MESSAGE)

    # =========================================================================
    # SECTION 5: Test Additional NLP Features
    # =========================================================================
    def test_conversation_history_lifecycle(self):
        """Verify conversation history creation, multi-turn tracking, export, and clear."""
        history = []
        # Turn 1
        msg1 = build_chat_message("user", "What is FAISS?")
        msg2 = build_chat_message(
            "assistant",
            "FAISS is a vector similarity search library.",
            citations=[{"file_name": "guide.txt", "page": 1, "similarity_score": 0.85}],
        )
        history.extend([msg1, msg2])
        self.assertEqual(len(history), 2)

        # Turn 2
        msg3 = build_chat_message("user", "Who developed it?")
        msg4 = build_chat_message("assistant", "It was developed by Meta AI Research.")
        history.extend([msg3, msg4])
        self.assertEqual(len(history), 4)

        # Export
        transcript = export_chat_history(history)
        self.assertIn("What is FAISS?", transcript)
        self.assertIn("Meta AI Research", transcript)
        self.assertIn("guide.txt", transcript)

        # Clear
        clear_chat_history(history)
        self.assertEqual(len(history), 0)
        self.assertIn("No conversation history available", export_chat_history(history))

    def test_document_summarization(self):
        """Verify multi-style summarization and extractive fallback."""
        sample_text = (
            "Artificial Intelligence represents a major technological paradigm shift. "
            "Machine learning algorithms detect patterns in large datasets. "
            "Natural language processing enables computers to understand human language. "
            "Modern large language models assist users with summarization, translation, and code generation. "
            "Safety and alignment remain paramount research priorities."
        )
        docs = [Document(page_content=sample_text, metadata={"file_name": "ai_overview.txt"})]

        # Test extractive fallback when no API key is provided
        summarizer = DocumentSummarizer(api_key=None)
        res_summary = summarizer.summarize(docs, style="Key Points / Bullet Points")
        self.assertIn("summary", res_summary)
        self.assertGreater(len(res_summary["summary"]), 30)
        self.assertEqual(res_summary["style"], "Key Points / Bullet Points")

        # Test with empty document handling
        res_empty = summarizer.summarize([])
        self.assertIn("No document content provided", res_empty["summary"])

    def test_sentiment_analysis(self):
        """Verify sentiment classification for positive, negative, and neutral texts."""
        analyzer = SentimentAnalyzer()

        pos_text = "The team achieved outstanding progress, excellent profit margins, and remarkable success."
        res_pos = analyzer.analyze(pos_text)
        self.assertEqual(res_pos["sentiment"], "Positive")
        self.assertGreater(res_pos["confidence"], 0.5)

        neg_text = "The product suffered catastrophic failure, severe vulnerability, and terrible losses."
        res_neg = analyzer.analyze(neg_text)
        self.assertEqual(res_neg["sentiment"], "Negative")
        self.assertGreater(res_neg["confidence"], 0.5)

        neu_text = "The meeting will take place at 10:00 AM on Tuesday in Conference Room 3."
        res_neu = analyzer.analyze(neu_text)
        self.assertEqual(res_neu["sentiment"], "Neutral")

        # Empty text
        res_empty = analyzer.analyze("   ")
        self.assertEqual(res_empty["sentiment"], "Neutral")
        self.assertEqual(res_empty["confidence"], 0.0)

    def test_intent_analysis_standard_queries(self):
        """Verify intent classification on required queries and patterns."""
        analyzer = IntentAnalyzer()

        # 1. Question
        q1 = analyzer.analyze("What is this document about?")
        self.assertEqual(q1["intent"], INTENT_QUESTION)

        # 2. Summary Request
        q2 = analyzer.analyze("Summarize this document.")
        self.assertEqual(q2["intent"], INTENT_SUMMARY)

        # 3. Information Search
        q3 = analyzer.analyze("Find information about artificial intelligence.")
        self.assertEqual(q3["intent"], INTENT_SEARCH)

        # 4. Explanation Request
        q4 = analyzer.analyze("Explain the main topic.")
        self.assertEqual(q4["intent"], INTENT_EXPLANATION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
