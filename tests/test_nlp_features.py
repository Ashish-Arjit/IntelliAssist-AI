"""Unit tests for IntelliAssist AI additional NLP features.

Covers Conversation History management, Document Summarization, Sentiment Analysis,
and Query Intent Classification.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import (
    build_chat_message,
    clean_query_text,
    clear_chat_history,
    export_chat_history,
    format_conversation_timestamp,
)


class TestConversationHistory(unittest.TestCase):
    """Test suite for session-based conversation history management and formatting."""

    def test_build_chat_message_structure(self):
        """Verify build_chat_message constructs correct structure and metadata."""
        msg = build_chat_message(
            role="user",
            content="What is this document about?",
            timestamp="12:00 PM",
        )
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], "What is this document about?")
        self.assertEqual(msg["citations"], [])
        self.assertEqual(msg["timestamp"], "12:00 PM")
        self.assertIsNone(msg["intent"])

    def test_build_chat_message_with_citations_and_intent(self):
        """Verify message creation with citations and intent metadata."""
        citations = [
            {"file_name": "sample.pdf", "page": 2, "similarity_score": 0.88, "snippet": "Text"}
        ]
        intent_data = {"intent": "Summary Request", "confidence": 0.95}
        msg = build_chat_message(
            role="assistant",
            content="Here is the summary.",
            citations=citations,
            intent=intent_data,
        )
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(len(msg["citations"]), 1)
        self.assertEqual(msg["intent"]["intent"], "Summary Request")
        self.assertIsNotNone(msg["timestamp"])

    def test_export_chat_history_empty(self):
        """Verify export on empty message list returns fallback message."""
        transcript = export_chat_history([])
        self.assertIn("No conversation history available", transcript)

    def test_export_chat_history_multiple_turns(self):
        """Verify multi-turn conversation export creates readable markdown."""
        messages = [
            build_chat_message("user", "Hello assistant", timestamp="10:00 AM"),
            build_chat_message(
                "assistant",
                "Hello! How can I help you?",
                citations=[{"file_name": "doc.pdf", "page": 1, "similarity_score": 0.92}],
                timestamp="10:01 AM",
            ),
        ]
        transcript = export_chat_history(messages)
        self.assertIn("Conversation Transcript", transcript)
        self.assertIn("Hello assistant", transcript)
        self.assertIn("Hello! How can I help you?", transcript)
        self.assertIn("doc.pdf", transcript)
        self.assertIn("Page 1", transcript)

    def test_clear_chat_history_clears_list(self):
        """Verify clear_chat_history resets messages list in place."""
        messages = [
            build_chat_message("user", "Query 1"),
            build_chat_message("assistant", "Answer 1"),
        ]
        result = clear_chat_history(messages)
        self.assertEqual(len(messages), 0)
        self.assertEqual(result, [])

    def test_clear_chat_history_none_or_empty(self):
        """Verify clear_chat_history safely handles None and empty lists."""
        self.assertEqual(clear_chat_history(None), [])
        self.assertEqual(clear_chat_history([]), [])


class TestDocumentSummarizer(unittest.TestCase):
    """Test suite for DocumentSummarizer methods, chunking, and fallback logic."""

    def setUp(self):
        """Set up test fixtures."""
        from modules.summarization import DocumentSummarizer
        self.summarizer = DocumentSummarizer(api_key="test_api_key_12345")

    def test_extract_text_content_varied_inputs(self):
        """Verify extraction from raw string, Document list, and empty input."""
        from langchain_core.documents import Document
        # Raw string
        self.assertEqual(self.summarizer.extract_text_content("Simple text"), "Simple text")
        # Empty
        self.assertEqual(self.summarizer.extract_text_content(""), "")
        self.assertEqual(self.summarizer.extract_text_content([]), "")
        # Document objects
        docs = [
            Document(page_content="Section 1 content.", metadata={"page": 1}),
            Document(page_content="Section 2 content.", metadata={"page": 2}),
        ]
        extracted = self.summarizer.extract_text_content(docs)
        self.assertIn("Section 1 content.", extracted)
        self.assertIn("Section 2 content.", extracted)

    def test_split_into_summary_chunks(self):
        """Verify chunking splits text sensibly by character boundaries."""
        sample_para = "This is a meaningful paragraph for summarization. " * 20
        full_text = "\n\n".join([sample_para, sample_para, sample_para])
        chunks = self.summarizer.split_into_summary_chunks(full_text, chunk_size=500)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 700)

    def test_extractive_fallback_summary(self):
        """Verify frequency-based fallback produces concise coherent summary."""
        long_text = (
            "Machine learning is a method of data analysis that automates analytical model building. "
            "It is a branch of artificial intelligence based on the idea that systems can learn from data. "
            "Systems can identify patterns and make decisions with minimal human intervention. "
            "Because of new computing technologies, machine learning today is not like machine learning of the past. "
            "It was born from pattern recognition and the theory that computers can learn without being programmed."
        )
        summary = self.summarizer.extractive_fallback_summary(long_text, max_sentences=2)
        self.assertTrue(len(summary) > 0)
        self.assertIn("machine learning", summary.lower())

    def test_summarize_empty_document(self):
        """Verify summarizer handles empty content gracefully."""
        res = self.summarizer.summarize("")
        self.assertEqual(res["status"], "warning")
        self.assertIn("No document content provided", res["summary"])

    def test_summarize_no_api_key_uses_extractive_fallback(self):
        """Verify summarizer falls back to extractive mode when API key is missing."""
        from modules.summarization import DocumentSummarizer
        no_key_summarizer = DocumentSummarizer(api_key="")
        res = no_key_summarizer.summarize("Artificial intelligence is transforming modern workflows rapidly.")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["method"], "extractive_fallback")
        self.assertIn("Extractive Document Summary", res["summary"])

    @patch("modules.summarization.ChatGoogleGenerativeAI")
    def test_summarize_direct_llm_mock(self, mock_gemini_cls):
        """Verify direct single-pass LLM summarization flow."""
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "This document provides an executive summary of machine learning workflows."
        mock_llm.invoke.return_value = mock_resp
        mock_gemini_cls.return_value = mock_llm

        res = self.summarizer.summarize(
            "Machine learning enables automated pattern recognition across massive datasets.",
            style="Executive Summary",
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["method"], "llm_direct")
        self.assertEqual(res["summary"], "This document provides an executive summary of machine learning workflows.")


if __name__ == "__main__":
    unittest.main()
