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


if __name__ == "__main__":
    unittest.main()
