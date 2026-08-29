"""Unit tests for TextPreprocessor."""

import unittest
from langchain_core.documents import Document
from modules.preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """Test suite for TextPreprocessor."""

    def test_clean_text_whitespace_and_tabs(self):
        raw = "Hello   world!  \t  This is   a test.   "
        expected = "Hello world! This is a test."
        self.assertEqual(TextPreprocessor.clean_text(raw), expected)

    def test_clean_text_excessive_newlines(self):
        raw = "Paragraph 1\n\n\n\n\n\nParagraph 2\r\n\r\n\r\nParagraph 3"
        expected = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        self.assertEqual(TextPreprocessor.clean_text(raw), expected)

    def test_clean_text_preserves_punctuation_and_symbols(self):
        raw = "Invoice #1234: Total = $500.00 (incl. 18% VAT) -> Paid!"
        cleaned = TextPreprocessor.clean_text(raw)
        self.assertEqual(cleaned, raw)

    def test_clean_text_empty_and_spaces(self):
        self.assertEqual(TextPreprocessor.clean_text(""), "")
        self.assertEqual(TextPreprocessor.clean_text("   \n\n  \t  \n  "), "")
        self.assertEqual(TextPreprocessor.clean_text(None), "")

    def test_preprocess_documents(self):
        docs = [
            Document(
                page_content="  Page 1 text with   extra   spaces.  \n\n\n\nMore on page 1.  ",
                metadata={"page": 1, "source": "doc.pdf"},
            ),
            Document(
                page_content="   \n\n   ",
                metadata={"page": 2, "source": "doc.pdf"},
            ),
        ]

        cleaned_docs = TextPreprocessor.preprocess_documents(docs, drop_empty=True)
        self.assertEqual(len(cleaned_docs), 1)
        self.assertEqual(cleaned_docs[0].metadata["page"], 1)
        self.assertTrue(cleaned_docs[0].metadata["preprocessed"])
        self.assertEqual(
            cleaned_docs[0].page_content,
            "Page 1 text with extra spaces.\n\nMore on page 1.",
        )


if __name__ == "__main__":
    unittest.main()
