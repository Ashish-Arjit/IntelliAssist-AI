"""Unit tests for DocumentValidator."""

from io import BytesIO
import unittest
from langchain_core.documents import Document
from modules.validator import DocumentValidator


class DummyUploadedFile:
    """Mock Streamlit UploadedFile."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self._data


class TestDocumentValidator(unittest.TestCase):
    """Test suite for DocumentValidator."""

    def test_validate_none_upload(self):
        result = DocumentValidator.validate_file_upload(None)
        self.assertFalse(result.is_valid)
        self.assertIn("No document uploaded", result.error_message)

    def test_validate_unsupported_extension(self):
        fake_file = DummyUploadedFile("image.png", b"\x89PNG\r\n\x1a\n")
        result = DocumentValidator.validate_file_upload(fake_file)
        self.assertFalse(result.is_valid)
        self.assertIn("Unsupported file format", result.error_message)

    def test_validate_empty_file(self):
        fake_file = DummyUploadedFile("empty.txt", b"")
        result = DocumentValidator.validate_file_upload(fake_file)
        self.assertFalse(result.is_valid)
        self.assertIn("0 bytes", result.error_message)

    def test_validate_file_size_limit(self):
        large_data = b"a" * 1024
        fake_file = DummyUploadedFile("large.txt", large_data)
        result = DocumentValidator.validate_file_upload(fake_file, max_size_bytes=500)
        self.assertFalse(result.is_valid)
        self.assertIn("exceeds maximum allowed limit", result.error_message)

    def test_validate_valid_file(self):
        fake_file = DummyUploadedFile("document.pdf", b"%PDF-1.4...")
        result = DocumentValidator.validate_file_upload(fake_file)
        self.assertTrue(result.is_valid)

    def test_validate_extracted_content_empty_docs(self):
        result = DocumentValidator.validate_extracted_content([], filename="test.pdf")
        self.assertFalse(result.is_valid)
        self.assertIn("Failed to extract text", result.error_message)

    def test_validate_extracted_content_whitespace_only(self):
        docs = [Document(page_content="   \n\n\t  ", metadata={})]
        result = DocumentValidator.validate_extracted_content(docs, filename="scanned.pdf")
        self.assertFalse(result.is_valid)
        self.assertIn("No readable text found", result.error_message)

    def test_validate_extracted_content_valid(self):
        docs = [Document(page_content="This is valid content.", metadata={})]
        result = DocumentValidator.validate_extracted_content(docs, filename="valid.pdf")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.details["total_characters"], len("This is valid content."))


if __name__ == "__main__":
    unittest.main()
