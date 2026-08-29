"""Unit tests for DocumentChunker."""

import unittest
from langchain_core.documents import Document
from modules.chunker import DocumentChunker


class TestDocumentChunker(unittest.TestCase):
    """Test suite for DocumentChunker."""

    def test_chunking_preserves_metadata(self):
        sample_doc = Document(
            page_content="Word " * 500,  # ~2500 characters
            metadata={"source": "report.pdf", "page": 1, "file_type": "pdf"},
        )
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.split_documents([sample_doc])

        self.assertGreater(len(chunks), 1)
        for idx, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk.metadata["source"], "report.pdf")
            self.assertEqual(chunk.metadata["page"], 1)
            self.assertEqual(chunk.metadata["file_type"], "pdf")
            self.assertEqual(chunk.metadata["chunk_index"], idx)
            self.assertEqual(chunk.metadata["total_chunks"], len(chunks))
            self.assertLessEqual(len(chunk.page_content), 550)  # within size limit

    def test_chunking_empty_document(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.split_documents([])
        self.assertEqual(chunks, [])

        text_chunks = chunker.split_text("   ")
        self.assertEqual(text_chunks, [])

    def test_invalid_parameters_raise_error(self):
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=0)

        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=500, chunk_overlap=500)

        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=500, chunk_overlap=-10)


if __name__ == "__main__":
    unittest.main()
