"""Unit tests for DocumentLoader (PDF processing)."""

from io import BytesIO
import unittest
from pypdf import PageObject, PdfWriter
from modules.document_loader import DocumentLoader


class TestPDFLoader(unittest.TestCase):
    """Test suite for PDF document loading and metadata preservation."""

    def setUp(self):
        # Create a sample multi-page PDF in memory
        writer = PdfWriter()
        
        # We can add pages with some text content (or blank pages with text annotations)
        # In pypdf, we can write a simple test PDF
        page1 = PageObject.create_blank_page(width=200, height=200)
        writer.add_page(page1)
        
        stream = BytesIO()
        writer.write(stream)
        self.sample_pdf_bytes = stream.getvalue()

    def test_load_pdf_returns_documents(self):
        docs = DocumentLoader.load_pdf(self.sample_pdf_bytes, filename="sample.pdf")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_name"], "sample.pdf")
        self.assertEqual(docs[0].metadata["file_type"], "pdf")
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[0].metadata["total_pages"], 1)


if __name__ == "__main__":
    unittest.main()
