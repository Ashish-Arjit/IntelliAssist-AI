"""End-to-end pipeline integration tests for IntelliAssist AI."""

from io import BytesIO
import unittest
import docx
from pypdf import PageObject, PdfWriter

from modules.chunker import DocumentChunker
from modules.document_loader import DocumentLoader
from modules.preprocessor import TextPreprocessor
from modules.validator import DocumentValidator


class TestPipelineIntegration(unittest.TestCase):
    """Integration test suite for the complete document ingestion pipeline."""

    def test_full_pipeline_txt(self):
        raw_text = "IntelliAssist AI is a modular smart assistant.\n\n\n\nIt processes documents efficiently."
        txt_bytes = raw_text.encode("utf-8")

        # 1. Validation
        val = DocumentValidator.validate_file_upload(txt_bytes)
        # Note: bytes without filename will fail extension check unless name provided
        val = DocumentValidator.validate_file_upload(
            type("MockFile", (), {"name": "doc.txt", "size": len(txt_bytes), "getvalue": lambda self: txt_bytes})()
        )
        self.assertTrue(val.is_valid)

        # 2. Loading
        docs = DocumentLoader.load_document(txt_bytes, filename="doc.txt")
        self.assertEqual(len(docs), 1)

        # 3. Content Validation
        c_val = DocumentValidator.validate_extracted_content(docs, filename="doc.txt")
        self.assertTrue(c_val.is_valid)

        # 4. Preprocessing
        preprocessed = TextPreprocessor.preprocess_documents(docs)
        self.assertEqual(len(preprocessed), 1)
        self.assertEqual(
            preprocessed[0].page_content,
            "IntelliAssist AI is a modular smart assistant.\n\nIt processes documents efficiently.",
        )

        # 5. Chunking
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split_documents(preprocessed)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["file_name"], "doc.txt")
        self.assertEqual(chunks[0].metadata["chunk_index"], 1)

    def test_full_pipeline_docx(self):
        doc = docx.Document()
        doc.add_paragraph("Paragraph 1 with redundant    spaces.")
        doc.add_paragraph("Paragraph 2 content.")
        stream = BytesIO()
        doc.save(stream)
        docx_bytes = stream.getvalue()

        docs = DocumentLoader.load_document(docx_bytes, filename="spec.docx")
        preprocessed = TextPreprocessor.preprocess_documents(docs)
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_documents(preprocessed)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["file_type"], "docx")

    def test_full_pipeline_pdf(self):
        writer = PdfWriter()
        page = PageObject.create_blank_page(width=200, height=200)
        writer.add_page(page)
        stream = BytesIO()
        writer.write(stream)
        pdf_bytes = stream.getvalue()

        docs = DocumentLoader.load_document(pdf_bytes, filename="test.pdf")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_type"], "pdf")


if __name__ == "__main__":
    unittest.main()
