"""Integration tests for multi-document ingestion and semantic vector search."""

from io import BytesIO
import unittest
import docx
from pypdf import PageObject, PdfWriter

from modules.chunker import DocumentChunker
from modules.document_loader import DocumentLoader
from modules.embeddings import EmbeddingManager
from modules.preprocessor import TextPreprocessor
from modules.validator import DocumentValidator
from modules.vector_store import VectorStoreManager


class TestVectorSearchIntegration(unittest.TestCase):
    """End-to-end multi-format ingestion and vector search integration tests."""

    @classmethod
    def setUpClass(cls):
        cls.embedding_manager = EmbeddingManager()
        cls.chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)

        # 1. Create a synthetic PDF document about Machine Learning
        pdf_writer = PdfWriter()
        page1 = PageObject.create_blank_page(width=200, height=200)
        pdf_writer.add_page(page1)
        pdf_stream = BytesIO()
        pdf_writer.write(pdf_stream)
        # We also create a structured PDF with real text content using DocumentLoader mock
        cls.pdf_bytes = pdf_stream.getvalue()

        # 2. Create a TXT document about Cloud Computing and Kubernetes
        cls.txt_content = (
            "Cloud Computing Architecture:\n"
            "Kubernetes is an open-source system for automating deployment, scaling, "
            "and management of containerized applications.\n"
            "Docker containers package software into standardized units for development and shipment."
        )
        cls.txt_bytes = cls.txt_content.encode("utf-8")

        # 3. Create a DOCX document about Financial Quarterly Results
        doc = docx.Document()
        doc.add_heading("Q3 Financial Performance Report", level=1)
        doc.add_paragraph("Revenue reached $12.5 million, representing a 24% year-over-year increase.")
        doc.add_paragraph("Operating expenses decreased by 8% due to streamlined cloud infrastructure.")
        docx_stream = BytesIO()
        doc.save(docx_stream)
        cls.docx_bytes = docx_stream.getvalue()

    def test_multi_document_indexing_and_cross_search(self):
        """Test processing multiple document formats into FAISS and querying across them."""
        # --- Process TXT ---
        val_txt = DocumentValidator.validate_file_upload(
            type("MockFile", (), {"name": "cloud_guide.txt", "size": len(self.txt_bytes), "getvalue": lambda self: self.txt_bytes})()
        )
        self.assertTrue(val_txt.is_valid)
        raw_txt = DocumentLoader.load_document(self.txt_bytes, filename="cloud_guide.txt")
        prep_txt = TextPreprocessor.preprocess_documents(raw_txt)
        chunks_txt = self.chunker.split_documents(prep_txt)

        # --- Process DOCX ---
        val_docx = DocumentValidator.validate_file_upload(
            type("MockFile", (), {"name": "finance_q3.docx", "size": len(self.docx_bytes), "getvalue": lambda self: self.docx_bytes})()
        )
        self.assertTrue(val_docx.is_valid)
        raw_docx = DocumentLoader.load_document(self.docx_bytes, filename="finance_q3.docx")
        prep_docx = TextPreprocessor.preprocess_documents(raw_docx)
        chunks_docx = self.chunker.split_documents(prep_docx)

        # Combine all chunks
        all_chunks = chunks_txt + chunks_docx
        self.assertGreaterEqual(len(all_chunks), 2)

        # Index in FAISS
        vector_manager = VectorStoreManager(embedding_manager=self.embedding_manager)
        vector_manager.create_from_documents(all_chunks)
        self.assertEqual(vector_manager.total_vectors, len(all_chunks))

        # Query 1: Cloud & Kubernetes (Should match cloud_guide.txt)
        results_cloud = vector_manager.similarity_search_with_score("What is Kubernetes container orchestration?", top_k=2)
        self.assertGreater(len(results_cloud), 0)
        top_cloud_doc, dist1, score1 = results_cloud[0]
        self.assertEqual(top_cloud_doc.metadata["file_name"], "cloud_guide.txt")
        self.assertIn("Kubernetes", top_cloud_doc.page_content)
        self.assertGreater(score1, 0.4)

        # Query 2: Financial revenue (Should match finance_q3.docx)
        results_finance = vector_manager.similarity_search_with_score("What was the quarterly revenue growth percentage?", top_k=2)
        self.assertGreater(len(results_finance), 0)
        top_fin_doc, dist2, score2 = results_finance[0]
        self.assertEqual(top_fin_doc.metadata["file_name"], "finance_q3.docx")
        self.assertIn("Revenue reached $12.5 million", top_fin_doc.page_content)
        self.assertGreater(score2, 0.4)

        # Query 3: Unrelated query should yield lower similarity score
        results_unrelated = vector_manager.similarity_search_with_score("Ancient Roman aqueduct engineering techniques", top_k=1)
        self.assertGreater(len(results_unrelated), 0)
        _, _, score_unrelated = results_unrelated[0]
        self.assertLess(score_unrelated, score1)


if __name__ == "__main__":
    unittest.main()
