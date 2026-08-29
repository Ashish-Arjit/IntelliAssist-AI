"""Document loader module for extracting text and metadata from various file formats.

Supports PDF, TXT, and DOCX files.
"""

from io import BytesIO
import os
from typing import BinaryIO, List, Union
from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentLoader:
    """Modular document loader for processing uploaded or local files."""

    @staticmethod
    def load_pdf(
        file_source: Union[str, bytes, BinaryIO], filename: str = "document.pdf"
    ) -> List[Document]:
        """Extract text and metadata from a PDF file per page.

        Args:
            file_source: File path, bytes, or binary file-like object.
            filename: Display name of the file for metadata.

        Returns:
            List of LangChain Document objects with page content and metadata.
        """
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise FileNotFoundError(f"PDF file not found: {file_source}")
            with open(file_source, "rb") as f:
                stream = BytesIO(f.read())
            if not filename or filename == "document.pdf":
                filename = os.path.basename(file_source)
        elif isinstance(file_source, bytes):
            stream = BytesIO(file_source)
        else:
            if hasattr(file_source, "getvalue"):
                stream = BytesIO(file_source.getvalue())
            else:
                stream = BytesIO(file_source.read())
            if hasattr(file_source, "name") and file_source.name:
                filename = file_source.name

        reader = PdfReader(stream)
        total_pages = len(reader.pages)
        documents: List[Document] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            metadata = {
                "source": filename,
                "file_name": filename,
                "file_type": "pdf",
                "page": page_idx,
                "total_pages": total_pages,
                "character_count": len(text),
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
