"""Document validation module for checking file types, integrity, and extraction results.

Provides user-friendly validation error messages and prevents processing failures.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Set
from langchain_core.documents import Document


@dataclass
class ValidationResult:
    """Represents the outcome of a document validation check."""

    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class DocumentValidator:
    """Validator for file uploads and extracted document content."""

    ALLOWED_EXTENSIONS: Set[str] = {"pdf", "txt", "docx"}
    DEFAULT_MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB

    @classmethod
    def validate_file_upload(
        cls,
        uploaded_file: Any,
        max_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None,
    ) -> ValidationResult:
        """Validate an uploaded file object or path before processing.

        Args:
            uploaded_file: Streamlit UploadedFile, file-like object, or filepath string.
            max_size_bytes: Optional maximum allowed size in bytes.
            allowed_extensions: Optional set of allowed extensions.

        Returns:
            ValidationResult indicating whether the file is valid.
        """
        if uploaded_file is None:
            return ValidationResult(
                is_valid=False,
                error_message="No document uploaded. Please upload a file to proceed.",
            )

        allowed_exts = allowed_extensions or cls.ALLOWED_EXTENSIONS
        max_size = max_size_bytes or cls.DEFAULT_MAX_FILE_SIZE_BYTES

        # Determine filename and size
        filename = ""
        size = 0

        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"The specified file does not exist: {uploaded_file}",
                )
            filename = os.path.basename(uploaded_file)
            size = os.path.getsize(uploaded_file)
        else:
            filename = getattr(uploaded_file, "name", "") or "document"
            size = getattr(uploaded_file, "size", None)
            if size is None and hasattr(uploaded_file, "getvalue"):
                size = len(uploaded_file.getvalue())

        # Check extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if not ext or ext not in allowed_exts:
            supported_str = ", ".join(f".{e}" for e in sorted(allowed_exts))
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Unsupported file format '{ext}'. "
                    f"Please upload a supported document ({supported_str})."
                ),
                details={"filename": filename, "extension": ext},
            )

        # Check empty file
        if size == 0:
            return ValidationResult(
                is_valid=False,
                error_message="The uploaded file is empty (0 bytes). Please upload a valid document.",
                details={"filename": filename, "size_bytes": 0},
            )

        # Check max size
        if size > max_size:
            max_mb = max_size / (1024 * 1024)
            size_mb = size / (1024 * 1024)
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.2f} MB)."
                ),
                details={"filename": filename, "size_bytes": size, "max_bytes": max_size},
            )

        return ValidationResult(
            is_valid=True,
            details={"filename": filename, "extension": ext, "size_bytes": size},
        )

    @classmethod
    def validate_extracted_content(
        cls,
        documents: List[Document],
        filename: Optional[str] = None,
    ) -> ValidationResult:
        """Validate extracted LangChain Document content after parsing.

        Args:
            documents: List of LangChain Documents.
            filename: Optional filename for contextual messages.

        Returns:
            ValidationResult indicating whether valid text was extracted.
        """
        if not documents:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Failed to extract text from '{filename or 'the document'}'. "
                    "The document contains no readable pages or text."
                ),
            )

        total_text_length = sum(len((doc.page_content or "").strip()) for doc in documents)

        if total_text_length == 0:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"No readable text found in '{filename or 'the document'}'. "
                    "If this is a scanned PDF, optical character recognition (OCR) may be required."
                ),
                details={"total_pages": len(documents), "total_text_length": 0},
            )

        return ValidationResult(
            is_valid=True,
            details={
                "total_documents": len(documents),
                "total_characters": total_text_length,
            },
        )
