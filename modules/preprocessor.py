"""Text preprocessing module for cleaning and normalizing document text.

Performs non-destructive cleaning, whitespace normalization, and line reduction
while strictly preserving punctuation, numbers, and structural meaning.
"""

import re
import unicodedata
from typing import List
from langchain_core.documents import Document


class TextPreprocessor:
    """Provides methods for cleaning and normalizing text and LangChain Documents."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize a raw text string.

        - Standardizes line endings (\r\n and \r -> \n)
        - Normalizes Unicode characters (NFKC)
        - Replaces non-breaking spaces and irregular tabs
        - Collapses repeated spaces on the same line (multiple spaces -> single space)
        - Strips trailing/leading whitespace per line
        - Collapses 3+ consecutive newlines into 2 newlines (paragraph boundary)
        - Strips overall leading and trailing whitespace

        Args:
            text: Raw input string.

        Returns:
            Cleaned and normalized string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode (standardize accents, composite characters, etc.)
        normalized = unicodedata.normalize("NFKC", text)

        # Standardize line breaks
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # Replace non-breaking spaces and form feeds with standard spaces
        normalized = normalized.replace("\u00a0", " ").replace("\x0c", " ")

        # Process line by line: remove trailing/leading spaces and collapse multiple spaces/tabs
        cleaned_lines = []
        for line in normalized.split("\n"):
            # Replace tabs and multiple horizontal whitespace characters with a single space
            cleaned_line = re.sub(r"[^\S\n]+", " ", line).strip()
            cleaned_lines.append(cleaned_line)

        # Reconstruct text
        processed = "\n".join(cleaned_lines)

        # Collapse 3 or more consecutive newlines down to 2
        processed = re.sub(r"\n{3,}", "\n\n", processed)

        return processed.strip()

    @classmethod
    def preprocess_documents(
        cls, documents: List[Document], drop_empty: bool = True
    ) -> List[Document]:
        """Preprocess a list of LangChain Document objects.

        Args:
            documents: List of LangChain Documents.
            drop_empty: Whether to remove documents that become empty after cleaning.

        Returns:
            List of preprocessed Documents with updated content and metadata.
        """
        processed_docs: List[Document] = []

        for doc in documents:
            original_text = doc.page_content or ""
            cleaned = cls.clean_text(original_text)

            if drop_empty and not cleaned.strip():
                continue

            # Copy existing metadata and record preprocessing info
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["raw_character_count"] = len(original_text)
            metadata["cleaned_character_count"] = len(cleaned)
            metadata["preprocessed"] = True

            processed_docs.append(Document(page_content=cleaned, metadata=metadata))

        return processed_docs
