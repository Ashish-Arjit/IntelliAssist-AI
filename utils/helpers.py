"""Helper functions and formatting utilities for IntelliAssist AI."""

import re
from typing import Optional


def format_similarity_score(score: float) -> str:
    """Format a [0.0, 1.0] similarity score as a readable percentage string.

    Args:
        score: Normalized similarity score.

    Returns:
        Formatted percentage string (e.g. "94.2%").
    """
    safe_score = max(0.0, min(1.0, float(score)))
    return f"{safe_score * 100:.1f}%"


def get_score_badge_color(score: float) -> str:
    """Return appropriate badge color code based on similarity score.

    Args:
        score: Normalized similarity score [0.0 - 1.0].

    Returns:
        Hex color code string.
    """
    if score >= 0.75:
        return "#10B981"  # Emerald green (High relevance)
    elif score >= 0.50:
        return "#3B82F6"  # Blue (Moderate relevance)
    elif score >= 0.30:
        return "#F59E0B"  # Amber (Low-to-moderate relevance)
    else:
        return "#94A3B8"  # Slate/Gray (Weak relevance)


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes into human-readable string (KB, MB).

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def clean_query_text(query: Optional[str]) -> str:
    """Sanitize and clean query string.

    Args:
        query: Raw query text.

    Returns:
        Cleaned query text.
    """
    if not query:
        return ""
    # Strip whitespace and collapse redundant spaces
    return re.sub(r"\s+", " ", query).strip()


def truncate_snippet(text: str, max_chars: int = 280) -> str:
    """Truncate long text snippet cleanly with an ellipsis.

    Args:
        text: Input text string.
        max_chars: Maximum character limit.

    Returns:
        Truncated text string.
    """
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def format_citation_label(file_name: str, page: Optional[int] = None, chunk_index: Optional[int] = None) -> str:
    """Create a formatted display label for a citation.

    Args:
        file_name: Document file name.
        page: Optional page number for PDFs.
        chunk_index: Optional chunk index.

    Returns:
        Formatted label string.
    """
    parts = [file_name]
    if page is not None:
        parts.append(f"Page {page}")
    if chunk_index is not None:
        parts.append(f"Chunk #{chunk_index}")
    return " • ".join(parts)

