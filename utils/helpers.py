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


def format_conversation_timestamp() -> str:
    """Return formatted current time string for conversation history turns.

    Returns:
        Formatted timestamp string (e.g. "05:32 PM").
    """
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")


def build_chat_message(
    role: str,
    content: str,
    citations: Optional[list] = None,
    intent: Optional[dict] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Build a structured conversation history message entry.

    Args:
        role: Message author role ('user' or 'assistant').
        content: Text content of the message.
        citations: Optional list of document citation dictionaries.
        intent: Optional detected intent dictionary.
        timestamp: Optional formatted time string.

    Returns:
        Structured message dictionary for session state storage.
    """
    return {
        "role": role,
        "content": content,
        "citations": citations or [],
        "intent": intent,
        "timestamp": timestamp or format_conversation_timestamp(),
    }
def clear_chat_history(messages: Optional[list] = None) -> list:
    """Clear and reset session conversation history.

    Args:
        messages: Optional reference to the current messages list.

    Returns:
        Empty list representing a cleared conversation history.
    """
    if messages is not None and isinstance(messages, list):
        messages.clear()
    return []



def export_chat_history(messages: list) -> str:
    """Export conversation history messages into a clean text transcript.

    Args:
        messages: List of message dictionaries from session state.

    Returns:
        Clean plain text transcript string.
    """
    if not messages:
        return "No conversation history available."

    lines = ["# IntelliAssist AI - Conversation Transcript", ""]
    for idx, msg in enumerate(messages, start=1):
        role = "User" if msg.get("role") == "user" else "Assistant"
        ts = msg.get("timestamp", "")
        time_tag = f" [{ts}]" if ts else ""
        lines.append(f"### Turn {idx}: {role}{time_tag}")
        lines.append(msg.get("content", ""))

        citations = msg.get("citations", [])
        if citations:
            lines.append("\nSources Cited:")
            for cit in citations:
                doc = cit.get("file_name", "Document")
                page = f", Page {cit.get('page')}" if cit.get("page") is not None else ""
                score = cit.get("similarity_score", 0.0)
                lines.append(f"  - {doc}{page} (Score: {score:.1%})")
        lines.append("")

    return "\n".join(lines)

