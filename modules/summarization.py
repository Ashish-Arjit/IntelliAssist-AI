"""Document Summarization module for IntelliAssist AI.

Generates concise, coherent summaries of uploaded document content using
Google Gemini LLM with intelligent handling for large documents (map-reduce/chunked condensation)
and an extractive NLP fallback when an API key is unavailable.
"""

from typing import Any, Dict, List, Optional, Union
import os
import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SUMMARY_STYLE,
    MAX_DIRECT_SUMMARY_CHARS,
    MISSING_API_KEY_MESSAGE,
    SUMMARY_CHUNK_SIZE,
)


STYLE_PROMPTS = {
    "Executive Summary": (
        "Provide a professional Executive Summary of the following document content. "
        "Highlight the primary objective, major topics discussed, and overarching conclusions. "
        "Keep the summary structured, polished, and direct."
    ),
    "Key Points / Bullet Points": (
        "Extract the most important Key Takeaways and Core Insights from the following document content. "
        "Format your answer as clear, actionable bullet points with bold sub-headers where appropriate."
    ),
    "Comprehensive Overview": (
        "Provide a detailed, comprehensive overview of the following document content. "
        "Organize the summary into clear thematic sections covering all essential aspects, "
        "key figures, methodologies, and findings presented in the text."
    ),
}


class DocumentSummarizer:
    """Document summarizer supporting multi-style LLM summarization and large-document handling."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.2,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        api_key: Optional[str] = None,
    ):
        """Initialize the DocumentSummarizer.

        Args:
            model_name: Google Gemini model identifier.
            temperature: Sampling temperature (lower = more focused).
            max_output_tokens: Maximum output tokens in summary.
            api_key: Optional Google Gemini API key.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def is_api_key_configured(self) -> bool:
        """Check whether a valid Google API key is configured."""
        key = self.api_key
        if not key or not isinstance(key, str):
            return False
        key = key.strip()
        return bool(key and key != "your_google_api_key_here")

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or initialize the Gemini LLM client."""
        if not self.is_api_key_configured():
            raise ValueError(MISSING_API_KEY_MESSAGE)

        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        return self._llm

    @staticmethod
    def extract_text_content(documents: Union[str, List[Document], List[str]]) -> str:
        """Extract and normalize combined text from string or Document objects.

        Args:
            documents: Single text string, list of strings, or list of Document objects.

        Returns:
            Cleaned consolidated text string.
        """
        if not documents:
            return ""

        if isinstance(documents, str):
            return documents.strip()

        extracted_parts: List[str] = []
        for item in documents:
            if isinstance(item, Document):
                content = item.page_content.strip()
            elif isinstance(item, str):
                content = item.strip()
            elif isinstance(item, dict):
                content = item.get("page_content", item.get("text", "")).strip()
            else:
                content = str(item).strip()

            if content:
                extracted_parts.append(content)

        return "\n\n".join(extracted_parts).strip()

    @staticmethod
    def split_into_summary_chunks(text: str, chunk_size: int = SUMMARY_CHUNK_SIZE) -> List[str]:
        """Split text into manageable chunks respecting paragraph and sentence boundaries.

        Args:
            text: Input document text.
            chunk_size: Target maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_length + para_len + 2 <= chunk_size:
                current_chunk.append(para)
                current_length += para_len + 2
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # If single paragraph exceeds chunk size, split by sentences
                if para_len > chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    s_chunk: List[str] = []
                    s_len = 0
                    for s in sentences:
                        if s_len + len(s) + 1 <= chunk_size:
                            s_chunk.append(s)
                            s_len += len(s) + 1
                        else:
                            if s_chunk:
                                chunks.append(" ".join(s_chunk))
                            s_chunk = [s]
                            s_len = len(s)
                    if s_chunk:
                        chunks.append(" ".join(s_chunk))
                else:
                    current_chunk.append(para)
                    current_length = para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks or [text[:chunk_size]]

    @staticmethod
    def extractive_fallback_summary(text: str, max_sentences: int = 5) -> str:
        """Generate an extractive frequency-based summary when LLM is unavailable.

        Args:
            text: Raw input text.
            max_sentences: Maximum number of sentences to select.

        Returns:
            Extractive summary string.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
        if not sentences:
            return text[:400] + "..." if len(text) > 400 else text

        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        # Word frequency weighting
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "have", "are", "was",
            "were", "will", "been", "they", "their", "which", "about", "there", "these",
        }
        word_freq: Dict[str, int] = {}
        for w in words:
            if w not in stop_words:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Score sentences
        sentence_scores: List[tuple[int, float, str]] = []
        for idx, s in enumerate(sentences):
            s_words = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
            score = sum(word_freq.get(w, 0) for w in s_words) / (len(s_words) + 1)
            # Favor earlier sentences slightly for introductory context
            position_bias = 1.2 if idx < 3 else 1.0
            sentence_scores.append((idx, score * position_bias, s))

        # Pick top sentences and sort by original order
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sentences_ordered = sorted(top_sentences, key=lambda x: x[0])

        return " ".join(s for _, _, s in top_sentences_ordered)

    def summarize(
        self,
        documents: Union[str, List[Document], List[str]],
        style: str = DEFAULT_SUMMARY_STYLE,
    ) -> Dict[str, Any]:
        """Summarize document content using LLM or fallback.

        Args:
            documents: Uploaded documents or plain text.
            style: Requested summary style ("Executive Summary", "Key Points", "Comprehensive Overview").

        Returns:
            Dictionary containing summary result, metadata, and status.
        """
        raw_text = self.extract_text_content(documents)

        # 1. Validation: Empty Content
        if not raw_text or not raw_text.strip():
            return {
                "summary": "No document content provided. Please upload a valid document to generate a summary.",
                "style": style,
                "method": "none",
                "original_length": 0,
                "summary_length": 0,
                "chunks_processed": 0,
                "status": "warning",
                "error": None,
            }

        style_instruction = STYLE_PROMPTS.get(style, STYLE_PROMPTS[DEFAULT_SUMMARY_STYLE])

        # 2. Check API key configuration
        if not self.is_api_key_configured():
            fallback_text = self.extractive_fallback_summary(raw_text)
            return {
                "summary": (
                    f"**Extractive Document Summary (Offline/Fallback Mode):**\n\n{fallback_text}\n\n"
                    f"*Note: Configure a valid Google Gemini API key to enable AI-synthesized {style}.*"
                ),
                "style": style,
                "method": "extractive_fallback",
                "original_length": len(raw_text),
                "summary_length": len(fallback_text),
                "chunks_processed": 1,
                "status": "success",
                "error": None,
            }

        # 3. Handle Direct vs. Large Document Map-Reduce
        try:
            llm = self.get_llm()

            if len(raw_text) <= MAX_DIRECT_SUMMARY_CHARS:
                # Direct single-pass summarization
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert document summarization assistant for IntelliAssist AI. "
                        "Produce high-quality, grounded summaries based solely on the provided document.",
                    ),
                    (
                        "human",
                        f"{style_instruction}\n\nDocument Content:\n{raw_text}\n\nSummary:",
                    ),
                ])
                resp = llm.invoke(prompt.format_messages())
                summary_content = resp.content if hasattr(resp, "content") else str(resp)
                if isinstance(summary_content, list):
                    summary_text = "".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(getattr(b, "text", b))
                        for b in summary_content
                    ).strip()
                else:
                    summary_text = str(summary_content).strip()

                return {
                    "summary": summary_text,
                    "style": style,
                    "method": "llm_direct",
                    "original_length": len(raw_text),
                    "summary_length": len(summary_text),
                    "chunks_processed": 1,
                    "status": "success",
                    "error": None,
                }
            else:
                # Large Document: Chunked Map-Reduce summarization
                chunks = self.split_into_summary_chunks(raw_text, chunk_size=SUMMARY_CHUNK_SIZE)
                intermediate_summaries: List[str] = []

                for idx, chunk in enumerate(chunks, start=1):
                    chunk_prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            "Extract the key facts, findings, and statements from this document section. Keep it concise.",
                        ),
                        ("human", f"Section {idx}/{len(chunks)} Content:\n{chunk}\n\nKey Takeaways:"),
                    ])
                    c_resp = llm.invoke(chunk_prompt.format_messages())
                    c_text = c_resp.content if hasattr(c_resp, "content") else str(c_resp)
                    if isinstance(c_text, list):
                        c_str = "".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(getattr(b, "text", b))
                            for b in c_text
                        ).strip()
                    else:
                        c_str = str(c_text).strip()
                    intermediate_summaries.append(f"[Section {idx} Summary]:\n{c_str}")

                combined_intermediate = "\n\n".join(intermediate_summaries)

                # Final synthesis pass
                final_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert document summarization assistant for IntelliAssist AI. "
                        "Synthesize the section summaries into a single cohesive, unified document summary.",
                    ),
                    (
                        "human",
                        f"{style_instruction}\n\nCombined Section Summaries:\n{combined_intermediate}\n\nFinal Unified Summary:",
                    ),
                ])
                final_resp = llm.invoke(final_prompt.format_messages())
                final_content = final_resp.content if hasattr(final_resp, "content") else str(final_resp)
                if isinstance(final_content, list):
                    final_summary_text = "".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(getattr(b, "text", b))
                        for b in final_content
                    ).strip()
                else:
                    final_summary_text = str(final_content).strip()

                return {
                    "summary": final_summary_text,
                    "style": style,
                    "method": "llm_map_reduce",
                    "original_length": len(raw_text),
                    "summary_length": len(final_summary_text),
                    "chunks_processed": len(chunks),
                    "status": "success",
                    "error": None,
                }

        except Exception as e:
            # On LLM error, fall back gracefully to extractive summary
            fallback = self.extractive_fallback_summary(raw_text)
            return {
                "summary": (
                    f"**Extractive Document Summary (LLM Fallback):**\n\n{fallback}\n\n"
                    f"*(LLM generation encountered an issue: {type(e).__name__}. Extractive summary provided.)*"
                ),
                "style": style,
                "method": "extractive_fallback",
                "original_length": len(raw_text),
                "summary_length": len(fallback),
                "chunks_processed": 1,
                "status": "warning",
                "error": str(e),
            }
