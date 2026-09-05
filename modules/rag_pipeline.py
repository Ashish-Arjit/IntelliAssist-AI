"""RAG Pipeline module for grounded document question answering and source attribution.

Connects FAISS vector store retrieval with Google Gemini LLM to generate accurate,
grounded responses with full source citations (filename, page numbers, chunk index, scores).
"""

from typing import Any, Dict, List, Optional, Tuple
import os
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MIN_SIMILARITY_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    NO_CONTEXT_FOUND_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    MISSING_API_KEY_MESSAGE,
)
from modules.vector_store import VectorStoreManager


RAG_SYSTEM_PROMPT = """You are IntelliAssist AI, an intelligent, precise, and helpful document question-answering assistant.

Your primary mission is to provide accurate, grounded answers strictly based on the provided document context retrieved from user-uploaded files.

Strict Grounding Rules:
1. Answer the question using ONLY the factual information present in the Context section below.
2. If the answer cannot be found in or directly inferred from the Context, clearly state:
   "{no_context_message}"
3. Do NOT invent, assume, extrapolate, or hallucinate facts, figures, dates, or specifications that are not explicitly supported by the Context.
4. Do NOT act as a general-purpose open-domain chatbot when the question asks about document content not found in the context.
5. If the context contains relevant information, synthesize a clear, well-structured, professional answer.
6. When relevant, reference specific sections, page numbers, or documents mentioned in the Context.
"""

RAG_HUMAN_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) Pipeline for IntelliAssist AI."""

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        api_key: Optional[str] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            vector_store_manager: Optional VectorStoreManager with FAISS index.
            model_name: Gemini model name (e.g., 'gemini-1.5-flash').
            temperature: LLM sampling temperature (low for deterministic grounded answers).
            max_output_tokens: Maximum tokens in generated answer.
            api_key: Optional Google API key (falls back to GOOGLE_API_KEY / GEMINI_API_KEY env).
        """
        self._vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets"):
                        self.api_key = (
                            st.secrets.get("GOOGLE_API_KEY", "")
                            or st.secrets.get("GEMINI_API_KEY", "")
                            or None
                        )
                except Exception:
                    pass
        self._llm: Optional[ChatGoogleGenerativeAI] = None

        self.prompt_template = self._build_prompt_template()

    @property
    def vector_store_manager(self) -> VectorStoreManager:
        """Get or lazily initialize VectorStoreManager."""
        if self._vector_store_manager is None:
            self._vector_store_manager = VectorStoreManager()
        return self._vector_store_manager

    @vector_store_manager.setter
    def vector_store_manager(self, manager: VectorStoreManager) -> None:
        """Set the VectorStoreManager instance."""
        self._vector_store_manager = manager

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Create the grounded ChatPromptTemplate."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT.format(no_context_message=NO_CONTEXT_FOUND_MESSAGE)),
                ("human", RAG_HUMAN_PROMPT),
            ]
        )

    def is_api_key_configured(self) -> bool:
        """Check whether a valid Google API key is available."""
        key = self.api_key
        if not key or not isinstance(key, str):
            return False
        key = key.strip()
        return bool(key and key != "your_google_api_key_here")

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or initialize the Google Gemini Chat Model instance."""
        if not self.is_api_key_configured():
            raise ValueError(MISSING_API_KEY_MESSAGE)

        if self._llm is None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini LLM ({self.model_name}): {str(e)}") from e

        return self._llm

    @staticmethod
    def format_context_from_documents(documents: List[Tuple[Document, float, float]]) -> str:
        """Format retrieved documents and metadata into a structured Context string for the LLM.

        Args:
            documents: List of tuples (Document, raw_distance, similarity_score).

        Returns:
            Formatted context string.
        """
        if not documents:
            return "No relevant document context found."

        context_blocks: List[str] = []
        for rank, (doc, _, score) in enumerate(documents, start=1):
            meta = doc.metadata or {}
            source_name = meta.get("file_name", meta.get("source", "Document"))
            page_info = f", Page {meta.get('page')}" if meta.get("page") is not None else ""
            chunk_info = f", Chunk {meta.get('chunk_index')}" if meta.get("chunk_index") is not None else ""

            header = f"[Source #{rank}: {source_name}{page_info}{chunk_info} | Relevance: {score:.1%}]"
            block = f"{header}\n{doc.page_content.strip()}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    @staticmethod
    def extract_citations(documents: List[Tuple[Document, float, float]]) -> List[Dict[str, Any]]:
        """Extract structured source citation items from retrieved document tuples.

        Args:
            documents: List of tuples (Document, raw_distance, similarity_score).

        Returns:
            List of citation dictionaries containing metadata and snippet.
        """
        citations: List[Dict[str, Any]] = []
        for rank, (doc, raw_distance, score) in enumerate(documents, start=1):
            meta = doc.metadata or {}
            source_file = meta.get("file_name", meta.get("source", "Uploaded Document"))
            page = meta.get("page", None)
            chunk_idx = meta.get("chunk_index", rank)
            total_chunks = meta.get("total_chunks", None)

            citations.append({
                "rank": rank,
                "file_name": source_file,
                "page": page,
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "similarity_score": score,
                "raw_distance": raw_distance,
                "snippet": doc.page_content.strip(),
            })

        return citations

    def retrieve_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float, float]]:
        """Retrieve relevant document chunks with relevance scores from FAISS.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.
            score_threshold: Optional minimum relevance score filter.

        Returns:
            List of tuples (Document, raw_distance, similarity_score).
        """
        if not query or not query.strip():
            return []

        if not self.vector_store_manager.is_initialized or self.vector_store_manager.total_vectors == 0:
            return []

        return self.vector_store_manager.similarity_search_with_score(
            query=query.strip(),
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute complete RAG flow: Retrieval -> Context Formatting -> LLM Grounded Answer -> Citations.

        Args:
            question: User's natural language question.
            top_k: Maximum number of chunks to retrieve.
            score_threshold: Optional similarity threshold.

        Returns:
            Dictionary containing 'answer', 'citations', 'retrieved_chunks_count', and status.
        """
        cleaned_question = (question or "").strip()

        # 1. Validation: Empty Query
        if not cleaned_question:
            return {
                "answer": "Please enter a valid question to search your documents.",
                "citations": [],
                "retrieved_chunks_count": 0,
                "status": "warning",
                "error": None,
            }

        # 2. Validation: Vector Store Initialized
        if not self.vector_store_manager.is_initialized or self.vector_store_manager.total_vectors == 0:
            return {
                "answer": NO_DOCUMENTS_MESSAGE,
                "citations": [],
                "retrieved_chunks_count": 0,
                "status": "warning",
                "error": None,
            }

        # 3. Validation: LLM API Key
        if not self.is_api_key_configured():
            return {
                "answer": MISSING_API_KEY_MESSAGE,
                "citations": [],
                "retrieved_chunks_count": 0,
                "status": "error",
                "error": "MISSING_API_KEY",
            }

        # 4. Retrieval: FAISS Similarity Search
        effective_threshold = (
            score_threshold if score_threshold is not None else DEFAULT_MIN_SIMILARITY_THRESHOLD
        )
        try:
            retrieved_docs = self.retrieve_context(
                query=cleaned_question,
                top_k=top_k,
                score_threshold=effective_threshold,
            )
        except Exception as ret_err:
            return {
                "answer": f"Retrieval failed: {str(ret_err)}",
                "citations": [],
                "retrieved_chunks_count": 0,
                "status": "error",
                "error": str(ret_err),
            }

        # 5. Check if any context retrieved
        if not retrieved_docs:
            return {
                "answer": NO_CONTEXT_FOUND_MESSAGE,
                "citations": [],
                "retrieved_chunks_count": 0,
                "status": "no_context",
                "error": None,
            }

        citations = self.extract_citations(retrieved_docs)
        formatted_context = self.format_context_from_documents(retrieved_docs)

        # 6. LLM Generation
        try:
            llm = self.get_llm()
            messages = self.prompt_template.format_messages(
                context=formatted_context,
                question=cleaned_question,
            )
            response = llm.invoke(messages)
            raw_answer = response.content if hasattr(response, "content") else str(response)

            # Ensure string output
            if isinstance(raw_answer, list):
                text_parts = []
                for b in raw_answer:
                    if isinstance(b, dict):
                        text_parts.append(b.get("text", str(b)))
                    elif hasattr(b, "text"):
                        text_parts.append(str(getattr(b, "text")))
                    else:
                        text_parts.append(str(b))
                answer_text = "".join(text_parts).strip()
            else:
                answer_text = str(raw_answer).strip()


            return {
                "answer": answer_text,
                "citations": citations,
                "retrieved_chunks_count": len(retrieved_docs),
                "status": "success",
                "error": None,
            }

        except Exception as llm_err:
            error_msg = str(llm_err)
            # Friendly formatting for common API errors
            if "API_KEY_INVALID" in error_msg or "invalid api key" in error_msg.lower():
                user_msg = "Invalid Google Gemini API key provided. Please verify your GOOGLE_API_KEY."
            elif "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                user_msg = "Google Gemini API rate limit or quota exceeded. Please try again in a moment."
            else:
                user_msg = f"LLM Generation error: {type(llm_err).__name__}. Please check your API configuration."

            return {
                "answer": user_msg,
                "citations": citations,
                "retrieved_chunks_count": len(retrieved_docs),
                "status": "error",
                "error": error_msg,
            }
