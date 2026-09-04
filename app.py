"""IntelliAssist AI - Smart Document AI Assistant.

Complete RAG Pipeline with Google Gemini LLM, FAISS Vector Search,
Grounded Answering, and Verified Source Citations.
"""

from typing import Any, Dict, List
import os
import streamlit as st
from langchain_core.documents import Document

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_TOP_K,
    INTENT_CATEGORIES,
    MAX_TOP_K,
    MIN_TOP_K,
    NO_CONTEXT_FOUND_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    MISSING_API_KEY_MESSAGE,
    SUMMARY_STYLES,
)
from modules.chunker import DocumentChunker
from modules.document_loader import DocumentLoader
from modules.embeddings import EmbeddingManager
from modules.intent import IntentAnalyzer
from modules.preprocessor import TextPreprocessor
from modules.rag_pipeline import RAGPipeline
from modules.sentiment import SentimentAnalyzer
from modules.summarization import DocumentSummarizer
from modules.validator import DocumentValidator
from modules.vector_store import VectorStoreManager
from utils.helpers import (
    build_chat_message,
    clean_query_text,
    clear_chat_history,
    export_chat_history,
    format_file_size,
    format_similarity_score,
    get_score_badge_color,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntelliAssist AI - Document RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Modern CSS Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 0.98rem;
            color: #475569;
            margin-bottom: 1.25rem;
        }
        .metric-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            text-align: center;
        }
        .metric-title {
            font-size: 0.75rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #1E293B;
            margin-top: 0.2rem;
        }
        .feature-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.15rem;
            margin-bottom: 0.75rem;
            height: 100%;
        }
        .citation-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #3B82F6;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.88rem;
        }
        .citation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 0.35rem;
        }
        .citation-snippet {
            font-family: monospace;
            font-size: 0.82rem;
            color: #334155;
            background: #FFFFFF;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            border: 1px solid #CBD5E1;
            white-space: pre-wrap;
            line-height: 1.4;
        }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .badge-meta {
            background-color: #F1F5F9;
            color: #475569;
            border: 1px solid #E2E8F0;
        }
        .badge-score {
            color: #FFFFFF;
        }
        .chunk-content {
            background-color: #F8FAFC;
            border: 1px solid #F1F5F9;
            border-radius: 6px;
            padding: 0.85rem;
            margin-top: 0.75rem;
            font-family: monospace;
            font-size: 0.88rem;
            white-space: pre-wrap;
            color: #1E293B;
            line-height: 1.5;
        }
        .stChatMessage {
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Hugging Face embedding model...")
def get_embedding_manager(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingManager:
    """Initialize and cache the Hugging Face EmbeddingManager instance."""
    return EmbeddingManager(model_name=model_name)


@st.cache_resource(show_spinner="Loading Hugging Face sentiment model...")
def get_sentiment_analyzer(model_name: str = DEFAULT_SENTIMENT_MODEL) -> SentimentAnalyzer:
    """Initialize and cache the SentimentAnalyzer instance."""
    return SentimentAnalyzer(model_name=model_name)


@st.cache_resource
def get_intent_analyzer() -> IntentAnalyzer:
    """Initialize and cache the IntentAnalyzer instance."""
    return IntentAnalyzer()


# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_processed_files" not in st.session_state:
    st.session_state["last_processed_files"] = []

if "summary_result" not in st.session_state:
    st.session_state["summary_result"] = None

if "sentiment_result" not in st.session_state:
    st.session_state["sentiment_result"] = None

if "intent_test_result" not in st.session_state:
    st.session_state["intent_test_result"] = None

if "doc_cache_key" not in st.session_state:
    st.session_state["doc_cache_key"] = None

if "cached_vector_manager" not in st.session_state:
    st.session_state["cached_vector_manager"] = None

if "cached_preprocessed_docs" not in st.session_state:
    st.session_state["cached_preprocessed_docs"] = []

if "cached_chunks" not in st.session_state:
    st.session_state["cached_chunks"] = []

if "cached_processing_status" not in st.session_state:
    st.session_state["cached_processing_status"] = []


# ---------------------------------------------------------
# Sidebar Controls & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 IntelliAssist AI")
    st.caption("Document RAG Chatbot & Semantic Search")
    st.divider()

    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select one or more documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload PDF (.pdf), Plain Text (.txt), or Word (.docx) files.",
    )

    st.divider()
    st.subheader("🔑 LLM Configuration")

    # Check for API Key in environment or session
    env_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    custom_api_key = st.text_input(
        "Google Gemini API Key",
        value=env_api_key if env_api_key and env_api_key != "your_google_api_key_here" else "",
        type="password",
        placeholder="Enter API key if not in .env",
        help="Google Gemini API key used to generate grounded RAG answers.",
    )

    active_api_key = custom_api_key.strip() if custom_api_key else env_api_key

    if active_api_key and active_api_key != "your_google_api_key_here":
        st.success("✅ Gemini API Key Configured")
    else:
        st.warning("⚠️ No Gemini API Key found. Add `GOOGLE_API_KEY` to `.env` or input above.")

    st.divider()
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Retrieved Context Chunks (Top-K)",
        min_value=MIN_TOP_K,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
        help="Number of most relevant document chunks supplied to the LLM.",
    )

    st.subheader("⚙️ Chunking Parameters")
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2500,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        help="Maximum character length per chunk.",
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        step=25,
        help="Overlapping characters between adjacent chunks.",
    )

    st.divider()
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #64748B; background: #F8FAFC; padding: 0.6rem; border-radius: 6px; border: 1px solid #E2E8F0;">
            <strong>LLM:</strong> <code>{DEFAULT_LLM_MODEL}</code><br/>
            <strong>Embedding:</strong> <code>{DEFAULT_EMBEDDING_MODEL.split('/')[-1]}</code><br/>
            <strong>Vector Store:</strong> FAISS (In-Memory)<br/>
            <strong>Grounding:</strong> Strict Document Attribution
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history(st.session_state["messages"])
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🤖 IntelliAssist AI — Document RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Ask questions grounded in your uploaded documents with precise source citations & page references."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Welcome Screen (No Documents Uploaded)
# ---------------------------------------------------------
if not uploaded_files:
    # Clear cached document state when all files are removed
    st.session_state["doc_cache_key"] = None
    st.session_state["cached_vector_manager"] = None
    st.session_state["cached_preprocessed_docs"] = []
    st.session_state["cached_chunks"] = []
    st.session_state["cached_processing_status"] = []

    st.info("👋 **Welcome to IntelliAssist AI!** Upload one or more **PDF**, **TXT**, or **DOCX** files in the sidebar to start asking questions grounded in your documents.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📑 Multi-Doc Ingestion</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    Extract and normalize text and page metadata across PDF, TXT, and DOCX files simultaneously.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>⚡ FAISS Semantic Search</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    Dense vector similarity retrieval using Hugging Face embeddings (<code>all-MiniLM-L6-v2</code>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🛡️ Grounded Answers</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    Strict prompt constraints instructing Google Gemini to answer purely from retrieved document context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📌 Source Citations</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    Every answer is linked with verified source documents, PDF page numbers, and relevance scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # ---------------------------------------------------------
    # Ingestion & Vector Index Pipeline
    # ---------------------------------------------------------
    try:
        embedding_manager = get_embedding_manager(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        st.error("Failed to initialize Hugging Face embedding model. Please check dependencies.")
        st.caption(f"Details: {type(e).__name__}: {str(e)}")
        st.stop()

    # Compute current upload cache signature
    current_cache_key = (
        tuple((f.name, getattr(f, "size", 0)) for f in uploaded_files),
        chunk_size,
        chunk_overlap,
    )

    # Check if we can reuse cached documents and vector index
    if (
        st.session_state.get("doc_cache_key") == current_cache_key
        and st.session_state.get("cached_vector_manager") is not None
        and st.session_state["cached_vector_manager"].is_initialized
    ):
        all_preprocessed_docs = st.session_state["cached_preprocessed_docs"]
        all_chunks = st.session_state["cached_chunks"]
        file_processing_status = st.session_state["cached_processing_status"]
        vector_manager = st.session_state["cached_vector_manager"]
    else:
        all_preprocessed_docs: List[Document] = []
        all_chunks: List[Document] = []
        file_processing_status: List[Dict[str, Any]] = []

        with st.spinner("Processing documents, generating embeddings, and building FAISS vector index..."):
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for up_file in uploaded_files:
                file_name = up_file.name
                # 1. Validation
                val_res = DocumentValidator.validate_file_upload(up_file)
                if not val_res.is_valid:
                    file_processing_status.append({
                        "filename": file_name,
                        "status": "error",
                        "message": val_res.error_message,
                        "chunks_count": 0,
                    })
                    continue

                try:
                    # 2. Text Extraction
                    raw_docs = DocumentLoader.load_document(up_file, filename=file_name)
                    
                    # 3. Content Validation
                    c_val = DocumentValidator.validate_extracted_content(raw_docs, filename=file_name)
                    if not c_val.is_valid:
                        file_processing_status.append({
                            "filename": file_name,
                            "status": "warning",
                            "message": c_val.error_message,
                            "chunks_count": 0,
                        })
                        continue

                    # 4. Preprocessing
                    prep_docs = TextPreprocessor.preprocess_documents(raw_docs, drop_empty=True)
                    
                    # 5. Chunking
                    doc_chunks = chunker.split_documents(prep_docs)

                    all_preprocessed_docs.extend(prep_docs)
                    all_chunks.extend(doc_chunks)

                    file_processing_status.append({
                        "filename": file_name,
                        "status": "success",
                        "size_bytes": getattr(up_file, "size", 0),
                        "sections_count": len(prep_docs),
                        "chunks_count": len(doc_chunks),
                    })
                except Exception as doc_err:
                    file_processing_status.append({
                        "filename": file_name,
                        "status": "error",
                        "message": f"Extraction error: {str(doc_err) or type(doc_err).__name__}",
                        "chunks_count": 0,
                    })

            # 6. Build or Update FAISS Vector Store Index
            vector_manager = VectorStoreManager(embedding_manager=embedding_manager)
            if all_chunks:
                try:
                    vector_manager.create_from_documents(all_chunks)
                except Exception as vec_err:
                    st.error("Failed to build FAISS vector index from document chunks.")
                    st.caption(f"Error: {type(vec_err).__name__}")

            # Update cache in session state
            st.session_state["doc_cache_key"] = current_cache_key
            st.session_state["cached_preprocessed_docs"] = all_preprocessed_docs
            st.session_state["cached_chunks"] = all_chunks
            st.session_state["cached_processing_status"] = file_processing_status
            st.session_state["cached_vector_manager"] = vector_manager

    # 7. Initialize RAG Pipeline with active credentials
    rag_pipeline = RAGPipeline(
        vector_store_manager=vector_manager,
        model_name=DEFAULT_LLM_MODEL,
        api_key=active_api_key,
    )

    # ---------------------------------------------------------
    # Ingestion Status & Metrics Bar
    # ---------------------------------------------------------
    successful_files = [f for f in file_processing_status if f["status"] == "success"]
    total_words = sum(len(d.page_content.split()) for d in all_preprocessed_docs)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Uploaded Files</div>
                <div class="metric-value">{len(successful_files)}/{len(uploaded_files)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Sections / Pages</div>
                <div class="metric-value">{len(all_preprocessed_docs)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Words</div>
                <div class="metric-value">{total_words:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Vector Chunks</div>
                <div class="metric-value" style="color: #4F46E5;">{len(all_chunks)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">RAG Pipeline</div>
                <div class="metric-value" style="color: {'#10B981' if vector_manager.is_initialized else '#EF4444'}; font-size: 1.15rem;">
                    {"Ready (Active)" if vector_manager.is_initialized else "Not Ready"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Document Status List expander
    with st.expander("📋 Processed Document Ingestion Details", expanded=False):
        for f_stat in file_processing_status:
            if f_stat["status"] == "success":
                st.markdown(
                    f"✅ **{f_stat['filename']}** — {format_file_size(f_stat.get('size_bytes', 0))} | "
                    f"{f_stat['sections_count']} sections | **{f_stat['chunks_count']} chunks indexed**"
                )
            elif f_stat["status"] == "warning":
                st.warning(f"⚠️ **{f_stat['filename']}**: {f_stat['message']}")
            else:
                st.error(f"❌ **{f_stat['filename']}**: {f_stat['message']}")

    st.write("")

    # ---------------------------------------------------------
    # Main Tabs: Chat Assistant, NLP Features & Vector Inspection
    # ---------------------------------------------------------
    main_tab_chat, main_tab_summary, main_tab_sentiment, main_tab_intent, main_tab_search, main_tab_chunks, main_tab_meta = st.tabs(
        [
            "💬 Document Chatbot",
            "📝 Document Summarization",
            "🎭 Sentiment Analysis",
            "🎯 Intent Analysis",
            "🔍 Semantic Search",
            "🧩 Indexed Chunks",
            "🔍 Metadata & Summary",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: RAG Chatbot Interface
    # ---------------------------------------------------------
    with main_tab_chat:
        st.subheader("💬 Ask Your Documents")
        st.caption("Enter questions below. The assistant retrieves relevant chunks from your documents and synthesizes grounded answers with citations.")

        # Chat controls toolbar
        chat_bar_col1, chat_bar_col2, chat_bar_col3 = st.columns([3, 1, 1])
        with chat_bar_col1:
            msg_count = len(st.session_state["messages"])
            st.caption(f"Session history: **{msg_count}** turn{'s' if msg_count != 1 else ''}")
        with chat_bar_col2:
            if st.button("🗑️ Clear Chat", key="btn_clear_chat_tab", use_container_width=True, disabled=(msg_count == 0)):
                clear_chat_history(st.session_state["messages"])
                st.session_state["messages"] = []
                st.rerun()
        with chat_bar_col3:
            transcript_text = export_chat_history(st.session_state["messages"])
            st.download_button(
                label="📥 Export Chat",
                data=transcript_text,
                file_name="intelliassist_chat_transcript.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=(msg_count == 0),
                key="btn_export_chat_transcript",
            )

        st.divider()

        # Display previous chat messages
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                ts = msg.get("timestamp", "")
                intent_data = msg.get("intent")
                header_parts = []
                if intent_data and intent_data.get("badge"):
                    header_parts.append(f'<span class="badge badge-meta" style="font-size: 0.72rem;">{intent_data["badge"]}</span>')
                if ts:
                    header_parts.append(f'<span style="font-size: 0.75rem; color: #94A3B8;">{ts}</span>')

                if header_parts:
                    st.markdown(
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: -8px; margin-bottom: 4px;">{"".join(header_parts)}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(msg["content"])
                
                # Render source citations if available for assistant messages
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"📌 View Source Citations ({len(citations)} source{'s' if len(citations) > 1 else ''})", expanded=False):
                        for cit in citations:
                            doc_name = cit.get("file_name", "Document")
                            page_num = cit.get("page")
                            chunk_idx = cit.get("chunk_index", 1)
                            score = cit.get("similarity_score", 0.0)
                            badge_col = get_score_badge_color(score)
                            score_pct = format_similarity_score(score)
                            page_text = f"Page {page_num}" if page_num is not None else "Page N/A"
                            
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <div>
                                            <span>📄 <strong>{doc_name}</strong></span>
                                            <span class="badge badge-meta" style="margin-left: 0.5rem;">{page_text}</span>
                                            <span class="badge badge-meta" style="margin-left: 0.25rem;">Chunk #{chunk_idx}</span>
                                        </div>
                                        <div>
                                            <span class="badge badge-score" style="background-color: {badge_col};">
                                                Relevance: {score_pct}
                                            </span>
                                        </div>
                                    </div>
                                    <div class="citation-snippet">{cit.get('snippet', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Chat Input Box
        user_query = st.chat_input("Ask a question about the uploaded documents...")

        if user_query:
            cleaned_q = clean_query_text(user_query)

            # Analyze query intent in real time
            intent_analyzer = get_intent_analyzer()
            intent_info = intent_analyzer.analyze(cleaned_q)

            # Display user message immediately with intent badge
            with st.chat_message("user"):
                st.markdown(
                    f'<div style="margin-top: -8px; margin-bottom: 4px;"><span class="badge badge-meta" style="font-size: 0.72rem;">{intent_info["badge"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(cleaned_q)
            st.session_state["messages"].append(
                build_chat_message(role="user", content=cleaned_q, intent=intent_info)
            )

            # Check API Key before execution
            if not rag_pipeline.is_api_key_configured():
                warning_ans = MISSING_API_KEY_MESSAGE
                with st.chat_message("assistant"):
                    st.error(warning_ans)
                st.session_state["messages"].append(
                    build_chat_message(role="assistant", content=warning_ans)
                )
            elif not vector_manager.is_initialized or vector_manager.total_vectors == 0:
                warning_ans = NO_DOCUMENTS_MESSAGE
                with st.chat_message("assistant"):
                    st.warning(warning_ans)
                st.session_state["messages"].append(
                    build_chat_message(role="assistant", content=warning_ans)
                )
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving document chunks & generating grounded answer..."):
                        rag_response = rag_pipeline.answer_question(
                            question=cleaned_q,
                            top_k=top_k,
                        )

                    answer_text = rag_response.get("answer", NO_CONTEXT_FOUND_MESSAGE)
                    citations = rag_response.get("citations", [])
                    status = rag_response.get("status", "success")

                    if status == "error":
                        st.error(answer_text)
                    elif status == "no_context":
                        st.info(answer_text)
                    else:
                        st.markdown(answer_text)

                    # Render citations for the current response
                    if citations:
                        with st.expander(f"📌 View Source Citations ({len(citations)} source{'s' if len(citations) > 1 else ''})", expanded=True):
                            for cit in citations:
                                doc_name = cit.get("file_name", "Document")
                                page_num = cit.get("page")
                                chunk_idx = cit.get("chunk_index", 1)
                                score = cit.get("similarity_score", 0.0)
                                badge_col = get_score_badge_color(score)
                                score_pct = format_similarity_score(score)
                                page_text = f"Page {page_num}" if page_num is not None else "Page N/A"

                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-header">
                                            <div>
                                                <span>📄 <strong>{doc_name}</strong></span>
                                                <span class="badge badge-meta" style="margin-left: 0.5rem;">{page_text}</span>
                                                <span class="badge badge-meta" style="margin-left: 0.25rem;">Chunk #{chunk_idx}</span>
                                            </div>
                                            <div>
                                                <span class="badge badge-score" style="background-color: {badge_col};">
                                                    Relevance: {score_pct}
                                                </span>
                                            </div>
                                        </div>
                                        <div class="citation-snippet">{cit.get('snippet', '')}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Append assistant response to session messages with citations
                    st.session_state["messages"].append(
                        build_chat_message(
                            role="assistant",
                            content=answer_text,
                            citations=citations,
                        )
                    )

    # ---------------------------------------------------------
    # TAB 2: Document Summarization Interface
    # ---------------------------------------------------------
    with main_tab_summary:
        st.subheader("📝 Document Summarization")
        st.caption("Generate AI-powered summaries of your uploaded documents with multi-style control and large-document handling.")

        if not all_preprocessed_docs:
            st.info("Please upload and process at least one document to generate a summary.")
        else:
            doc_file_names = sorted(list({
                d.metadata.get("file_name", "Document")
                for d in all_preprocessed_docs
                if d.metadata and "file_name" in d.metadata
            }))

            scope_options = ["All Uploaded Documents"] + doc_file_names
            sum_col1, sum_col2 = st.columns([2, 2])
            with sum_col1:
                selected_scope = st.selectbox(
                    "Select Document Scope",
                    options=scope_options,
                    index=0,
                    help="Summarize all documents combined or choose a specific document.",
                    key="select_summary_scope",
                )
            with sum_col2:
                selected_style = st.selectbox(
                    "Summary Style",
                    options=SUMMARY_STYLES,
                    index=0,
                    help="Choose the style and depth of the generated summary.",
                    key="select_summary_style",
                )

            # Filter documents according to selected scope
            if selected_scope == "All Uploaded Documents":
                target_docs = all_preprocessed_docs
            else:
                target_docs = [
                    d for d in all_preprocessed_docs
                    if d.metadata.get("file_name") == selected_scope
                ]

            total_scope_words = sum(len(d.page_content.split()) for d in target_docs)
            total_scope_chars = sum(len(d.page_content) for d in target_docs)

            st.caption(f"Target scope: **{len(target_docs)}** section(s), **{total_scope_words:,}** words (~{total_scope_chars:,} characters)")

            generate_summary_btn = st.button("✨ Generate Document Summary", type="primary", use_container_width=True, key="btn_generate_summary")

            if generate_summary_btn:
                summarizer = DocumentSummarizer(
                    model_name=DEFAULT_LLM_MODEL,
                    api_key=active_api_key,
                )

                with st.spinner(f"Generating {selected_style}..."):
                    summary_output = summarizer.summarize(target_docs, style=selected_style)
                    st.session_state["summary_result"] = summary_output

            # Display generated summary if present in session
            current_sum = st.session_state.get("summary_result")
            if current_sum:
                st.divider()
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Summary Style", current_sum.get("style", "Summary"))
                with stat_col2:
                    mode_label = "AI Synthesized (Gemini)" if "llm" in current_sum.get("method", "") else "Extractive NLP Fallback"
                    st.metric("Generation Method", mode_label)
                with stat_col3:
                    st.metric("Processed Chunks", current_sum.get("chunks_processed", 1))

                st.markdown(
                    f"""
                    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.5rem; margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <h4 style="color: #1E293B; margin-bottom: 0.75rem;">📋 {current_sum.get('style', 'Summary')}</h4>
                        <div style="font-size: 0.95rem; line-height: 1.6; color: #334155; white-space: pre-wrap;">{current_sum.get('summary', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")
                st.download_button(
                    label="📥 Download Summary (.md)",
                    data=current_sum.get("summary", ""),
                    file_name=f"intelliassist_summary_{selected_style.lower().replace(' ', '_').replace('/', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="btn_download_summary",
                )

    # ---------------------------------------------------------
    # TAB 3: Document Sentiment Analysis Interface
    # ---------------------------------------------------------
    with main_tab_sentiment:
        st.subheader("🎭 Document Sentiment Analysis")
        st.caption("Evaluate document sentiment (Positive, Negative, Neutral) and confidence score using Hugging Face NLP.")

        sentiment_source = st.radio(
            "Sentiment Input Source",
            options=["Uploaded Document Content", "Custom Text Excerpt"],
            horizontal=True,
            key="radio_sentiment_source",
        )

        text_for_sentiment = ""
        if sentiment_source == "Uploaded Document Content":
            if not all_preprocessed_docs:
                st.info("Please upload at least one document to analyze its sentiment.")
            else:
                doc_file_names = sorted(list({
                    d.metadata.get("file_name", "Document")
                    for d in all_preprocessed_docs
                    if d.metadata and "file_name" in d.metadata
                }))
                scope_options = ["All Uploaded Documents"] + doc_file_names
                selected_doc = st.selectbox(
                    "Select Document to Analyze",
                    options=scope_options,
                    index=0,
                    key="select_sentiment_doc",
                )

                if selected_doc == "All Uploaded Documents":
                    text_for_sentiment = "\n\n".join(d.page_content for d in all_preprocessed_docs)
                else:
                    text_for_sentiment = "\n\n".join(
                        d.page_content for d in all_preprocessed_docs
                        if d.metadata.get("file_name") == selected_doc
                    )

                with st.expander(f"📄 Preview Selected Text ({len(text_for_sentiment)} chars)", expanded=False):
                    st.text(text_for_sentiment[:2000] + ("..." if len(text_for_sentiment) > 2000 else ""))
        else:
            # Custom text input with demonstration presets
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            with preset_col1:
                if st.button("🌟 Positive Sample", use_container_width=True, key="btn_pos_sample"):
                    st.session_state["custom_sentiment_text"] = "The quarterly performance delivered exceptional results, outstanding revenue growth, and strong operational profits."
            with preset_col2:
                if st.button("⚠️ Negative Sample", use_container_width=True, key="btn_neg_sample"):
                    st.session_state["custom_sentiment_text"] = "The system experienced a severe breakdown, catastrophic failure, and critical data loss."
            with preset_col3:
                if st.button("⚖️ Neutral Sample", use_container_width=True, key="btn_neu_sample"):
                    st.session_state["custom_sentiment_text"] = "The advisory committee will convene on Thursday at 2:00 PM in Conference Room B."

            custom_input = st.text_area(
                "Enter text to analyze:",
                value=st.session_state.get("custom_sentiment_text", ""),
                height=130,
                placeholder="Type or paste any document excerpt to evaluate its emotional tone...",
                key="textarea_sentiment_input",
            )
            text_for_sentiment = custom_input

        analyze_sentiment_btn = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
            key="btn_run_sentiment_analysis",
            disabled=(not text_for_sentiment.strip()),
        )

        if analyze_sentiment_btn:
            sentiment_analyzer = get_sentiment_analyzer()
            with st.spinner("Analyzing text sentiment with Hugging Face model..."):
                s_res = sentiment_analyzer.analyze(text_for_sentiment)
                st.session_state["sentiment_result"] = s_res

        # Display Sentiment Results
        sent_res = st.session_state.get("sentiment_result")
        if sent_res and sent_res.get("status") in ("success", "warning"):
            st.divider()
            sent_label = sent_res.get("sentiment", "Neutral")
            conf_val = float(sent_res.get("confidence", 0.0))
            badge_color = "#10B981" if sent_label == "Positive" else ("#EF4444" if sent_label == "Negative" else "#64748B")
            icon = "🟢" if sent_label == "Positive" else ("🔴" if sent_label == "Negative" else "⚪")

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Overall Sentiment</div>
                        <div class="metric-value" style="color: {badge_color};">{icon} {sent_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with scol2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Confidence Score</div>
                        <div class="metric-value">{conf_val:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with scol3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Model / Pipeline</div>
                        <div class="metric-value" style="font-size: 1rem; color: #475569;">{sent_res.get('method', 'NLP Model')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            score_dist = sent_res.get("scores", {})
            if score_dist:
                st.markdown("**Probability Distribution:**")
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    p_pos = score_dist.get("Positive", 0.0)
                    st.caption(f"Positive: {p_pos:.1%}")
                    st.progress(float(p_pos))
                with p_col2:
                    p_neu = score_dist.get("Neutral", 0.0)
                    st.caption(f"Neutral: {p_neu:.1%}")
                    st.progress(float(p_neu))
                with p_col3:
                    p_neg = score_dist.get("Negative", 0.0)
                    st.caption(f"Negative: {p_neg:.1%}")
                    st.progress(float(p_neg))

    # ---------------------------------------------------------
    # TAB 4: Query Intent Classification Interface
    # ---------------------------------------------------------
    with main_tab_intent:
        st.subheader("🎯 Query Intent Classification")
        st.caption("Identify user query purpose (Question, Summary Request, Information Search, Explanation Request) using NLP classification.")

        st.markdown("**Quick-Test Sample Queries (Click to test):**")
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            if st.button("❓ What is this document about?", use_container_width=True, key="btn_sample_q1"):
                st.session_state["intent_input_text"] = "What is this document about?"
        with tcol2:
            if st.button("📝 Summarize this document.", use_container_width=True, key="btn_sample_q2"):
                st.session_state["intent_input_text"] = "Summarize this document."
        with tcol3:
            if st.button("🔍 Find info about machine learning.", use_container_width=True, key="btn_sample_q3"):
                st.session_state["intent_input_text"] = "Find information about machine learning."
        with tcol4:
            if st.button("💡 Explain the main concept.", use_container_width=True, key="btn_sample_q4"):
                st.session_state["intent_input_text"] = "Explain the main concept in this document."

        current_intent_query = st.text_input(
            "Query to analyze:",
            value=st.session_state.get("intent_input_text", ""),
            placeholder="Type any user query or question...",
            key="input_intent_query_text",
        )

        classify_intent_btn = st.button("🎯 Classify Intent", type="primary", use_container_width=True, key="btn_classify_intent")

        if classify_intent_btn or current_intent_query:
            if not current_intent_query.strip():
                st.warning("Please enter a non-empty query to analyze intent.")
            else:
                intent_analyzer = get_intent_analyzer()
                detected = intent_analyzer.analyze(current_intent_query)
                st.session_state["intent_test_result"] = detected

        intent_res = st.session_state.get("intent_test_result")
        if intent_res and intent_res.get("status") == "success":
            st.divider()
            icol1, icol2, icol3 = st.columns(3)
            with icol1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Detected Intent</div>
                        <div class="metric-value" style="color: #4F46E5;">{intent_res.get('badge', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with icol2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Confidence Score</div>
                        <div class="metric-value">{intent_res.get('confidence', 0.0):.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with icol3:
                matched_kws = ", ".join(intent_res.get('matched_keywords', [])) or "None"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Matched Linguistic Markers</div>
                        <div class="metric-value" style="font-size: 1.05rem; color: #334155;"><code>{matched_kws}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <strong>Intent Description:</strong> {intent_res.get('description', '')}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # TAB 5: Semantic Similarity Search (Preserved)
    # ---------------------------------------------------------
    with main_tab_search:
        st.subheader("🔍 FAISS Semantic Similarity Search")
        st.caption("Query raw vector embeddings to inspect nearest neighbor chunks with distance and relevance metrics.")

        search_col1, search_col2 = st.columns([5, 1])
        with search_col1:
            raw_query = st.text_input(
                "Search Query",
                placeholder="Type a natural language query or concept...",
                label_visibility="collapsed",
                key="tab_semantic_query_input",
            )
        with search_col2:
            search_button = st.button("Search Chunks", type="primary", use_container_width=True)

        if search_button or raw_query:
            cleaned_search_query = clean_query_text(raw_query)
            if not cleaned_search_query:
                st.warning("Please enter a non-empty search query.")
            elif not vector_manager.is_initialized or vector_manager.total_vectors == 0:
                st.warning("No valid document chunks available in FAISS vector store.")
            else:
                with st.spinner("Searching FAISS index..."):
                    try:
                        search_results = vector_manager.similarity_search_with_score(
                            query=cleaned_search_query,
                            top_k=top_k,
                        )
                    except Exception as search_err:
                        st.error(f"Semantic search query failed: {type(search_err).__name__}")
                        search_results = []

                if not search_results:
                    st.info("No matching chunks found for your query.")
                else:
                    st.markdown(f"**Found {len(search_results)} relevant chunk(s) for:** *\"{cleaned_search_query}\"*")

                    for rank, (doc, raw_dist, score) in enumerate(search_results, start=1):
                        meta = doc.metadata or {}
                        doc_name = meta.get("file_name", meta.get("source", "Document"))
                        page_num = meta.get("page", None)
                        chunk_idx = meta.get("chunk_index", rank)
                        total_c = meta.get("total_chunks", "")
                        score_pct = format_similarity_score(score)
                        badge_color = get_score_badge_color(score)

                        page_badge = f'<span class="badge badge-meta">Page {page_num}</span>' if page_num else ""
                        chunk_badge = f'<span class="badge badge-meta">Chunk {chunk_idx}{"/" + str(total_c) if total_c else ""}</span>'

                        st.markdown(
                            f"""
                            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                                    <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                                        <strong style="color: #1E293B; font-size: 1rem;">Rank #{rank}</strong>
                                        <span class="badge badge-meta">📄 {doc_name}</span>
                                        {page_badge}
                                        {chunk_badge}
                                    </div>
                                    <div>
                                        <span class="badge badge-score" style="background-color: {badge_color};">
                                            Relevance: {score_pct}
                                        </span>
                                    </div>
                                </div>
                                <div class="chunk-content">{doc.page_content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # ---------------------------------------------------------
    # TAB 6: Indexed Chunks Inspection (Preserved)
    # ---------------------------------------------------------
    with main_tab_chunks:
        st.subheader(f"All Indexed Chunks ({len(all_chunks)} total)")
        st.caption(f"Chunk Size: {chunk_size} | Overlap: {chunk_overlap} characters")

        if not all_chunks:
            st.info("No chunks generated.")
        else:
            for chunk in all_chunks:
                meta = chunk.metadata
                chunk_idx = meta.get("chunk_index", 1)
                total_c = meta.get("total_chunks", len(all_chunks))
                char_count = meta.get("chunk_character_count", len(chunk.page_content))
                word_count = meta.get("chunk_word_count", len(chunk.page_content.split()))
                file_source = meta.get("file_name", "document")
                page_num = meta.get("page", None)
                page_info = f" • Page: {page_num}" if page_num else ""

                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.5rem; display: flex; gap: 1rem; font-weight: 500;">
                            <span><strong>📄 {file_source}</strong></span>
                            <span>Chunk {chunk_idx}/{total_c}</span>
                            <span>{char_count} chars</span>
                            <span>{word_count} words</span>
                            <span>{page_info}</span>
                        </div>
                        <div style="font-family: monospace; font-size: 0.88rem; white-space: pre-wrap; color: #1E293B;">{chunk.page_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---------------------------------------------------------
    # TAB 7: Metadata & Pipeline Summary (Preserved & Enhanced)
    # ---------------------------------------------------------
    with main_tab_meta:
        st.subheader("Metadata & NLP Pipeline Summary")
        meta_summary = {
            "llm_provider": "Google Gemini",
            "llm_model": DEFAULT_LLM_MODEL,
            "api_key_configured": rag_pipeline.is_api_key_configured(),
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "vector_dimension": embedding_manager.dimension,
            "total_vectors_in_faiss": vector_manager.total_vectors,
            "sentiment_model": DEFAULT_SENTIMENT_MODEL,
            "intent_categories": INTENT_CATEGORIES,
            "summarization_styles": SUMMARY_STYLES,
            "total_documents": len(successful_files),
            "total_sections": len(all_preprocessed_docs),
            "total_chunks": len(all_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_top_k": top_k,
            "files": file_processing_status,
        }
        st.json(meta_summary)
