"""IntelliAssist AI - Smart Document AI Assistant.

Complete RAG Pipeline with Google Gemini LLM, FAISS Vector Search,
Grounded Answering, and Verified Source Citations.
"""

from typing import Any, Dict, List, Optional
import os
import streamlit as st
from langchain_core.documents import Document

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SUMMARY_STYLE,
    DEFAULT_TOP_K,
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
from modules.preprocessor import TextPreprocessor
from modules.rag_pipeline import RAGPipeline
from modules.summarization import DocumentSummarizer
from modules.validator import DocumentValidator
from modules.vector_store import VectorStoreManager
from utils.helpers import (
    build_chat_message,
    clean_query_text,
    clear_chat_history,
    export_chat_history,
    format_citation_label,
    format_conversation_timestamp,
    format_file_size,
    format_similarity_score,
    get_score_badge_color,
    truncate_snippet,
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


# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_processed_files" not in st.session_state:
    st.session_state["last_processed_files"] = []

if "summary_result" not in st.session_state:
    st.session_state["summary_result"] = None


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

    # Process all uploaded files
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
                    "size_bytes": up_file.size,
                    "sections_count": len(prep_docs),
                    "chunks_count": len(doc_chunks),
                })
            except Exception as doc_err:
                file_processing_status.append({
                    "filename": file_name,
                    "status": "error",
                    "message": f"Extraction error: {type(doc_err).__name__}",
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

        # 7. Initialize RAG Pipeline
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
    # Main Tabs: Chat Assistant & Vector Inspection
    # ---------------------------------------------------------
    main_tab_chat, main_tab_summary, main_tab_search, main_tab_chunks, main_tab_meta = st.tabs(
        [
            "💬 Document Chatbot",
            "📝 Document Summarization",
            "🔍 Semantic Search",
            "🧩 Indexed Chunks",
            "🔍 Metadata & Pipeline Summary",
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
                if ts:
                    st.markdown(
                        f'<div style="font-size: 0.75rem; color: #94A3B8; text-align: right; margin-top: -8px;">{ts}</div>',
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

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(cleaned_q)
            st.session_state["messages"].append(
                build_chat_message(role="user", content=cleaned_q)
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
    # TAB 3: Semantic Similarity Search (Preserved)
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
    # TAB 3: Indexed Chunks Inspection (Preserved)
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
    # TAB 4: Metadata & Pipeline Summary (Preserved)
    # ---------------------------------------------------------
    with main_tab_meta:
        st.subheader("Metadata & RAG Architecture Summary")
        meta_summary = {
            "llm_provider": "Google Gemini",
            "llm_model": DEFAULT_LLM_MODEL,
            "api_key_configured": rag_pipeline.is_api_key_configured(),
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "vector_dimension": embedding_manager.dimension,
            "total_vectors_in_faiss": vector_manager.total_vectors,
            "total_documents": len(successful_files),
            "total_sections": len(all_preprocessed_docs),
            "total_chunks": len(all_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_top_k": top_k,
            "files": file_processing_status,
        }
        st.json(meta_summary)
