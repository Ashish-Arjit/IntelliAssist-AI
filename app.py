"""IntelliAssist AI - Smart Document AI Assistant.

Hugging Face Embeddings & FAISS Semantic Search Layer with Multi-Document Support.
"""

from typing import Any, Dict, List, Optional
import streamlit as st
from langchain_core.documents import Document

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
)
from modules.chunker import DocumentChunker
from modules.document_loader import DocumentLoader
from modules.embeddings import EmbeddingManager
from modules.preprocessor import TextPreprocessor
from modules.validator import DocumentValidator
from modules.vector_store import VectorStoreManager
from utils.helpers import (
    clean_query_text,
    format_file_size,
    format_similarity_score,
    get_score_badge_color,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntelliAssist AI - Semantic Search",
    page_icon="📄",
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
            font-size: 2.2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.02rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            text-align: center;
        }
        .metric-title {
            font-size: 0.8rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #1E293B;
            margin-top: 0.25rem;
        }
        .feature-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            height: 100%;
        }
        .search-result-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease;
        }
        .search-result-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        }
        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .badge-score {
            color: #FFFFFF;
        }
        .badge-meta {
            background-color: #F1F5F9;
            color: #475569;
            border: 1px solid #E2E8F0;
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
        .status-badge-active {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
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
# Sidebar Controls & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.title("📄 IntelliAssist AI")
    st.caption("Embeddings & FAISS Semantic Search")
    st.divider()

    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select one or more documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload PDF (.pdf), Plain Text (.txt), or Word (.docx) files.",
    )

    st.divider()
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
    st.subheader("🔍 Search Configuration")
    top_k = st.slider(
        "Top-K Retrieved Results",
        min_value=MIN_TOP_K,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
        help="Number of most relevant document chunks to return for search queries.",
    )

    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #64748B; background: #F8FAFC; padding: 0.6rem; border-radius: 6px; border: 1px solid #E2E8F0;">
            <strong>Model:</strong> <code>{DEFAULT_EMBEDDING_MODEL.split('/')[-1]}</code><br/>
            <strong>Vector Store:</strong> FAISS (In-Memory)<br/>
            <strong>Similarity Metric:</strong> Cosine / Normalized L2
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.info("💡 **Embeddings & Search Stage**: Multi-doc ingestion, Hugging Face vectors, and FAISS similarity search.")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">📄 IntelliAssist AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Hugging Face Embeddings & FAISS Vector Store Semantic Search"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Welcome Screen (No Documents Uploaded)
# ---------------------------------------------------------
if not uploaded_files:
    st.info("👋 Welcome! Upload one or more **PDF**, **TXT**, or **DOCX** files in the sidebar to build your FAISS vector store and perform semantic searches.")

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
                <h4>🧩 Semantic Chunking</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    Configurable RecursiveCharacterTextSplitter with lineage and index tracking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🧠 HF Embeddings</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    Sentence-transformers (<code>all-MiniLM-L6-v2</code>) generating 384-dimensional dense vectors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <h4>⚡ FAISS Vector Search</h4>
                <p style="color: #64748B; font-size: 0.88rem;">
                    High-speed nearest-neighbor similarity search with normalized relevance score rankings.
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
        st.error("Failed to initialize Hugging Face embedding model. Please check network connectivity and local dependencies.")
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

    # ---------------------------------------------------------
    # Ingestion Status & Metrics
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
                <div class="metric-title">FAISS Index Status</div>
                <div class="metric-value" style="color: #10B981; font-size: 1.15rem;">
                    {"Ready (384-d)" if vector_manager.is_initialized else "Not Ready"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Document Status List expander
    with st.expander("📋 Processed Document Details", expanded=False):
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
    # 🔍 Semantic Search Section
    # ---------------------------------------------------------
    st.markdown("### 🔍 Semantic Similarity Search")
    st.caption("Query the FAISS vector database to retrieve the most semantically relevant document chunks.")

    search_col1, search_col2 = st.columns([5, 1])
    with search_col1:
        query_input = st.text_input(
            "Search Query",
            placeholder="Type a natural language query or concept...",
            label_visibility="collapsed",
            key="semantic_query_input",
        )
    with search_col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Execute Search
    if search_button or query_input:
        cleaned_query = clean_query_text(query_input)

        if not cleaned_query:
            st.warning("Please enter a non-empty search query.")
        elif not vector_manager.is_initialized or vector_manager.total_vectors == 0:
            st.warning("No valid document chunks available in FAISS vector store. Please upload valid documents.")
        else:
            with st.spinner("Embedding query & searching FAISS index..."):
                try:
                    search_results = vector_manager.similarity_search_with_score(
                        query=cleaned_query,
                        top_k=top_k,
                    )
                except Exception as search_err:
                    st.error("An error occurred during similarity search.")
                    st.caption(f"Error details: {type(search_err).__name__}")
                    search_results = []

            if not search_results:
                st.info("No matching chunks found for your query.")
            else:
                st.markdown(f"**Found {len(search_results)} relevant chunk(s) for:** *\"{cleaned_query}\"*")

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
                        <div class="search-result-card">
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

    st.divider()

    # ---------------------------------------------------------
    # Inspection Tabs (Ingestion, Cleaned Text, Metadata)
    # ---------------------------------------------------------
    tab_chunks, tab_preview, tab_meta = st.tabs(
        ["🧩 Indexed Chunks", "📄 Cleaned Text Previews", "🔍 Metadata & Ingestion Metrics"]
    )

    with tab_chunks:
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

    with tab_preview:
        st.subheader("Extracted & Normalized Text")
        if not all_preprocessed_docs:
            st.info("No preprocessed document text.")
        else:
            for idx, doc in enumerate(all_preprocessed_docs, start=1):
                f_name = doc.metadata.get("file_name", "Document")
                p_label = f"Page {doc.metadata.get('page')}" if "page" in doc.metadata else f"Section {idx}"
                with st.expander(f"📖 {f_name} — {p_label} ({len(doc.page_content)} chars)", expanded=(idx == 1)):
                    st.text_area(
                        f"Content for {f_name} - {p_label}",
                        value=doc.page_content,
                        height=220,
                        disabled=True,
                        key=f"text_area_{idx}_{f_name}",
                    )

    with tab_meta:
        st.subheader("Metadata & Vector Ingestion Summary")
        meta_summary = {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "vector_dimension": embedding_manager.dimension,
            "total_vectors_in_faiss": vector_manager.total_vectors,
            "total_documents": len(successful_files),
            "total_sections": len(all_preprocessed_docs),
            "total_chunks": len(all_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "files": file_processing_status,
        }
        st.json(meta_summary)
