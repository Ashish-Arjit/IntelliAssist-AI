"""IntelliAssist AI - Smart Document AI Assistant.

Initial Foundation: Document Ingestion, Preprocessing, and Chunking Pipeline.
"""

from io import BytesIO
import json
import streamlit as st

from modules.chunker import DocumentChunker
from modules.document_loader import DocumentLoader
from modules.preprocessor import TextPreprocessor
from modules.validator import DocumentValidator

# Page Configuration
st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling for a polished, modern interface
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
            font-size: 1.05rem;
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
            font-size: 1.5rem;
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
        }
        .chunk-box {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            color: #1E293B;
        }
        .chunk-meta {
            font-size: 0.8rem;
            color: #64748B;
            margin-bottom: 0.5rem;
            display: flex;
            gap: 1rem;
            font-family: sans-serif;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.title("📄 IntelliAssist AI")
    st.caption("Smart Document Ingestion & Chunking")
    st.divider()

    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Select a document to process",
        type=["pdf", "txt", "docx"],
        help="Supported formats: PDF (.pdf), Plain Text (.txt), Word (.docx)",
    )

    st.divider()
    st.subheader("⚙️ Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2500,
        value=1000,
        step=50,
        help="Maximum number of characters per document chunk.",
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=200,
        step=25,
        help="Number of overlapping characters between consecutive chunks.",
    )

    st.divider()
    st.info("💡 **Foundation Stage**: Ingestion, validation, preprocessing, and chunking pipeline.")

# Main Application Area
st.markdown('<div class="main-header">📄 IntelliAssist AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Smart document ingestion, metadata-preserving normalization, and configurable text chunking."
    "</div>",
    unsafe_allow_html=True,
)

if not uploaded_file:
    st.info("👋 Welcome! Upload a **PDF**, **TXT**, or **DOCX** file from the sidebar to inspect its ingestion pipeline.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📑 Document Ingestion</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Load PDF, TXT, and DOCX files with rich metadata extraction (file type, line counts, pages, and sources).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🧹 Text Normalization</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Cleans irregular whitespace, collapses redundant blank lines, and normalizes characters while preserving punctuation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🧩 Configurable Chunking</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Splits content into coherent chunks using LangChain text splitters with complete metadata continuity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    # 1. File Upload Validation
    file_val = DocumentValidator.validate_file_upload(uploaded_file)
    if not file_val.is_valid:
        st.error(file_val.error_message)
    else:
        try:
            with st.spinner("Processing document..."):
                # 2. Extract Document Text and Metadata
                raw_documents = DocumentLoader.load_document(
                    uploaded_file, filename=uploaded_file.name
                )

                # 3. Validate Extracted Content
                content_val = DocumentValidator.validate_extracted_content(
                    raw_documents, filename=uploaded_file.name
                )

                if not content_val.is_valid:
                    st.warning(content_val.error_message)
                else:
                    # 4. Preprocess Text
                    preprocessed_docs = TextPreprocessor.preprocess_documents(
                        raw_documents, drop_empty=True
                    )

                    # 5. Chunk Documents
                    chunker = DocumentChunker(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    chunks = chunker.split_documents(preprocessed_docs)

            # Processing Summary Metrics
            st.success(f"Successfully processed **{uploaded_file.name}**")

            total_chars = sum(len(doc.page_content) for doc in preprocessed_docs)
            total_words = sum(len(doc.page_content.split()) for doc in preprocessed_docs)
            file_ext = uploaded_file.name.rsplit(".", 1)[-1].upper()
            file_size_kb = uploaded_file.size / 1024

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Format</div>
                        <div class="metric-value">{file_ext}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">File Size</div>
                        <div class="metric-value">{file_size_kb:.1f} KB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Pages / Sections</div>
                        <div class="metric-value">{len(preprocessed_docs)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Words</div>
                        <div class="metric-value">{total_words:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Generated Chunks</div>
                        <div class="metric-value" style="color: #4F46E5;">{len(chunks)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            # Inspection Tabs
            tab_chunks, tab_preview, tab_meta = st.tabs(
                ["🧩 Generated Chunks", "📄 Cleaned Text Preview", "🔍 Metadata & Ingestion Info"]
            )

            with tab_chunks:
                st.subheader(f"Document Chunks ({len(chunks)} total)")
                st.caption(
                    f"Configured Chunk Size: {chunk_size} chars | Overlap: {chunk_overlap} chars"
                )

                if not chunks:
                    st.info("No chunks generated.")
                else:
                    for chunk in chunks:
                        meta = chunk.metadata
                        chunk_idx = meta.get("chunk_index", 1)
                        total_c = meta.get("total_chunks", len(chunks))
                        char_count = meta.get("chunk_character_count", len(chunk.page_content))
                        word_count = meta.get("chunk_word_count", len(chunk.page_content.split()))
                        page_num = meta.get("page", None)
                        page_info = f" • Page: {page_num}" if page_num else ""

                        st.markdown(
                            f"""
                            <div class="chunk-box">
                                <div class="chunk-meta">
                                    <span><strong>Chunk {chunk_idx}/{total_c}</strong></span>
                                    <span>{char_count} chars</span>
                                    <span>{word_count} words</span>
                                    <span>{page_info}</span>
                                </div>
                                <div>{chunk.page_content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with tab_preview:
                st.subheader("Extracted & Normalized Text")
                for idx, doc in enumerate(preprocessed_docs, start=1):
                    page_label = f"Page {doc.metadata.get('page')}" if "page" in doc.metadata else f"Section {idx}"
                    with st.expander(f"📖 {page_label} ({len(doc.page_content)} characters)", expanded=(idx == 1)):
                        st.text_area(
                            f"Content for {page_label}",
                            value=doc.page_content,
                            height=250,
                            disabled=True,
                            key=f"text_area_{idx}",
                        )

            with tab_meta:
                st.subheader("Extracted Metadata")
                meta_summary = {
                    "filename": uploaded_file.name,
                    "format": file_ext.lower(),
                    "size_bytes": uploaded_file.size,
                    "total_sections": len(preprocessed_docs),
                    "total_characters": total_chars,
                    "total_words": total_words,
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "section_metadata": [doc.metadata for doc in preprocessed_docs],
                }
                st.json(meta_summary)

        except Exception as e:
            st.error("An error occurred while processing the document. Please verify the document format and try again.")
            st.caption(f"Error details: {type(e).__name__}")
