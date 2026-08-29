"""IntelliAssist AI - Smart Document AI Assistant.

Initial Foundation Interface.
"""

import streamlit as st

# Configure page layout and visual theme
st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, clean, modern presentation
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .feature-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .metric-badge {
            background-color: #EEF2FF;
            color: #4F46E5;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar configuration
with st.sidebar:
    st.title("📄 IntelliAssist AI")
    st.caption("Smart Document Ingestion & Processing")
    st.divider()

    st.subheader("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        help="Supported formats: PDF, TXT, DOCX",
    )

    st.divider()
    st.subheader("⚙️ Chunking Configuration")
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=1000,
        step=50,
        help="Maximum number of characters per chunk.",
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
    st.info("💡 Stage 1: Document Ingestion & Preprocessing Foundation")

# Main content header
st.markdown('<div class="main-header">📄 IntelliAssist AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "A modular document processing foundation for smart document analysis, text normalization, and semantic chunking."
    "</div>",
    unsafe_allow_html=True,
)

if not uploaded_file:
    st.info("👋 Welcome! Please upload a PDF, TXT, or DOCX document from the sidebar to begin.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📑 Multi-Format Ingestion</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Extracts text and metadata from PDF, TXT, and DOCX documents with structured page and source tracking.
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
                    Performs non-destructive preprocessing, whitespace cleaning, and structural normalization.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🧩 Semantic Chunking</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Splits documents into coherent chunks with customizable sizes and preserved metadata for future retrieval.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.success(f"Selected file: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
    st.write("Ready for processing.")
