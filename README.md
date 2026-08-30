# IntelliAssist AI — Embeddings and Semantic Search

IntelliAssist AI is a modular document processing, embedding, and semantic search assistant. It ingests multi-format documents (PDF, TXT, DOCX), validates content, cleans and normalizes text, splits documents into manageable chunks, generates dense vector embeddings using Hugging Face Sentence Transformers, and performs high-speed semantic similarity searches using a FAISS vector store.

---

## 📌 Pipeline Architecture

```text
Document Upload (PDF, TXT, DOCX — Single or Multiple)
        ↓
Text & Metadata Extraction (DocumentLoader)
        ↓
Validation & Quality Checks (DocumentValidator)
        ↓
Text Normalization & Cleaning (TextPreprocessor)
        ↓
Configurable Text Chunking (DocumentChunker)
        ↓
Hugging Face Embeddings (EmbeddingManager: all-MiniLM-L6-v2)
        ↓
FAISS Vector Store Indexing (VectorStoreManager)
        ↓
User Natural Language Query → Query Vector → FAISS Similarity Search → Top-K Ranked Results
```

---

## ✨ Implemented Features

* **Multi-Format Document Parsing**:
  * Page-level extraction from **PDF** (`pypdf`).
  * Structured paragraph and table extraction from **DOCX** (`python-docx`).
  * Text file processing with encoding fallbacks for **TXT** (UTF-8 / Latin-1).
* **Multi-Document Support**: Upload and index multiple files at once. Lineage and source metadata are preserved across all documents in a unified FAISS index.
* **Non-Destructive Text Preprocessing**: Strips excessive whitespace, normalizes Unicode (NFKC), and standardizes line breaks while strictly preserving punctuation and structural meaning.
* **Configurable Semantic Chunking**: Splits normalized text using LangChain's `RecursiveCharacterTextSplitter` with customizable chunk size and overlap parameters.
* **Hugging Face Sentence Embeddings**:
  * Uses the open-source `sentence-transformers/all-MiniLM-L6-v2` model (384-dimensional dense vectors).
  * Runs entirely on CPU without any paid API keys or external services.
  * Encapsulated in a modular `EmbeddingManager` with automatic resource caching.
* **FAISS Vector Store**:
  * In-memory FAISS indexing with full document chunk and metadata persistence.
  * Fast similarity search using cosine and normalized Euclidean (L2) distance.
  * Dynamic index construction and updating when documents are processed.
* **Semantic Search Interface**:
  * Natural language query input with ranked results.
  * Normalized similarity/relevance scores (0% to 100%) with color-coded badges.
  * Detailed result cards showing source document filename, page/section number, chunk index, and retrieved text.
* **Robust Error Handling**: Friendly error banners for missing files, empty queries, unreadable scans, or processing issues without exposing technical stack traces.

---

## 🛠️ Technology Stack

* **Language**: Python 3.10+
* **Frontend**: Streamlit
* **Embeddings**: `sentence-transformers`, `langchain-huggingface` (`all-MiniLM-L6-v2`)
* **Vector Store**: FAISS (`faiss-cpu`)
* **Document Processing & Chunking**: `langchain-core`, `langchain-community`, `langchain-text-splitters`
* **File Extractors**: `pypdf` (PDF), `python-docx` (DOCX), built-in Python I/O (TXT)
* **Configuration**: `python-dotenv`
* **Testing**: Python `unittest`

---

## 📁 Project Structure

```text
IntelliAssist-AI/
├── modules/
│   ├── __init__.py                  # Core modules package exports
│   ├── chunker.py                   # Configurable LangChain document chunking
│   ├── document_loader.py           # Multi-format document loading (PDF, TXT, DOCX)
│   ├── embeddings.py                # Hugging Face sentence-transformers embedding manager
│   ├── preprocessor.py              # Text cleaning and Unicode normalization
│   ├── validator.py                 # File upload and content validation
│   └── vector_store.py              # FAISS vector store and similarity search manager
├── utils/
│   ├── __init__.py                  # Utility package exports
│   └── helpers.py                   # Score formatting, color coding, and string helpers
├── tests/
│   ├── __init__.py                  # Test suite package
│   ├── test_chunker.py              # Unit tests for text chunker
│   ├── test_document_loader.py      # Unit tests for loaders (PDF, TXT, DOCX)
│   ├── test_embeddings.py           # Unit tests for Hugging Face embeddings
│   ├── test_pipeline.py             # Full ingestion & search pipeline integration tests
│   ├── test_preprocessor.py         # Unit tests for text preprocessor
│   ├── test_validator.py            # Unit tests for document validator
│   ├── test_vector_search_integration.py # Multi-document vector search tests
│   └── test_vector_store.py         # Unit tests for FAISS vector store
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git ignore rules
├── app.py                           # Streamlit web application entrypoint
├── config.py                        # Centralized application and model configuration
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ashish-Arjit/IntelliAssist-AI.git
cd IntelliAssist-AI
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)

```bash
cp .env.example .env
```

---

## 💻 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will launch locally at `http://localhost:8501`.

---

## 🔍 How to Perform Semantic Search

1. **Upload Documents**: Select one or more `.pdf`, `.txt`, or `.docx` files using the sidebar file uploader.
2. **Configure Parameters (Optional)**: Adjust chunk size, overlap, or the top-K retrieved results in the sidebar.
3. **Automatic Indexing**: The application extracts text, normalizes content, generates chunk embeddings using `all-MiniLM-L6-v2`, and constructs the FAISS vector index.
4. **Enter Search Query**: Type any natural language question or topic in the **Semantic Similarity Search** input field (e.g., *"What were the quarterly financial highlights?"* or *"Explain cloud architecture"*).
5. **Inspect Retrieved Results**: Review ranked chunks with relevance percentage scores, document origins, page numbers, and exact chunk text.

---

## 🧪 Running Automated Tests

Run the complete test suite using `unittest`:

```bash
python -m unittest discover -s tests -v
```
