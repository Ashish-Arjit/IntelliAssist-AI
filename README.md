# IntelliAssist AI — Document RAG Chatbot and Source Citations

IntelliAssist AI is a modular, production-ready Retrieval-Augmented Generation (RAG) assistant designed for intelligent document question answering. It ingests multi-format documents (PDF, TXT, DOCX), validates and normalizes text, splits content into semantic chunks, generates dense vector embeddings using Hugging Face Sentence Transformers, indexes vectors using an in-memory FAISS vector store, and integrates with **Google Gemini LLM** to deliver accurate, strictly grounded answers with verifiable source citations.

---

## 📌 RAG Architecture & Pipeline Flow

```text
User Question
      ↓
Query Embedding (sentence-transformers/all-MiniLM-L6-v2)
      ↓
FAISS Vector Similarity Search (Top-K Chunks)
      ↓
Retrieved Document Chunks + Lineage Metadata (Filename, Page, Index, Score)
      ↓
Grounded Context Assembly & Prompt Engineering
      ↓
Google Gemini LLM (gemini-3.6-flash / ChatGoogleGenerativeAI)
      ↓
Context-Aware Answer + Verified Source Citations (Page Numbers & Snippets)
```

---

## ✨ Key Implemented Features

* **Grounded Question Answering**:
  * Answers are synthesized strictly from the retrieved document chunks.
  * Explicit fallback response (`"I couldn't find this information in the uploaded documents."`) whenever relevant context is unavailable or insufficient.
  * Strictly avoids hallucinations, inventions, or unrestricted open-domain chatbot behavior.

* **Verified Source Citations**:
  * Every generated answer is paired with collapsible source citation cards.
  * Displays source document filename, PDF page number, chunk index, relevance score badges, and context preview snippets.
  * Citations map 1:1 with the chunks retrieved for that specific query.

* **Multi-Document & Multi-Format Ingestion**:
  * Page-level extraction from **PDF** (`pypdf`).
  * Structured paragraph and table extraction from **DOCX** (`python-docx`).
  * Text file processing with encoding fallbacks for **TXT** (UTF-8 / Latin-1).
  * Simultaneously indexes multiple uploaded files into a unified FAISS vector store.

* **Semantic Chunking & Lineage Preservation**:
  * Recursive character text splitting with configurable chunk size and chunk overlap.
  * Preserves metadata including source document name, page numbers, character/word counts, and chunk indices.

* **Hugging Face Sentence Embeddings & FAISS Retrieval**:
  * Open-source `sentence-transformers/all-MiniLM-L6-v2` dense 384-dimensional embeddings.
  * High-speed FAISS vector store nearest-neighbor similarity search with normalized relevance score calculations (0%–100%).
  * Configurable Top-K retrieval parameters.

* **Modern Streamlit Chat Interface**:
  * Interactive conversational interface with chat messages, user query inputs, and instant response rendering.
  * Multi-tab inspection dashboard with dedicated views for Chatbot, Semantic Search, Indexed Chunks, and Ingestion Metadata.

* **Robust Error Handling**:
  * Handles missing API keys, empty queries, unindexed vector stores, quota limits, and API connection failures cleanly without crashing or exposing technical stack traces.

---

## 🛠️ Technology Stack

* **Language**: Python 3.10+
* **LLM Provider**: Google Gemini (`gemini-3.6-flash` via `langchain-google-genai`)
* **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2` via `langchain-huggingface`)
* **Vector Store**: FAISS (`faiss-cpu`)
* **RAG Orchestration**: `langchain`, `langchain-core`, `langchain-text-splitters`
* **Frontend**: Streamlit
* **Document Parsing**: `pypdf` (PDF), `python-docx` (DOCX), standard library (TXT)
* **Configuration**: `python-dotenv`
* **Testing**: Python `unittest`

---

## 📁 Project Structure

```text
IntelliAssist-AI/
├── modules/
│   ├── __init__.py                  # Core package exports (RAGPipeline, VectorStoreManager, etc.)
│   ├── chunker.py                   # Configurable LangChain document chunking
│   ├── document_loader.py           # Multi-format document loading (PDF, TXT, DOCX)
│   ├── embeddings.py                # Hugging Face sentence-transformers embedding manager
│   ├── preprocessor.py              # Text cleaning and Unicode normalization
│   ├── rag_pipeline.py              # RAG pipeline with Gemini LLM, grounding prompts & citations
│   ├── validator.py                 # File upload and content validation
│   └── vector_store.py              # FAISS vector store and similarity search manager
├── utils/
│   ├── __init__.py                  # Utility package exports
│   └── helpers.py                   # Score formatting, color coding, citation labels & string helpers
├── tests/
│   ├── __init__.py                  # Test suite package
│   ├── test_chunker.py              # Unit tests for text chunker
│   ├── test_document_loader.py      # Unit tests for loaders (PDF, TXT, DOCX)
│   ├── test_embeddings.py           # Unit tests for Hugging Face embeddings
│   ├── test_pipeline.py             # Full ingestion & search pipeline integration tests
│   ├── test_preprocessor.py         # Unit tests for text preprocessor
│   ├── test_rag_pipeline.py         # Unit & integration tests for RAG pipeline & LLM grounding
│   ├── test_validator.py            # Unit tests for document validator
│   ├── test_vector_search_integration.py # Multi-document vector search tests
│   └── test_vector_store.py         # Unit tests for FAISS vector store
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git ignore rules
├── app.py                           # Streamlit web application entrypoint
├── config.py                        # Centralized application, RAG, and model configuration
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ashish-Arjit/IntelliAssist-AI.git
cd IntelliAssist-AI
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# Windows (Command Prompt / PowerShell)
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and set your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_actual_google_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-3.6-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=1024
DEFAULT_TOP_K=4
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
```

> **Note**: Get a free API key from [Google AI Studio](https://aistudio.google.com/). The API key can also be provided directly through the Streamlit sidebar at runtime.

---

## 🚀 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Using the Document Chatbot:
1. **Upload Documents**: Upload one or more PDF, TXT, or DOCX files using the sidebar.
2. **Review Ingestion**: Ingestion metrics bar displays processed files, sections, and FAISS vector count.
3. **Ask Questions**: Use the **Document Chatbot** tab to enter questions about your files.
4. **Inspect Answers & Citations**: Read the grounded AI answer and expand the **View Source Citations** panel to see the exact document, page number, relevance score, and content snippet.
5. **Inspect Vectors**: Switch to the **Semantic Search** or **Indexed Chunks** tabs to explore underlying vector distances and chunk breakdowns.

---

## 🧪 Running Automated Tests

Run the test suite across all modules and the RAG pipeline:

```bash
python -m unittest discover -s tests
```

To run only the RAG pipeline test suite:

```bash
python -m unittest tests/test_rag_pipeline.py
```
