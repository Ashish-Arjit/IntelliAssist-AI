# IntelliAssist AI — Smart Document RAG Assistant & NLP Toolkit

IntelliAssist AI is a modular, production-ready Document AI and Retrieval-Augmented Generation (RAG) assistant designed for intelligent document question answering, summarization, sentiment evaluation, and query intent classification. It ingests multi-format documents (PDF, TXT, DOCX), extracts and normalizes text, splits content into semantic chunks, generates dense vector embeddings using Hugging Face Sentence Transformers, indexes vectors using an in-memory FAISS vector store, and integrates with **Google Gemini LLM** to deliver accurate, strictly grounded answers with verifiable source citations and advanced NLP capabilities.

---

## 📌 Problem Statement & Solution Overview

### Problem Statement
Organizations and individuals frequently handle dense, multi-format documents (reports, contracts, manuals, financial disclosures). Manually searching through multiple files to extract relevant information is time-consuming and error-prone. Traditional keyword search often misses conceptual relevance, while standard LLMs suffer from hallucinations when asked about domain-specific or unindexed private documents.

### Solution Overview
IntelliAssist AI solves these challenges by combining:
1. **Multi-Format Ingestion**: Unified text extraction and metadata preservation for PDF, TXT, and DOCX files.
2. **Dense Semantic Search**: High-dimensional embeddings and an in-memory FAISS vector store that capture conceptual meaning rather than just exact keywords.
3. **Strictly Grounded RAG Generation**: Constrained prompting instructing Google Gemini to synthesize answers strictly from retrieved document chunks, returning an explicit fallback message when information is absent.
4. **Transparent Source Citations**: Every answer provides clickable source cards with file names, PDF page numbers, chunk indices, and relevance scores.
5. **Integrated NLP Toolkit**: Multi-style document summarization (with large-document map-reduce), sentiment analysis, query intent classification, and multi-turn conversation memory.

---

## 🏗️ System Architecture

The following diagram illustrates the actual implemented end-to-end architecture of IntelliAssist AI:

```text
User
  ↓
Streamlit Interface
  ↓
Document Upload
  ↓
Text Extraction
  ↓
Text Preprocessing
  ↓
Text Chunking
  ↓
Hugging Face Embeddings
  ↓
FAISS Vector Store
  ↓
Semantic Retrieval
  ↓
Relevant Context
  ↓
LLM
  ↓
Context-Aware Answer
  ↓
Source Citations
```

### Architectural Component Breakdown

1. **User Interaction**: Users interact with the system via a modern, intuitive Streamlit web interface.
2. **Document Ingestion & Validation**: Uploaded documents (PDF, TXT, DOCX) pass through format and size validation (`DocumentValidator`).
3. **Text Extraction**: Specialized loaders extract raw text while tracking document-level and page-level metadata (`DocumentLoader`).
4. **Text Preprocessing**: Normalizes Unicode characters, cleans excessive whitespace, and eliminates noise while preserving semantic structure (`TextPreprocessor`).
5. **Text Chunking**: Breaks normalized text into semantically cohesive overlapping segments (default chunk size: 1000 characters, overlap: 200 characters) preserving source tags (`DocumentChunker`).
6. **Hugging Face Embeddings**: Chunks are converted into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` (`EmbeddingManager`).
7. **FAISS Vector Store**: Embeddings and document metadata are indexed in an in-memory FAISS flat L2 vector index (`VectorStoreManager`).
8. **Semantic Retrieval**: Queries are embedded into the same vector space, and the top-$k$ nearest neighbors are retrieved based on vector distance.
9. **Relevant Context Assembly**: Retrieved chunks above the similarity threshold are extracted with citations.
10. **LLM Synthesis**: Context and user query are formatted with a strict grounding prompt and sent to Google Gemini (`gemini-1.5-flash`).
11. **Context-Aware Answer**: The synthesized answer is displayed with explicit guardrails against hallucination.
12. **Source Citations**: Collapsible citation cards display source file names, PDF page numbers, chunk indices, similarity scores, and text previews.

---

## 🔄 RAG Workflow

The Retrieval-Augmented Generation (RAG) execution workflow operates through four stages:

1. **Document Ingestion Phase**:
   - The user selects one or more files (`.pdf`, `.txt`, `.docx`).
   - The files are checked for valid extensions, non-empty content, and size limits ($\le$ 25 MB).
   - Text is parsed using `pypdf` for PDFs, `python-docx` for Word documents, and multi-encoding readers for plain text.
   - Text is cleaned with Unicode NFKC normalization and formatted into structured chunks.

2. **Embedding & Vector Indexing Phase**:
   - The local Sentence Transformers model computes dense vector representations for all chunks.
   - Vectors are indexed in an in-memory FAISS index.
   - Chunks, metadata, and the FAISS vector index are cached in Streamlit's `session_state` to prevent redundant re-computation across reruns.

3. **Query & Retrieval Phase**:
   - The user inputs a natural language question.
   - The query intent is classified (`Question`, `Summary Request`, `Information Search`, `Explanation Request`).
   - The query is vectorized and queried against the FAISS index to retrieve the top-$k$ most relevant chunks.
   - Normalized relevance scores (`1 - (distance^2 / 2)`) are computed; chunks falling below the minimum threshold (default 0.20) are excluded.

4. **Generation & Verification Phase**:
   - If no chunks meet the relevance threshold, the pipeline immediately returns: `"I couldn't find this information in the uploaded documents."`
   - Otherwise, the retrieved chunks are formatted into a constrained prompt instructing Gemini to answer solely based on the provided context.
   - The generated response is returned alongside structured source citations.

---

## ✨ Key Features

### 1. 💬 Grounded Document Question Answering (RAG)
* **Strict Grounding**: Answers are synthesized strictly from retrieved document chunks.
* **Anti-Hallucination Guardrails**: Explicit fallback (`"I couldn't find this information in the uploaded documents."`) whenever relevant context is unavailable, insufficient, or below the minimum similarity threshold.
* **Verified Source Citations**: Collapsible citation cards displaying the source file, PDF page number, chunk index, relevance score badges (0%–100%), and snippet previews.

### 2. 🕒 Session Conversation History
* **Multi-Turn Memory**: Preserves user questions, AI responses, timestamps, and citation metadata across interactions in Streamlit session state.
* **Previous Messages Visible**: Full chat history remains accessible as users ask follow-up questions or explore tabs.
* **Clear Conversation**: Easily clear session dialogue with a single click from the sidebar or chat toolbar.
* **Transcript Export**: Download the full conversation session as a formatted Markdown transcript (`intelliassist_chat_transcript.md`).

### 3. 📝 Document Summarization
* **Upload-Grounded Summaries**: Generates comprehensive summaries based strictly on uploaded documents.
* **Multiple Summary Styles**:
  * **Executive Summary**: High-level synthesis of objectives, key findings, and overarching conclusions.
  * **Key Points / Bullet Points**: Structured, actionable core insights with bulleted highlights.
  * **Comprehensive Overview**: Detailed section-by-section breakdown.
* **Intelligent Large-Document Handling**: Automatically divides lengthy documents into structured chunks and uses a map-reduce synthesis flow to prevent token overflow.
* **Offline Fallback**: Extractive NLP frequency-based summarization fallback when an LLM API key is not configured.

### 4. 🎭 Document Sentiment Analysis
* **Emotional Tone Detection**: Evaluates document content into **Positive**, **Negative**, or **Neutral** sentiment.
* **Confidence Scoring**: Computes probabilistic confidence scores and complete distribution breakdowns across all sentiment classes.
* **Flexible Analysis Scope**: Analyze entire uploaded files or inspect custom excerpt snippets.
* **Hugging Face Model with Robust Fallback**: Leverages Hugging Face NLP transformers with an offline valence lexicon fallback.

### 5. 🎯 Query Intent Analysis
* **Functional Purpose Detection**: Identifies the purpose behind user queries before or during question answering.
* **Core Intent Categories**:
  * **Question**: Factual queries seeking specific answers (e.g., *"What is this document about?"*).
  * **Summary Request**: Requests to condense or outline text (e.g., *"Summarize this document."*).
  * **Information Search**: Targeted retrieval for topics or data (e.g., *"Find information about machine learning."*).
  * **Explanation Request**: Conceptual inquiries seeking mechanisms or reasoning (e.g., *"Explain the main concept in this document."*).
* **Real-Time Integration**: Visual intent badge rendered directly alongside user chat messages in the conversation view.

### 6. 📑 Multi-Format Ingestion & FAISS Vector Search
* **Multi-Format Processing**: Simultaneous ingestion of **PDF** (`pypdf`), **DOCX** (`python-docx`), and **TXT** files.
* **Hugging Face Sentence Transformers**: High-performance open-source `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense embeddings.
* **In-Memory FAISS Vector Store**: Fast L2 similarity search with normalized percentage relevance score calculation (`1 - (distance^2 / 2)`).
* **Smart Session Caching**: Vector store and document chunks are cached in Streamlit session state, eliminating redundant embedding computations on user interactions.

---

## 🛠️ Technology Stack

* **Programming Language**: Python 3.10+
* **LLM Engine**: Google Gemini (`gemini-1.5-flash` via `langchain-google-genai`)
* **Embedding Model**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2` via `langchain-huggingface`)
* **Vector Store**: FAISS (`faiss-cpu`)
* **NLP & Sentiment**: Hugging Face Transformers (`transformers`), Rule-Based Lexicon NLP
* **RAG Orchestration**: `langchain`, `langchain-core`, `langchain-text-splitters`
* **Web UI**: Streamlit
* **Document Processing**: `pypdf`, `python-docx`
* **Testing**: Python standard library `unittest`, `pytest`

---

## 📁 Project Structure

```text
IntelliAssist-AI/
├── modules/
│   ├── __init__.py                  # Public exports for core modules
│   ├── chunker.py                   # Recursive text chunking with metadata preservation
│   ├── document_loader.py           # Multi-format document loading (PDF, TXT, DOCX) & error handling
│   ├── embeddings.py                # Hugging Face embedding manager (all-MiniLM-L6-v2)
│   ├── intent.py                    # Query intent classification (Question, Summary, Search, Explain)
│   ├── preprocessor.py              # Text cleaning and Unicode normalization
│   ├── rag_pipeline.py              # Grounded RAG answering with Gemini & source citations
│   ├── sentiment.py                 # Sentiment analysis (Positive, Negative, Neutral + confidence)
│   ├── summarization.py             # Multi-style summarizer with large-document map-reduce
│   ├── validator.py                 # File format, size, and content validation
│   └── vector_store.py              # FAISS vector indexing and similarity retrieval
├── utils/
│   ├── __init__.py                  # Utility package exports
│   └── helpers.py                   # Formatting, conversation state, badges, and export helpers
├── tests/
│   ├── __init__.py                  # Test suite package
│   ├── test_chunker.py              # Unit tests for text chunker
│   ├── test_document_loader.py      # Unit tests for loaders & corrupted file handling
│   ├── test_embeddings.py           # Unit tests for Hugging Face embeddings
│   ├── test_nlp_features.py         # Unit tests for history, summarization, sentiment & intent
│   ├── test_pipeline.py             # End-to-end multi-format ingestion & search tests
│   ├── test_preprocessor.py         # Unit tests for text preprocessor
│   ├── test_rag_pipeline.py         # Unit & integration tests for RAG pipeline & LLM grounding
│   ├── test_validator.py            # Unit tests for document validator
│   ├── test_vector_search_integration.py # Multi-document concurrent vector search tests
│   └── test_vector_store.py         # Unit tests for FAISS vector store
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git ignore rules (protects .env and secrets)
├── app.py                           # Multi-tab Streamlit web application with session caching
├── config.py                        # Centralized application, NLP, and model configuration
├── requirements.txt                 # Complete project dependencies
└── README.md                        # Project documentation
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

# Windows (PowerShell)
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_actual_google_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-1.5-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=1024
MIN_SIMILARITY_THRESHOLD=0.20
DEFAULT_TOP_K=4
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
```

> **Note**: You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/). The API key can also be provided directly via the Streamlit sidebar input during runtime without modifying `.env`.

---

## 🚀 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Interface Navigation:
1. **Upload Documents**: Upload one or more PDF, TXT, or DOCX files via the sidebar.
2. **💬 Document Chatbot**: Ask questions grounded in your files, inspect source citations, view detected query intent badges, and manage session history.
3. **📝 Document Summarization**: Choose a summary style (Executive, Key Points, Comprehensive) and generate concise, structured summaries with download option.
4. **🎭 Sentiment Analysis**: Evaluate emotional tone (Positive, Negative, Neutral) and confidence scores of uploaded documents or custom excerpts.
5. **🎯 Intent Analysis**: Interactively test and verify query intents with sample test queries.
6. **🔍 Semantic Search**: Search the raw FAISS vector index and review nearest neighbors with relevance scores.
7. **🧩 Indexed Chunks**: Inspect all processed chunks, character lengths, and metadata.
8. **🔍 Metadata & Summary**: Review overall RAG architecture parameters and active NLP configuration.

---

## 🎓 University Viva Demonstration Guide

When presenting this project in a viva or technical demonstration, follow this step-by-step walkthrough:

### Step 1: Explain the Document Ingestion Pipeline
* Show the sidebar file uploader and upload sample PDF, TXT, and DOCX files.
* Explain how `DocumentLoader` inspects file extension and extracts text (`pypdf` for page-level PDF, `docx` for paragraphs/tables, `txt` with auto-encoding).
* Show that `TextPreprocessor` normalizes Unicode (NFKC) and cleans linebreaks without losing numbers or punctuation.
* Explain that `DocumentChunker` breaks text into semantically cohesive overlapping blocks while embedding rich metadata (`file_name`, `page`, `chunk_index`, `total_chunks`).

### Step 2: Dense Embeddings & Vector Search (FAISS)
* Switch to the **Indexed Chunks** tab to show the chunks created with metadata tags.
* Switch to the **Semantic Search** tab. Type a semantic concept (not necessarily exact keywords from the text).
* Show how FAISS finds nearest neighbors and computes normalized cosine similarity scores (`0% - 100%`).

### Step 3: Grounded RAG Chatbot & Citations
* Switch to the **Document Chatbot** tab.
* Ask a factual question present in the document. Point out the structured answer synthesized by Gemini.
* Expand the **View Source Citations** panel to show the exact source document, page number, chunk ID, and snippet.
* Ask an out-of-domain question (e.g., *"What is the recipe for lasagna?"* against a tech document).
* Show the anti-hallucination guardrail triggering: `"I couldn't find this information in the uploaded documents."`

### Step 4: Multi-Turn Conversation Memory & Export
* Demonstrate follow-up questions in the chat session.
* Click **Export Chat** to download the Markdown transcript showing all turns, timestamps, and citations.
* Click **Clear Chat** to show instant session history resetting.

### Step 5: Document Summarization (Multi-Style & Offline Fallback)
* Switch to the **Document Summarization** tab.
* Select **Executive Summary**, **Key Points**, or **Comprehensive Overview**.
* Show how large documents are condensed via chunked map-reduce. Mention the extractive frequency-based fallback that operates if no API key is set.

### Step 6: Sentiment & Intent NLP Features
* Switch to **Sentiment Analysis** to show document emotional tone classification with confidence distribution.
* Switch to **Intent Analysis** to show how incoming queries are categorized into *Question*, *Summary Request*, *Information Search*, or *Explanation Request*.

---

## 🧪 Running Automated Tests

Run the complete test suite using Python's built-in `unittest` runner:

```bash
python -m unittest discover tests
```

Or using `pytest`:

```bash
pytest
```

Run specific test modules:

```bash
# Test multi-format ingestion and cross-retrieval
python -m unittest tests/test_vector_search_integration.py

# Test RAG grounded generation and guardrails
python -m unittest tests/test_rag_pipeline.py

# Test document loaders and corrupt file handling
python -m unittest tests/test_document_loader.py

# Test NLP features (History, Summarization, Sentiment, Intent)
python -m unittest tests/test_nlp_features.py
```

---

## 🔒 Security & Privacy

* **No Hardcoded Keys**: API keys are never stored in the repository. All keys are loaded from environment variables (`.env`) or provided via the password-masked Streamlit UI input.
* **Ignored Secrets**: `.env` is explicitly declared in `.gitignore` to prevent accidental credential leakage.
* **In-Memory Storage**: FAISS vector indexes are generated in-memory during the application session and are not stored in unencrypted remote databases.
