# IntelliAssist AI: Smart Document AI Assistant

IntelliAssist AI is a modular, production-ready Document AI and Retrieval-Augmented Generation (RAG) assistant designed for intelligent document question answering, multi-style summarization, sentiment evaluation, and query intent classification. It ingests multi-format documents (PDF, TXT, DOCX), extracts and normalizes text, splits content into semantic chunks, generates dense vector embeddings using Hugging Face Sentence Transformers, indexes vectors using an in-memory FAISS vector store, and integrates with **Google Gemini LLM** to deliver accurate, strictly grounded answers with verifiable source citations and advanced NLP capabilities.

---

## 📌 Project Overview

IntelliAssist AI provides an end-to-end intelligent document question-answering system. Built with Python and Streamlit, it bridges the gap between raw document archives and actionable insights. By leveraging local high-dimensional embeddings and an in-memory vector store alongside a cloud-hosted LLM, the system keeps text parsing and semantic indexing local and fast, while utilizing Google Gemini for grounded natural language synthesis.

Key design principles:
* **Strict Grounding**: The assistant is strictly instructed to answer only from the provided document context, eliminating generative hallucination.
* **Verifiable Transparency**: Every answer links back to the exact source document, page number, chunk ID, and similarity relevance percentage.
* **Comprehensive NLP**: Includes multi-style document summarization, document sentiment tone analysis, and query intent classification.
* **Zero Database Overhead**: Operates with an in-memory FAISS flat index, making installation and deployment lightweight and immediate.

---

## ❓ Problem Statement

Organizations, researchers, and students routinely work with dense, multi-format documents such as research papers, legal agreements, technical manuals, and financial reports. 

Key challenges:
1. **Information Overload**: Manually scanning hundreds of pages across multiple formats to locate specific answers is inefficient and error-prone.
2. **Keyword Search Limitations**: Standard lexical search (Ctrl+F) requires exact word matches and fails to understand semantic meaning or synonyms.
3. **LLM Hallucinations**: Generic commercial LLMs lack access to private, unpublished, or domain-specific files and frequently generate plausible-sounding but factually incorrect assertions.
4. **Lack of Provenance**: Most automated question-answering systems do not cite the exact document section or page number where source evidence originated.

---

## 💡 Solution Overview

IntelliAssist AI solves these problems through an integrated Retrieval-Augmented Generation (RAG) pipeline:
* **Multi-Format Ingestion**: Unifies text extraction across PDF, TXT, and Word documents while preserving document and page-level metadata.
* **Dense Semantic Representation**: Converts text chunks into 384-dimensional dense vector embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`.
* **In-Memory Vector Search**: Employs FAISS (Facebook AI Similarity Search) to retrieve the most contextually relevant passages within milliseconds.
* **Constrained LLM Synthesis**: Google Gemini (`gemini-1.5-flash`) synthesizes answers strictly from retrieved context chunks. When relevant information is missing, it returns an explicit fallback message.
* **Granular Source Citations**: Each generated response includes collapsible source citation cards showing the file name, page number, chunk index, relevance score, and matched text snippet.
* **Modular NLP Toolkit**: Incorporates multi-style document summarization (with large-document map-reduce), sentiment analysis, and query intent detection.

---

## ✨ Features

### 1. 💬 Grounded Document Question Answering (RAG)
* **Strict Context Grounding**: Answers are synthesized strictly from retrieved document chunks.
* **Anti-Hallucination Guardrail**: Returns `"I couldn't find this information in the uploaded documents."` whenever context is absent or falls below the similarity threshold.
* **Customizable Retrieval Parameters**: Adjustable Top-$K$ retrieved chunks (1–10) and minimum similarity score threshold via the UI.

### 2. 📌 Verified Source Citations
* **Provenance Cards**: Collapsible cards accompanying each answer display the source document name, PDF page number, chunk index, and preview snippet.
* **Relevance Score Badges**: Normalized percentage badges ($\ge 70\%$ High, $\ge 40\%$ Medium, $< 40\%$ Low) derived from Euclidean vector distances.

### 3. 🕒 Session Conversation History
* **Multi-Turn Memory**: Maintains user queries, assistant answers, timestamps, and citation metadata across Streamlit sessions.
* **Transcript Export**: Download the full dialogue history as a formatted Markdown file (`intelliassist_chat_transcript.md`).
* **Session Reset**: Clear chat dialogue with a single click.

### 4. 📝 Document Summarization
* **Multi-Style Summaries**:
  * **Executive Summary**: High-level synthesis of objectives, findings, and conclusions.
  * **Key Points / Bullet Points**: Structured core takeaways with bulleted highlights.
  * **Comprehensive Overview**: Detailed, section-by-section breakdown.
* **Large-Document Map-Reduce**: Automatically breaks extensive documents into chunks and synthesizes them hierarchically to prevent token overflow.
* **Offline NLP Fallback**: Extractive frequency-based summarization fallback when an LLM API key is not configured.

### 5. 🎭 Document Sentiment Analysis
* **Emotional Tone Classification**: Evaluates document content into **Positive**, **Negative**, or **Neutral** sentiment.
* **Confidence Distribution**: Computes confidence scores across all three classes.
* **Flexible Scope**: Analyze entire uploaded files or inspect custom user excerpts.
* **Robust Fallback**: Uses Hugging Face Transformers with a rule-based lexicon fallback.

### 6. 🎯 Query Intent Analysis
* **Intent Categorization**: Classifies user queries into:
  * **Question**: Factual queries seeking specific answers.
  * **Summary Request**: Prompts requesting an overview or condensation.
  * **Information Search**: Targeted lookup for specific topics or entities.
  * **Explanation Request**: Inquiries exploring mechanisms or conceptual explanations.
* **Visual Intent Badges**: Rendered alongside user messages in the chat interface.

### 7. 🔍 Semantic Vector Search & Chunk Inspector
* **Raw Index Querying**: Test FAISS retrieval directly with semantic queries and inspect nearest neighbor chunks with relevance scores.
* **Chunk Inspector**: Browse all processed text chunks, token lengths, character counts, and metadata tags.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language |
| **Web Interface** | Streamlit ($\ge 1.35.0$) | Reactive user interface and session state management |
| **LLM Engine** | Google Gemini (`gemini-1.5-flash`) | Context-aware answer generation and synthesis |
| **LLM Integration** | `langchain-google-genai` | Interface for Google Gemini API |
| **Embedding Model** | Hugging Face `all-MiniLM-L6-v2` | Dense 384-dimensional vector embeddings |
| **Embedding Framework** | `langchain-huggingface`, `sentence-transformers` | Embedding generation pipeline |
| **Vector Store** | FAISS (`faiss-cpu`) | High-performance in-memory similarity search |
| **RAG Orchestration** | LangChain (`langchain`, `langchain-core`) | Prompt formatting and pipeline chaining |
| **Text Chunking** | `langchain-text-splitters` | Recursive character text splitting |
| **PDF Processing** | `pypdf` ($\ge 4.2.0$) | Page-level text extraction and metadata tracking |
| **Word Processing** | `python-docx` ($\ge 1.1.0$) | DOCX paragraph and table parsing |
| **Environment & Secrets** | `python-dotenv`, Streamlit Secrets | Secure configuration management |
| **Testing** | `unittest`, `pytest` ($\ge 8.0.0$) | Unit and integration test suites |

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

1. **Streamlit Interface (`app.py`)**: Multi-tab interface handling file uploads, user prompts, configuration sliders, and session state.
2. **Document Ingestion & Validation (`modules/validator.py`, `modules/document_loader.py`)**: Validates MIME types, extensions, and file sizes ($\le 25\text{ MB}$), then extracts text using format-specific loaders.
3. **Text Preprocessing (`modules/preprocessor.py`)**: Normalizes Unicode characters (NFKC), strips excessive whitespace, and cleans formatting while preserving punctuation and numbers.
4. **Text Chunking (`modules/chunker.py`)**: Splits documents into overlapping segments (default 1000 characters with 200 character overlap) while embedding metadata tags (`file_name`, `page`, `chunk_index`).
5. **Embedding Generation (`modules/embeddings.py`)**: Transforms text chunks into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
6. **FAISS Vector Index (`modules/vector_store.py`)**: Indexes dense embeddings in an in-memory FAISS flat L2 vector store.
7. **Semantic Retrieval**: Queries are embedded into the same 384-dimensional vector space, retrieving top-$k$ chunks based on distance.
8. **Relevance Scoring**: Converts Euclidean distances into normalized similarity percentages:
   $$\text{Similarity Score} = \max\left(0.0, 1.0 - \frac{\text{distance}^2}{2}\right)$$
9. **RAG Pipeline (`modules/rag_pipeline.py`)**: Combines retrieved context chunks, user query, and anti-hallucination instructions into a prompt for Google Gemini (`gemini-1.5-flash`).
10. **Source Citation Engine**: Constructs verified citation cards linking the answer to original document chunks.

---

## 🔄 RAG Workflow

The Retrieval-Augmented Generation (RAG) execution workflow operates through four sequential phases:

```text
[Document Upload] ──► [Extraction & Cleaning] ──► [Chunking & Embeddings] ──► [FAISS Index]
                                                                                   │
[User Query] ──► [Intent Analysis] ──► [Dense Retrieval] ──► [Context Filtering] ──┘
                                                                   │
                                                                   ▼
[Source Citations] ◄── [Answer Display] ◄── [Gemini LLM Generation]
```

1. **Document Ingestion Phase**:
   - The user uploads one or more supported files (`.pdf`, `.txt`, `.docx`).
   - `DocumentValidator` verifies file presence, size limits, and allowed extensions.
   - `DocumentLoader` extracts text and associates metadata (e.g., page numbers for PDFs).
   - `TextPreprocessor` normalizes text and removes noise.

2. **Embedding & Vector Indexing Phase**:
   - `DocumentChunker` breaks text into semantically cohesive overlapping chunks.
   - `EmbeddingManager` computes embeddings using `all-MiniLM-L6-v2`.
   - `VectorStoreManager` populates the in-memory FAISS index.
   - Processed chunks, metadata, and the FAISS index are cached in Streamlit's `session_state` to prevent re-computation on UI interactions.

3. **Query & Retrieval Phase**:
   - The user enters a question in the chat input.
   - `IntentAnalyzer` determines the user's intent.
   - The query is vectorized and FAISS retrieves the top-$k$ nearest neighbors.
   - Chunks falling below the minimum similarity threshold (default $0.20$) are discarded.

4. **Generation & Verification Phase**:
   - If no chunks pass the threshold, the system returns: `"I couldn't find this information in the uploaded documents."`
   - Otherwise, retrieved chunks are formatted into a constrained prompt sent to Gemini.
   - The synthesized response is displayed with citation cards and saved to the session history.

---

## 📁 Project Structure

```text
IntelliAssist-AI/
├── .streamlit/
│   └── config.toml                  # Streamlit Cloud server, theme, and upload configuration
├── modules/
│   ├── __init__.py                  # Public exports for modules package
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
│   ├── test_document_loader.py      # Unit tests for document loaders & corrupt file handling
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

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ashish-Arjit/IntelliAssist-AI.git
cd IntelliAssist-AI
```

### 2. Set Up a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

### 1. Configure the `.env` File

Copy the provided example file:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```env
# Google Gemini API Key (Required for RAG answers and summarization)
GOOGLE_API_KEY=your_actual_google_api_key_here

# LLM & RAG Settings
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-1.5-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=1024
MIN_SIMILARITY_THRESHOLD=0.20

# Search & Retrieval Settings
DEFAULT_TOP_K=4

# Document Processing Settings
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
MAX_UPLOAD_SIZE_MB=25

# Embedding Settings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
NORMALIZE_EMBEDDINGS=True
```

### 2. Obtaining an API Key

You can obtain a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

> **Note**: An API key can also be entered directly in the Streamlit sidebar during runtime without editing the `.env` file.

---

## 🚀 How to Run

Launch the Streamlit application:

```bash
streamlit run app.py
```

Once started, open your web browser and navigate to:

```text
http://localhost:8501
```

---

## 📄 Supported Document Formats

| Format | Extension | Extraction Library | Metadata Preserved |
| :--- | :--- | :--- | :--- |
| **PDF Document** | `.pdf` | `pypdf` | Document name, page number, chunk index, character count |
| **Plain Text** | `.txt` | Native Python (UTF-8 / Latin-1 / CP1252) | Document name, chunk index, character count |
| **Word Document** | `.docx` | `python-docx` | Document name, paragraphs, table cells, chunk index |

* **Max File Size**: 25 MB per file (configurable via `MAX_UPLOAD_SIZE_MB`).
* **Multi-File Upload**: Supports uploading multiple files simultaneously across different formats.

---

## 🖥️ Application Usage

Follow this standard user journey:

```text
Open Application ──► Upload Documents ──► Process Documents ──► Ask Question ──► View Answer & Citations ──► Explore NLP Features
```

1. **Upload Documents**: Use the sidebar uploader to select one or more PDF, TXT, or DOCX files.
2. **Configure Settings**:
   - Verify that your Gemini API key is detected or enter it in the sidebar.
   - Adjust chunking parameters (chunk size, overlap) and retrieval Top-$K$ as needed.
3. **Inspect Metrics**: Review the ingestion metrics banner displaying uploaded files, total sections/pages, word count, and indexed vector chunks.
4. **💬 Document Chatbot Tab**:
   - Ask questions about your documents in natural language.
   - Review the synthesized answer grounded in your files.
   - Check the detected query intent badge displayed above the user message.
   - Expand the **View Source Citations** panel to see exact document sources and relevance scores.
   - Click **Export Chat** to download a Markdown transcript of your session.
5. **📝 Document Summarization Tab**:
   - Choose a document scope (all files or a specific file).
   - Select a style: *Executive Summary*, *Key Points*, or *Comprehensive Overview*.
   - Click **Generate Document Summary** and download the resulting summary.
6. **🎭 Sentiment Analysis Tab**:
   - Select an uploaded file or enter custom text to analyze emotional tone (Positive, Negative, Neutral) with confidence metrics.
7. **🎯 Intent Analysis Tab**:
   - Test queries interactively to see intent classification results.
8. **🔍 Semantic Search & Indexed Chunks Tabs**:
   - Perform raw similarity searches against the FAISS index.
   - Inspect individual text chunks, token lengths, and metadata.

---

## 📌 Source Citations

IntelliAssist AI enforces transparent attribution for all synthesized responses:

* **Source File Name**: Identifies the exact document that provided context.
* **Page / Section**: Identifies the 1-indexed PDF page number or section.
* **Chunk Identifier**: References the specific indexed chunk number within the document.
* **Relevance Score**: Computed normalized similarity percentage based on FAISS vector distance:
  * 🟢 **High Relevance** ($\ge 70\%$): Strong conceptual match.
  * 🟡 **Medium Relevance** ($40\% - 69\%$): Partial conceptual match.
  * 🔴 **Low Relevance** ($< 40\%$): Weak match (filtered if below threshold).
* **Snippet Preview**: Displays the exact text excerpt retrieved from the document.

---

## ⚠️ Limitations

* **Scanned PDFs / OCR**: The current loader extracts programmatic text. Scanned image-only PDFs require pre-OCR processing.
* **In-Memory Vector Store**: Vector indexes are stored in memory (`session_state`). Restarting the Streamlit server or clearing the browser session clears the index.
* **Cloud LLM Dependency**: Generating natural language answers requires internet access to connect to the Google Gemini API. (Embeddings, FAISS search, and offline summarization fallback run entirely locally.)
* **API Rate Limits**: Response generation times depend on Google AI Studio API rate limits and quotas.

---

## 🔮 Future Scope

* **OCR Integration**: Incorporate Tesseract or easyOCR to automatically parse scanned PDF documents and image files.
* **Persistent Vector Databases**: Integrate disk-persisted vector databases (such as ChromaDB or LanceDB) to preserve indexes across sessions.
* **Hybrid Search (Dense + Sparse)**: Combine FAISS dense embeddings with BM25 keyword matching for optimal precision on technical terminology.
* **Additional File Formats**: Support Markdown (`.md`), PowerPoint (`.pptx`), CSV (`.csv`), and HTML files.
* **Local LLM Support**: Provide an option to run local quantized models (e.g., via Ollama or llama.cpp) for fully offline, air-gapped environments.

---

## ☁️ Deployment Instructions

### Deploying to Streamlit Cloud

IntelliAssist AI is fully pre-configured for deployment on **Streamlit Community Cloud**:

1. **Push to GitHub**:
   Ensure your repository is pushed to GitHub with all files (`app.py`, `requirements.txt`, `.streamlit/config.toml`, etc.).

2. **Access Streamlit Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.

3. **Create New App**:
   - Click **Create app**.
   - Select your repository: `Ashish-Arjit/IntelliAssist-AI` (or your fork).
   - Select Branch: `main`.
   - Set Main file path: `app.py`.

4. **Configure Secrets**:
   - Click **Advanced settings...**.
   - In the **Secrets** section, add your Google Gemini API key:
     ```toml
     GOOGLE_API_KEY = "your_actual_google_api_key_here"
     ```
   - (Optional) Configure additional settings if desired:
     ```toml
     LLM_MODEL_NAME = "gemini-1.5-flash"
     DEFAULT_TOP_K = "4"
     ```

5. **Deploy**:
   - Click **Deploy!**.
   - Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the application.

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
# Test multi-format ingestion and vector search
python -m unittest tests/test_vector_search_integration.py

# Test RAG grounded generation and guardrails
python -m unittest tests/test_rag_pipeline.py

# Test document loaders and error handling
python -m unittest tests/test_document_loader.py

# Test NLP features (History, Summarization, Sentiment, Intent)
python -m unittest tests/test_nlp_features.py
```

---

## 🔒 Security & Privacy

* **No Hardcoded Keys**: API keys are never stored in the repository. All keys are loaded from environment variables (`.env`), Streamlit Secrets, or entered directly in the password-masked UI.
* **Ignored Secrets**: Both `.env` and `.streamlit/secrets.toml` are explicitly listed in `.gitignore` to prevent credential exposure.
* **In-Memory Storage**: Vector embeddings are indexed in-memory and are not transmitted to third-party databases.
