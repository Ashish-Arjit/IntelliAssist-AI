# IntelliAssist AI — Smart Document RAG Assistant & NLP Toolkit

IntelliAssist AI is a modular, production-ready Document AI and Retrieval-Augmented Generation (RAG) assistant designed for intelligent document question answering, summarization, sentiment evaluation, and query intent classification. It ingests multi-format documents (PDF, TXT, DOCX), extracts and normalizes text, splits content into semantic chunks, generates dense vector embeddings using Hugging Face Sentence Transformers, indexes vectors using an in-memory FAISS vector store, and integrates with **Google Gemini LLM** to deliver accurate, strictly grounded answers with verifiable source citations and advanced NLP capabilities.

---

## 📌 Complete Architecture & Pipeline Flow

```text
                                Uploaded Documents (PDF / TXT / DOCX)
                                                  ↓
                                       DocumentLoader & Validator
                                                  ↓
                                        TextPreprocessor
                                                  ↓
                                        DocumentChunker
                                                  ↓
                         ┌────────────────────────┴────────────────────────┐
                         ↓                                                 ↓
      Hugging Face Embeddings (MiniLM-L6-v2)                    Document Summarization
                         ↓                                (Executive / Key Points / Overview)
               FAISS Vector Store                                          ↓
                         ↓                                  Document Sentiment Analysis
             Dense Similarity Search                      (Positive / Negative / Neutral)
                         ↓
            Retrieved Context Chunks + Citations
                         ↓
  User Query ──► Query Intent Analyzer ──► Grounded RAG Pipeline ──► Grounded Answer + Citations
  (Question / Summary / Search / Explain)     (Google Gemini LLM)             ↓
                                                                     Conversation History
                                                                (Session Multi-Turn Memory)
```

---

## ✨ Key Features

### 1. 💬 Grounded Document Question Answering (RAG)
* **Strict Grounding**: Answers are synthesized strictly from retrieved document chunks.
* **Anti-Hallucination Guardrails**: Explicit fallback (`"I couldn't find this information in the uploaded documents."`) whenever relevant context is unavailable or insufficient.
* **Verified Source Citations**: Collapsible citation cards displaying the source file, PDF page number, chunk index, relevance score badges (0%–100%), and snippet previews.

### 2. 🕒 Session Conversation History
* **Multi-Turn Memory**: Preserves user questions, AI responses, timestamps, and citation metadata across interactions in Streamlit session state.
* **Previous Messages Visible**: Full chat history remains accessible as users ask follow-up questions or explore tabs.
* **Clear Conversation**: Easily clear session dialogue with a single click from the sidebar or chat toolbar.
* **Transcript Export**: Download the full conversation session as a formatted Markdown transcript.

### 3. 📝 Document Summarization
* **Upload-Grounded Summaries**: Generates comprehensive summaries based strictly on uploaded documents.
* **Multiple Summary Styles**:
  * **Executive Summary**: High-level synthesis of objectives, key findings, and conclusions.
  * **Key Points / Bullet Points**: Structured, actionable core insights.
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
* **In-Memory FAISS Vector Store**: Fast L2/Inner-Product similarity search with normalized percentage scores.

---

## 🛠️ Technology Stack

* **Core Framework**: Python 3.10+
* **LLM Engine**: Google Gemini (`gemini-3.6-flash` via `langchain-google-genai`)
* **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2` via `langchain-huggingface`)
* **Vector Store**: FAISS (`faiss-cpu`)
* **NLP & Sentiment**: Hugging Face Transformers (`transformers`, `torch`), Lexicon NLP
* **RAG Orchestration**: `langchain`, `langchain-core`, `langchain-text-splitters`
* **Web UI**: Streamlit
* **Document Loaders**: `pypdf`, `python-docx`
* **Testing**: Python standard library `unittest`

---

## 📁 Project Structure

```text
IntelliAssist-AI/
├── modules/
│   ├── __init__.py                  # Exports all core modules
│   ├── chunker.py                   # Recursive text chunking with metadata preservation
│   ├── document_loader.py           # Multi-format document loading (PDF, TXT, DOCX)
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
│   ├── test_document_loader.py      # Unit tests for loaders (PDF, TXT, DOCX)
│   ├── test_embeddings.py           # Unit tests for Hugging Face embeddings
│   ├── test_nlp_features.py         # Unit tests for conversation history, summarization, sentiment & intent
│   ├── test_pipeline.py             # Ingestion & search integration tests
│   ├── test_preprocessor.py         # Unit tests for text preprocessor
│   ├── test_rag_pipeline.py         # Unit & integration tests for RAG pipeline & LLM grounding
│   ├── test_validator.py            # Unit tests for document validator
│   ├── test_vector_search_integration.py # Multi-document vector search tests
│   └── test_vector_store.py         # Unit tests for FAISS vector store
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git ignore rules
├── app.py                           # Multi-tab Streamlit web application
├── config.py                        # Centralized application, NLP, and model configuration
├── requirements.txt                 # Project dependencies
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
LLM_MODEL_NAME=gemini-3.6-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=1024
DEFAULT_TOP_K=4
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
```

> **Note**: You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/). The API key can also be provided directly via the Streamlit sidebar input during runtime.

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
3. **📝 Document Summarization**: Choose a summary style (Executive, Key Points, Comprehensive) and generate concise, structured summaries.
4. **🎭 Sentiment Analysis**: Evaluate emotional tone (Positive, Negative, Neutral) and confidence scores of uploaded documents or custom excerpts.
5. **🎯 Intent Analysis**: Interactively test and verify query intents with sample test queries.
6. **🔍 Semantic Search**: Search the raw FAISS vector index and review nearest neighbors with relevance scores.
7. **🧩 Indexed Chunks**: Inspect all processed chunks, character lengths, and metadata.
8. **🔍 Metadata & Summary**: Review overall RAG architecture parameters and active NLP configuration.

---

## 🧪 Running Automated Tests

Run the test suite across the NLP features (conversation history, summarization, sentiment analysis, and intent classification):

```bash
python tests/test_nlp_features.py
```

Run the RAG pipeline and grounded answering tests:

```bash
python tests/test_rag_pipeline.py
```

Run the entire test suite:

```bash
python -m unittest discover -s tests
```
