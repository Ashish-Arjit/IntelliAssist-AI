# IntelliAssist AI — Smart Document AI Assistant

IntelliAssist AI is a modular document processing and analysis assistant designed to ingest, validate, clean, and chunk multi-format documents for downstream natural language processing and retrieval pipelines.

---

## 📌 Problem Statement

Unstructured documents (PDFs, Word documents, text files) often contain irregular formatting, redundant whitespace, varying structural layouts, and missing metadata. Before modern search or language models can effectively index or reason over document collections, documents must undergo rigorous validation, text extraction, non-destructive normalization, and semantic chunking while preserving lineage and metadata.

---

## 💡 Solution Overview

IntelliAssist AI provides an end-to-end, modular document ingestion foundation built with Python, LangChain, and Streamlit. It automates:
1. **Multi-Format Document Parsing**: Structured page-level and section-level text extraction from PDF, TXT, and DOCX files.
2. **Text Normalization**: Cleans excessive spaces, standardizes line breaks, and handles unicode characters while strictly preserving punctuation and semantic context.
3. **Configurable Semantic Chunking**: Splits normalized text using LangChain's `RecursiveCharacterTextSplitter` with customizable chunk size and overlap, ensuring full metadata continuity (page numbers, chunk indices, character/word counts).
4. **Validation & Error Handling**: Proactive validation for unsupported formats, empty files, size limits, and unreadable/scanned documents with user-friendly feedback.

---

## ✨ Current Features

* **Multi-Format Document Upload**: Support for `.pdf`, `.txt`, and `.docx` via an intuitive Streamlit interface.
* **Metadata-Preserving Document Loading**: Extracts source names, page numbers, line counts, character counts, and paragraph counts.
* **Non-Destructive Text Preprocessing**: Strips redundant whitespace and blank lines without aggressive stripping of numbers, symbols, or punctuation.
* **Configurable Chunking**: Dynamic adjustment of chunk size and overlap parameters in the Streamlit UI.
* **Interactive Chunk & Metadata Inspector**: Visual cards and expandable tabs to examine generated chunks, clean text previews, and document metrics in real time.
* **Validation & Security**: Built-in file type and content validation, `.env.example` templates, and strict exclusion of credentials and secrets.

---

## 🛠️ Technology Stack

* **Language**: Python 3.10+
* **User Interface**: Streamlit
* **Text Processing & Chunking**: LangChain Core (`langchain-core`), LangChain Text Splitters (`langchain-text-splitters`)
* **Document Extraction**:
  * PDF: `pypdf`
  * DOCX: `python-docx`
  * TXT: Built-in Python I/O with UTF-8 / Latin-1 fallback
* **Testing**: Python `unittest`

---

## 📁 Project Structure

```text
IntelliAssist-AI/
├── modules/
│   ├── __init__.py           # Package exports
│   ├── chunker.py            # LangChain document chunking implementation
│   ├── document_loader.py    # Modular PDF, TXT, and DOCX text extraction
│   ├── preprocessor.py       # Whitespace cleaning and normalization
│   └── validator.py          # File upload and content validation
├── utils/
│   └── __init__.py           # Utility helpers
├── tests/
│   ├── __init__.py           # Test suite package
│   ├── test_chunker.py       # Chunker unit tests
│   ├── test_document_loader.py # Loader unit tests (PDF, TXT, DOCX)
│   ├── test_pipeline.py      # End-to-end integration tests
│   ├── test_preprocessor.py  # Text preprocessor unit tests
│   └── test_validator.py     # Validator unit tests
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── app.py                    # Streamlit web application entrypoint
├── requirements.txt          # Pinned project dependencies
└── README.md                 # Project documentation
```

---

## 📄 Supported Document Formats

| Format | Extension | Extraction Method | Extracted Metadata |
|---|---|---|---|
| **Portable Document Format** | `.pdf` | `pypdf` | Page numbers, total pages, character count, file name, file type |
| **Plain Text** | `.txt` | Python UTF-8 / Latin-1 | Line count, character count, file name, file type |
| **Microsoft Word** | `.docx` | `python-docx` | Paragraph count, table count, character count, file name, file type |

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

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

---

## 💻 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the complete test suite using `unittest`:

```bash
python -m unittest discover -s tests
```
