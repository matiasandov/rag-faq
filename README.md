# Company RAG Assistant 🚗

A Retrieval-Augmented Generation (RAG) application for answering questions about Company documents using local LLMs via Ollama.

## 🎯 Features

- **PDF Document Processing**: Extract and chunk PDF documents for semantic search
- **Vector Storage**: Persist document embeddings in ChromaDB
- **Semantic Search**: Find relevant context using sentence transformers
- **Local LLM Generation**: Generate answers using Ollama (privacy-friendly)
- **Interactive UI**: Streamlit-based web interface
- **Multilingual Support**: Works with Spanish and English documents

## 🏗️ Architecture

```
User Question
    ↓
[Query Embedding] → Vector (384 dimensions)
    ↓
[ChromaDB Similarity Search] → Top K relevant chunks
    ↓
[Context Assembly] → Combine retrieved chunks
    ↓
[Ollama LLM] → Generate answer based on context
    ↓
Final Answer + Sources
```

## 📋 Prerequisites

### 1. Install Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### 2. Pull a Model

```bash
# Recommended for Spanish documents
ollama pull llama3.2

# Other options
ollama pull mistral
ollama pull phi3
ollama pull gemma2
```

### 3. Python 3.8+

Make sure you have Python 3.8 or higher installed.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama Server

```bash
ollama serve
```

Keep this terminal open!

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Initialize RAG System

1. In the sidebar, configure:
   - **Collection Name**: Name for your document collection
   - **Persist Directory**: Where to store the vector database
   - **Embedding Model**: Choose an embedding model
2. Click "Initialize RAG System"

### Step 2: Connect to Ollama

1. Select your Ollama model (e.g., `llama3.2`)
2. Adjust temperature (0.0 = deterministic, 1.0 = creative)
3. Click "Connect to Ollama"

### Step 3: Upload Documents

1. Go to the "Upload PDF" tab
2. Upload your PDF file
3. Configure chunk size and overlap
4. Click "Process PDF"

### Step 4: Ask Questions

1. Go to the "Chat" tab
2. Type your question
3. Select number of context chunks
4. Choose response language
5. Click "Ask"

## 🔧 Configuration Options

### Embedding Models

- `sentence-transformers/all-MiniLM-L6-v2` (default, fast, English)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)
- `sentence-transformers/all-mpnet-base-v2` (high quality, English)

### Ollama Models

- `llama3.2` - Best for Spanish, good general performance
- `mistral` - Fast, good for technical content
- `phi3` - Lightweight, fast inference
- `gemma2` - Good multilingual support
- `qwen2.5` - Excellent for technical documents

### Chunk Settings

- **Chunk Size**: 300-1000 characters (default: 500)
  - Smaller = more precise, but may miss context
  - Larger = more context, but less precise
- **Overlap**: 0-200 characters (default: 50)
  - Prevents information loss at chunk boundaries

## 📁 Project Structure

```
.
├── app.py                 # Streamlit application
├── embedder.py           # RAG embedding and ChromaDB logic
├── query_interface.py    # LLM query interface (Ollama)
├── scrapper.py          # PDF processing and extraction
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── chroma_db/           # Vector database (auto-created)
```

## 🔍 Example Queries

### Spanish
```
¿Cuántos puntos se otorgan en el programa AOC DELL?
¿Cuál es el proceso de gestión de accesorios?
¿Quién es el responsable del proceso de AOC?
Explica las fases del proceso de homologación
```

### English
```
How many points are awarded in the AOC DELL program?
What is the accessories management process?
Who is responsible for the AOC process?
Explain the phases of the approval process
```

## 🛠️ Advanced Usage

### Command Line Interface

You can also use the components directly from Python:

```python
from embedder import MultiDocumentRAG
from query_interface import OllamaQueryInterface

# Initialize RAG
rag = MultiDocumentRAG(
    collection_name="my_docs",
    persist_directory="./chroma_db"
)

# Process PDF
rag.process_pdf("document.pdf", chunk_size=500, overlap=50)

# Initialize query interface
query_interface = OllamaQueryInterface(
    embedder=rag.embedder,
    ollama_model="llama3.2"
)

# Ask question
result = query_interface.query(
    question="What is the AOC process?",
    n_results=5,
    language="en"
)

print(result["answer"])
```

### Batch Processing

```python
# Process multiple PDFs
rag.process_directory(
    directory_path="./pdfs",
    chunk_size=500,
    overlap=50
)
```

## 🐛 Troubleshooting

### Ollama Connection Error

**Problem**: `Could not connect to Ollama API`

**Solution**:
```bash
# Make sure Ollama is running
ollama serve

# Check if model is downloaded
ollama list

# Pull model if needed
ollama pull llama3.2
```

### Out of Memory

**Problem**: Application crashes with memory error

**Solutions**:
- Use smaller Ollama model (e.g., `phi3` instead of `llama3.2`)
- Reduce `n_results` (number of context chunks)
- Reduce `chunk_size`
- Use smaller embedding model

### Slow Responses

**Solutions**:
- Use faster Ollama model (`phi3`, `mistral`)
- Reduce `max_tokens` in query_interface.py
- Reduce `n_results` (fewer context chunks)
- Ensure SSD for ChromaDB storage

### Poor Answer Quality

**Solutions**:
- Increase `n_results` (more context)
- Increase `chunk_size` (more complete chunks)
- Use better embedding model (paraphrase-multilingual)
- Use more capable Ollama model
- Adjust temperature (lower = more factual)

## 📊 Performance Tips

1. **First Run**: First query will be slower (model loading)
2. **GPU**: Ollama will use GPU if available (much faster)
3. **Persistence**: ChromaDB persists data, no need to re-process
4. **Batch Processing**: Process multiple PDFs in one session

## 🔒 Privacy

- All processing happens locally
- No data sent to external APIs
- Documents stored locally in ChromaDB
- Ollama runs entirely on your machine

## 📝 License

This project is for internal Company use. Please review company policies before sharing externally.

## 🤝 Contributing

To add new features or improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For issues or questions:
- Check the Troubleshooting section
- Review Ollama documentation: https://ollama.com
- Review ChromaDB docs: https://docs.trychroma.com

## 🎓 Learning Resources

- **RAG Fundamentals**: https://www.pinecone.io/learn/retrieval-augmented-generation/
- **Ollama Guide**: https://github.com/ollama/ollama
- **ChromaDB Tutorial**: https://docs.trychroma.com/getting-started
- **Streamlit Docs**: https://docs.streamlit.io


## Workflow

1. scapper.py: Extract text and creates chunks with metadata

2. embedder.py: 
Initially
 - Embeds chunks
- creates ChromaDB (vector DB) 

While receiving user query
- enables similarity search
- retrieves top K results

3. query_interface.py:

```
embeds user query -> similairty search in DB -> retreieve results from DB using *embedder.py::query function* -> format context exctracted -> sends prmpt with results to LLM -> generates answer 
```

4. app.py: user interface with streamlit with chat history and coordinates rest of the files

## Technical Concepts

### Retriever used : Dense (Chroma's default one)
- K-Nearest Neighbors (KNN) Search
Instead of searching for exact keyword matches (like "AOC" or "DELL"), When you trigger the query() function, ChromaDB takes the vector (since we used an embedded model) of your question and plots it in a high-dimensional mathematical space. It then calculates the "distance" between your question's vector and all the document chunk vectors stored in the database.

### The Embedding Model: all-MiniLM-L6-v2
Its only job is to take text (either your PDF chunks or your user's search query) and translate it into a 384-dimensional mathematical array (a dense vector).