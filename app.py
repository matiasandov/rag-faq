import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime

from embedder import MultiDocumentRAG
from query_interface import OllamaQueryInterface

# Page configuration
st.set_page_config(
    page_title="Car RAG Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003478;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #003478;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #003478;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'query_interface' not in st.session_state:
    st.session_state.query_interface = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'collection_stats' not in st.session_state:
    st.session_state.collection_stats = None


def initialize_rag_system(collection_name, persist_dir, embedding_model):
    """Initialize the RAG system"""
    try:
        with st.spinner("Initializing RAG system..."):
            rag = MultiDocumentRAG(
                collection_name=collection_name,
                persist_directory=persist_dir,
                embedding_model=embedding_model
            )
            st.session_state.rag_system = rag
            st.session_state.collection_stats = rag.get_stats()
            return True, "RAG system initialized successfully!"
    except Exception as e:
        return False, f"Error initializing RAG: {str(e)}"


def initialize_query_interface(ollama_model, temperature):
    """Initialize the query interface"""
    if st.session_state.rag_system is None:
        return False, "Please initialize RAG system first"
    
    try:
        with st.spinner("Connecting to Ollama..."):
            query_interface = OllamaQueryInterface(
                embedder=st.session_state.rag_system.embedder,
                ollama_model=ollama_model,
                temperature=temperature
            )
            st.session_state.query_interface = query_interface
            return True, f"Connected to Ollama ({ollama_model})!"
    except Exception as e:
        return False, f"Error connecting to Ollama: {str(e)}"


def process_pdf(pdf_file, chunk_size, overlap):
    """Process uploaded PDF"""
    if st.session_state.rag_system is None:
        return False, "Please initialize RAG system first"
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{pdf_file.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        with st.spinner(f"Processing {pdf_file.name}..."):
            result = st.session_state.rag_system.process_pdf(
                temp_path,
                chunk_size=chunk_size,
                overlap=overlap
            )
        
        # Update stats
        st.session_state.collection_stats = st.session_state.rag_system.get_stats()
        
        # Clean up
        os.remove(temp_path)
        
        return True, result
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"


# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🚗 Car RAG Assistant</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # RAG Settings
        st.subheader("RAG System")
        collection_name = st.text_input("Collection Name", "ford_accessories")
        persist_dir = st.text_input("Persist Directory", "./chroma_db")
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ]
        )
        
        if st.button("Initialize RAG System"):
            success, message = initialize_rag_system(collection_name, persist_dir, embedding_model)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.divider()
        
        # LLM Settings
        st.subheader("LLM Configuration")
        ollama_model = st.selectbox(
            "Ollama Model",
            ["llama3.2", "mistral", "phi3", "gemma2", "qwen2.5"]
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        if st.button("Connect to Ollama"):
            success, message = initialize_query_interface(ollama_model, temperature)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        if st.session_state.rag_system:
            st.success("✅ RAG Initialized")
            if st.session_state.collection_stats:
                st.metric("Documents", st.session_state.collection_stats['total_documents'])
        else:
            st.warning("❌ RAG Not Initialized")
        
        if st.session_state.query_interface:
            st.success("✅ Ollama Connected")
        else:
            st.warning("❌ Ollama Not Connected")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Upload PDF", "📊 Collection Stats", "ℹ️ About"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("Ask Questions")
        
        if st.session_state.query_interface is None:
            st.warning("⚠️ Please initialize RAG system and connect to Ollama first")
        else:
            # Query input
            col1, col2 = st.columns([3, 1])
            with col1:
                question = st.text_input(
                    "Your question:",
                    placeholder="e.g., ¿Cuántos puntos se otorgan en el programa AOC DELL?"
                )
            with col2:
                n_results = st.number_input("Context chunks", 1, 10, 3)
            
            language = st.radio("Response Language", ["Español", "English"], horizontal=True)
            lang_code = "es" if language == "Español" else "en"
            
            if st.button("Ask", type="primary") and question:
                with st.spinner("Generating answer..."):
                    result = st.session_state.query_interface.query(
                        question=question,
                        n_results=n_results,
                        language=lang_code,
                        return_context=False
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now(),
                        "question": question,
                        "result": result
                    })
                
                # Display answer
                if result["success"]:
                    st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{result["answer"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Display sources
                    with st.expander(f"📚 View Sources ({len(result['sources'])})"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}:</strong> {source['section']}<br>
                                <strong>Relevance:</strong> {source['relevance']:.2%}<br>
                                <strong>Preview:</strong> {source['preview']}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("📝 Chat History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    with st.expander(f"{chat['timestamp'].strftime('%H:%M:%S')} - {chat['question'][:50]}..."):
                        st.write(f"**Q:** {chat['question']}")
                        st.write(f"**A:** {chat['result']['answer']}")
                
                if st.button("Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    # Tab 2: Upload PDF
    with tab2:
        st.header("Upload and Process PDF")
        
        if st.session_state.rag_system is None:
            st.warning("⚠️ Please initialize RAG system first")
        else:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", 100, 2000, 500, 100)
            with col2:
                overlap = st.number_input("Overlap", 0, 500, 50, 10)
            
            if uploaded_file and st.button("Process PDF"):
                success, result = process_pdf(uploaded_file, chunk_size, overlap)
                
                if success:
                    st.success("✅ PDF processed successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Sections", result['sections'])
                    col2.metric("Acronyms", result['acronyms'])
                    col3.metric("Processes", result['processes'])
                    col4.metric("Chunks", result['chunks'])
                else:
                    st.error(result)
    
    # Tab 3: Collection Stats
    with tab3:
        st.header("Collection Statistics")
        
        if st.session_state.collection_stats:
            stats = st.session_state.collection_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", stats['total_documents'])
                st.metric("Collection Name", stats['collection_name'])
            with col2:
                st.metric("Embedding Model", stats['embedding_model'])
                st.metric("Persist Directory", stats['persist_directory'])
            
            st.subheader("Metadata Fields")
            st.write(stats['metadata_fields'])
            
            if st.button("Refresh Stats"):
                if st.session_state.rag_system:
                    st.session_state.collection_stats = st.session_state.rag_system.get_stats()
                    st.rerun()
        else:
            st.info("Initialize RAG system to view statistics")
    
    # Tab 4: About
    with tab4:
        st.header("About Car RAG Assistant")
        
        st.markdown("""
        ### 🎯 Purpose
        This application uses Retrieval-Augmented Generation (RAG) to answer questions about Car documents.
        
        ### 🔧 How it Works
        1. **Document Processing**: PDFs are chunked and embedded into vectors
        2. **Vector Storage**: Chunks are stored in ChromaDB for fast retrieval
        3. **Query Processing**: Questions are embedded and matched against document chunks
        4. **Response Generation**: Ollama LLM generates answers based on retrieved context
        
        ### 🚀 Getting Started
        1. Initialize the RAG system in the sidebar
        2. Connect to Ollama (make sure `ollama serve` is running)
        3. Upload your PDF documents or use existing collection
        4. Start asking questions!
        
        ### 📋 Prerequisites
        ```bash
        # Install Ollama
        curl -fsSL https://ollama.com/install.sh | sh
        
        # Pull a model
        ollama pull llama3.2
        
        # Start Ollama server
        ollama serve
        
        # Install Python dependencies
        pip install -r requirements.txt
        ```
        
        ### 🛠️ Tech Stack
        - **Embeddings**: SentenceTransformers
        - **Vector DB**: ChromaDB
        - **LLM**: Ollama (local)
        - **UI**: Streamlit
        - **PDF Processing**: PyPDF2
        
        ### 📚 Supported Models
        - llama3.2 (recommended for Spanish)
        - mistral
        - phi3
        - gemma2
        - qwen2.5
        """)


if __name__ == "__main__":
    main()
