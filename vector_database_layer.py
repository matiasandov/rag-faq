import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from pathlib import Path
import os
from datetime import datetime

#👀👀👀 Data layer class👀👀👀
class RAGEmbedder:
    """
    Handles embedding generation and ChromaDB storage for RAG applications
    
    Mechanism:
    
    Query: "¿Cuántos puntos AOC DELL?"
      ↓
    [Embedding Model] → [Query Vector: [0.12, -0.45, 0.78, ...]]
      ↓
    [ChromaDB Similarity Search] → Compare with all document vectors
      ↓
    [Top K Results] → Return most similar chunks

    """
    
    def __init__(
        self, 
        collection_name: str = "company_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the embedder with ChromaDB
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: HuggingFace model name or 'openai' for OpenAI embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        """
        This model does TWO things:

        Encodes your documents → Converts text chunks into vector embeddings (384 dimensions)
        Encodes your queries → Converts search queries into vectors (same 384 dimensions)
        ChromaDB performs similarity search → Finds the closest document vectors to your query vector (using cosine similarity)

        """
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initializa embedding model - default is from transformers -> used in ChromaDB initialization
        self.embedding_function = self._get_embedding_function()
        
        # Get or create collection (uses embedded model to initilize it)
        self.collection = self._get_or_create_collection()
    
    def _get_embedding_function(self):
        """Configure the embedding function based on model choice"""
        
        if self.embedding_model == "openai":
            # OpenAI embeddings (requires OPENAI_API_KEY environment variable)
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        else:
            # HuggingFace sentence transformers (local, free)
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection( 
                name=self.collection_name,
                #uses embedding model
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection '{self.collection_name}' with {collection.count()} documents")
        except:
            # Create new collection if there is not an existing one
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Company RAG document collection"}
            )
            print(f"Created new collection '{self.collection_name}'")
        
        return collection
    
    def add_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add chunks to ChromaDB with embeddings
        
        Args:
            chunks: List of chunk dictionaries from PDFScraper
            batch_size: Number of chunks to process at once
        """
        total_chunks = len(chunks)
        print(f"Adding {total_chunks} chunks to ChromaDB...")
        
        #anades chunk per batch
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in batch:
                # Generate unique ID
                chunk_id = chunk.get('id', f"chunk_{i}_{datetime.now().timestamp()}")
                ids.append(chunk_id)
                
                # Extract text content
                documents.append(chunk['text'])
                
                # Prepare metadata (ChromaDB doesn't accept nested dicts)
                metadata = self._flatten_metadata(chunk.get('metadata', {}))
                metadatas.append(metadata)
            
            # 👀👀👀👀Add to collection (embeddings generated automatically)👀👀👀
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
        
        print(f"✓ Successfully added {total_chunks} chunks to ChromaDB")
        print(f"Total documents in collection: {self.collection.count()}")
    
    def _flatten_metadata(self, metadata: Dict) -> Dict:
        """
        Flatten nested metadata for ChromaDB compatibility
        ChromaDB only accepts: str, int, float, bool
        """
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, dict):
                # Flatten nested dicts
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)):
                        flattened[f"{key}_{nested_key}"] = nested_value
            elif value is not None:
                # Convert other types to string
                flattened[key] = str(value)
        
        return flattened
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Query the collection for relevant chunks
        
        Args:
            query_text: The search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"section": "AOC process"})
            
        Returns:
            Dictionary with results including documents, distances, and metadata
        """
        where_filter = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Sample some documents to get metadata keys
        sample = self.collection.peek(limit=10)
        metadata_keys = set()
        
        if sample['metadatas']:
            for metadata in sample['metadatas']:
                metadata_keys.update(metadata.keys())
        
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'metadata_fields': list(metadata_keys),
            'embedding_model': self.embedding_model
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection '{self.collection_name}'")
    
    def update_chunk(self, chunk_id: str, new_text: str, new_metadata: Dict):
        """Update a specific chunk"""
        self.collection.update(
            ids=[chunk_id],
            documents=[new_text],
            metadatas=[self._flatten_metadata(new_metadata)]
        )
    
    def delete_chunks(self, chunk_ids: List[str]):
        """Delete specific chunks by ID"""
        self.collection.delete(ids=chunk_ids)
        print(f"Deleted {len(chunk_ids)} chunks")

