from embedder import RAGEmbedder
from scrapper import PDFScraper
import json
from pathlib import Path
from scrapper import PDFScraper
# Initializes RAG System
#👁️👁️👁️wraps previous class and uses scraper methods👁️👁️👁️

class MultiDocumentRAG:
    """
    Process multiple PDFs and store them in ChromaDB.
    Acts purely as an ingestion pipeline.
    """
    
    def __init__(self, embedder: RAGEmbedder):
        """
        Initialize with an existing RAGEmbedder instance
        """
        self.embedder = embedder
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 50):
        """Process a single PDF and add to ChromaDB"""
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print(f"{'='*60}")
        
        scraper = PDFScraper(pdf_path)
        scraper.extract_text()           
        scraper.identify_sections()      
        chunks = scraper.chunk_for_rag(chunk_size, overlap)  
        
        acronyms = scraper.extract_acronyms()
        processes = scraper.extract_processes()
        
        # Add to ChromaDB via the injected embedder
        self.embedder.add_chunks(chunks)
        
        return {
            'pdf_path': pdf_path,
            'sections': len(scraper.sections),      
            'acronyms': len(acronyms),              
            'processes': len(processes),            
            'chunks': len(chunks)
        }
    
    def process_directory(self, directory_path: str, chunk_size: int = 500, overlap: int = 50, file_pattern: str = "*.pdf"):
        """Process all PDFs in a directory"""
        pdf_files = list(Path(directory_path).glob(file_pattern))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(str(pdf_file), chunk_size=chunk_size, overlap=overlap)
                results.append(result)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        return results

# Example usage
if __name__ == "__main__":
    # 1. Initialize the Data Layer
    db_embedder = RAGEmbedder(
        collection_name="company_accessories",
        persist_directory="./chroma_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. Initialize the Ingestion Service
    document_processor = MultiDocumentRAG(embedder=db_embedder)
    
    # Process the PDF
    result = document_processor.process_pdf("accesorios.pdf", chunk_size=500, overlap=50)
    
    print(f"\n{'='*60}\nProcessing Summary:\n{'='*60}")
    print(json.dumps(result, indent=2))
    
    # Get collection statistics directly from the database layer
    stats = db_embedder.get_collection_stats()
    print(f"\n{'='*60}\nCollection Statistics:\n{'='*60}")
    print(json.dumps(stats, indent=2))