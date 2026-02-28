#!/usr/bin/env python3
"""
Simple CLI for Company RAG Assistant
Usage: python cli.py [command] [options]
"""

import argparse
import sys
from pathlib import Path
from vector_database_layer import MultiDocumentRAG
from query_interface import OllamaQueryInterface


def init_rag(args):
    """Initialize RAG system"""
    print(f"Initializing RAG system...")
    rag = MultiDocumentRAG(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model
    )
    
    stats = rag.get_stats()
    print(f"\n✓ RAG System Initialized")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Embedding Model: {stats['embedding_model']}")
    
    return rag


def process_pdf(args):
    """Process a PDF file"""
    rag = init_rag(args)
    
    print(f"\nProcessing PDF: {args.pdf_path}")
    result = rag.process_pdf(
        args.pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    print(f"\n✓ PDF Processed Successfully")
    print(f"  Sections: {result['sections']}")
    print(f"  Acronyms: {result['acronyms']}")
    print(f"  Processes: {result['processes']}")
    print(f"  Chunks: {result['chunks']}")


def process_directory(args):
    """Process all PDFs in a directory"""
    rag = init_rag(args)
    
    print(f"\nProcessing directory: {args.directory}")
    results = rag.process_directory(
        args.directory,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    print(f"\n✓ Processed {len(results)} PDFs")
    for result in results:
        print(f"  - {result['pdf_path']}: {result['chunks']} chunks")


def query(args):
    """Query the RAG system"""
    rag = init_rag(args)
    
    # Initialize query interface
    print(f"Connecting to Ollama ({args.model})...")
    query_interface = OllamaQueryInterface(
        embedder=rag.embedder,
        ollama_model=args.model,
        temperature=args.temperature
    )
    
    # Process query
    print(f"\nQuestion: {args.question}")
    print("-" * 60)
    
    result = query_interface.query(
        question=args.question,
        n_results=args.n_results,
        language=args.language
    )
    
    if result["success"]:
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        if args.show_sources:
            print(f"📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  {i}. {source['section']}")
                print(f"     Relevance: {source['relevance']:.2%}")
                print(f"     Preview: {source['preview'][:100]}...")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def interactive(args):
    """Interactive query mode"""
    rag = init_rag(args)
    
    print(f"\nConnecting to Ollama ({args.model})...")
    query_interface = OllamaQueryInterface(
        embedder=rag.embedder,
        ollama_model=args.model,
        temperature=args.temperature
    )
    
    print("\n" + "="*60)
    print("Interactive Mode - Type 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("📝 Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not question:
                continue
            
            print("-" * 60)
            result = query_interface.query(
                question=question,
                n_results=args.n_results,
                language=args.language
            )
            
            if result["success"]:
                print(f"💡 Answer:\n{result['answer']}\n")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def stats(args):
    """Show collection statistics"""
    rag = init_rag(args)
    stats = rag.get_stats()
    
    print("\n" + "="*60)
    print("Collection Statistics")
    print("="*60)
    print(f"Collection Name: {stats['collection_name']}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Embedding Model: {stats['embedding_model']}")
    print(f"Persist Directory: {stats['persist_directory']}")
    print(f"\nMetadata Fields:")
    for field in stats['metadata_fields']:
        print(f"  - {field}")


def main():
    parser = argparse.ArgumentParser(
        description="Company RAG Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF
  python cli.py process-pdf accesorios.pdf
  
  # Process all PDFs in a directory
  python cli.py process-dir ./pdfs
  
  # Ask a question
  python cli.py query "¿Cuántos puntos AOC DELL?"
  
  # Interactive mode
  python cli.py interactive
  
  # Show statistics
  python cli.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--collection', default='ford_accessories', help='Collection name')
    common.add_argument('--persist-dir', default='./chroma_db', help='Persist directory')
    common.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model')
    
    # Process PDF command
    parser_pdf = subparsers.add_parser('process-pdf', parents=[common], help='Process a PDF file')
    parser_pdf.add_argument('pdf_path', help='Path to PDF file')
    parser_pdf.add_argument('--chunk-size', type=int, default=500, help='Chunk size')
    parser_pdf.add_argument('--overlap', type=int, default=50, help='Chunk overlap')
    parser_pdf.set_defaults(func=process_pdf)
    
    # Process directory command
    parser_dir = subparsers.add_parser('process-dir', parents=[common], help='Process directory')
    parser_dir.add_argument('directory', help='Directory containing PDFs')
    parser_dir.add_argument('--chunk-size', type=int, default=500, help='Chunk size')
    parser_dir.add_argument('--overlap', type=int, default=50, help='Chunk overlap')
    parser_dir.set_defaults(func=process_directory)
    
    # Query command
    parser_query = subparsers.add_parser('query', parents=[common], help='Query the RAG system')
    parser_query.add_argument('question', help='Question to ask')
    parser_query.add_argument('--model', default='llama3.2', help='Ollama model')
    parser_query.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser_query.add_argument('--n-results', type=int, default=5, help='Number of results')
    parser_query.add_argument('--language', choices=['es', 'en'], default='es', help='Language')
    parser_query.add_argument('--show-sources', action='store_true', help='Show sources')
    parser_query.set_defaults(func=query)
    
    # Interactive command
    parser_interactive = subparsers.add_parser('interactive', parents=[common], 
                                               help='Interactive query mode')
    parser_interactive.add_argument('--model', default='llama3.2', help='Ollama model')
    parser_interactive.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser_interactive.add_argument('--n-results', type=int, default=5, help='Number of results')
    parser_interactive.add_argument('--language', choices=['es', 'en'], default='es', help='Language')
    parser_interactive.set_defaults(func=interactive)
    
    # Stats command
    parser_stats = subparsers.add_parser('stats', parents=[common], help='Show statistics')
    parser_stats.set_defaults(func=stats)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
