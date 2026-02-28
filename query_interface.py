import requests
import json
from typing import List, Dict, Optional
from embedder import RAGEmbedder


class OllamaQueryInterface:
    """
    Interface for querying the RAG system and generating responses using Ollama
    
    Flow:
    1. User Query → Embed query using same model as documents
    2. ChromaDB → Retrieve most similar chunks (semantic search)
    3. Context Assembly → Combine retrieved chunks
    4. LLM Generation → Use Ollama to generate answer based on context
    """
    
    def __init__(
        self,
        embedder: RAGEmbedder,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the query interface
        
        Args:
            embedder: RAGEmbedder instance with loaded documents
            ollama_model: Name of the Ollama model to use
            ollama_url: URL of the Ollama API
            temperature: Temperature for response generation (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.embedder = embedder
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Verify Ollama is running
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.ollama_model not in model_names:
                    print(f"⚠️  Warning: Model '{self.ollama_model}' not found in Ollama.")
                    print(f"Available models: {model_names}")
                    print(f"Run: ollama pull {self.ollama_model}")
                else:
                    print(f"✓ Connected to Ollama. Using model: {self.ollama_model}")
            else:
                print("⚠️  Warning: Could not connect to Ollama API")
        except Exception as e:
            print(f"⚠️  Warning: Ollama connection error: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve relevant context from ChromaDB
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            filter_metadata: Optional filters for metadata
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        results = self.embedder.query(
            query_text=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def _format_context(self, results: Dict) -> str:
        """
        Format retrieved chunks into context string for LLM
        
        Args:
            results: Results from ChromaDB query
            
        Returns:
            Formatted context string
        """
        if not results['documents'][0]:
            return "No relevant context found."
        
        context_parts = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            section = metadata.get('section', 'Unknown Section')
            context_parts.append(
                f"[Context {i+1}] (Section: {section}, Relevance: {1-distance:.2f})\n{doc}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, language: str = "es") -> str:
        """
        Build the prompt for the LLM
        
        Args:
            query: User's question
            context: Retrieved context
            language: Response language (es/en)
            
        Returns:
            Complete prompt string
        """
        if language == "es":
            prompt = f"""Eres un asistente experto en documentos de Ford. Tu tarea es responder preguntas basándote únicamente en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA: {query}

INSTRUCCIONES:
- Responde ÚNICAMENTE basándote en el contexto proporcionado
- Si la información no está en el contexto, di "No encuentro información específica sobre esto en los documentos"
- Sé preciso y conciso
- Usa el mismo idioma de la pregunta
- Si hay números o datos específicos, cítalos exactamente

RESPUESTA:"""
        else:
            prompt = f"""You are an expert assistant for Company documents. Your task is to answer questions based solely on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer ONLY based on the provided context
- If the information is not in the context, say "I cannot find specific information about this in the documents"
- Be precise and concise
- Use the same language as the question
- If there are specific numbers or data, cite them exactly

ANSWER:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> Dict:
        """
        Generate response using Ollama
        
        Args:
            prompt: Complete prompt with context and question
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": self.ollama_model,
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "response": ""
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. The model might be too large or slow.",
                "response": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating response: {str(e)}",
                "response": ""
            }
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        language: str = "es",
        filter_metadata: Optional[Dict] = None,
        return_context: bool = False
    ) -> Dict:
        """
        Complete RAG query pipeline
        
        Args:
            question: User's question
            n_results: Number of context chunks to retrieve
            language: Response language (es/en)
            filter_metadata: Optional metadata filters
            return_context: Whether to return retrieved context in response
            
        Returns:
            Dictionary with answer and metadata
        """
        # Step 1: Retrieve relevant context
        retrieval_results = self.retrieve_context(
            query=question,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        # Step 2: Format context
        context = self._format_context(retrieval_results)
        
        # Step 3: Build prompt
        prompt = self._build_prompt(question, context, language)
        
        # Step 4: Generate response
        llm_result = self.generate_response(prompt)
        
        # Compile final result
        result = {
            "question": question,
            "answer": llm_result.get("response", ""),
            "success": llm_result.get("success", False),
            "model": self.ollama_model,
            "sources": [
                {
                    "section": meta.get("section", "Unknown"),
                    "relevance": 1 - dist,
                    "preview": doc[:200] + "..." if len(doc) > 200 else doc
                }
                for doc, meta, dist in zip(
                    retrieval_results['documents'][0],
                    retrieval_results['metadatas'][0],
                    retrieval_results['distances'][0]
                )
            ]
        }
        
        if return_context:
            result["context"] = context
            result["prompt"] = prompt
        
        if not llm_result.get("success"):
            result["error"] = llm_result.get("error", "Unknown error")
        
        return result
    
    def stream_query(self, question: str, n_results: int = 5, language: str = "es"):
        """
        Stream response generation (for real-time display)
        
        Args:
            question: User's question
            n_results: Number of context chunks
            language: Response language
            
        Yields:
            Response chunks as they're generated
        """
        # Retrieve and format context
        retrieval_results = self.retrieve_context(question, n_results)
        context = self._format_context(retrieval_results)
        prompt = self._build_prompt(question, context, language)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"


# Example usage
if __name__ == "__main__":
    from embedder import RAGEmbedder
    
    # 1. Initialize the Data Layer (Database)
    print("Loading database connection...")
    db_embedder = RAGEmbedder(
        collection_name="ford_accessories",
        persist_directory="./chroma_db"
    )
    
    # 2. Initialize the Generation Layer (Query Interface)
    print("Initializing query interface...")
    query_interface = OllamaQueryInterface(
        embedder=db_embedder,
        ollama_model="llama3.2",  
        temperature=0.7
    )
    
    # Example queries
    queries = [
        "¿Cuántos puntos se otorgan en el programa AOC DELL?",
        "¿Cuál es el proceso de gestión de accesorios?",
        "¿Quién es el responsable del proceso de AOC?"
    ]
    
    print("\n" + "="*60)
    print("RAG QUERY EXAMPLES")
    print("="*60)
    
    for query in queries:
        print(f"\n📝 Question: {query}")
        print("-" * 60)
        
        result = query_interface.query(question=query, n_results=3, language="es")
        
        if result["success"]:
            print(f"💡 Answer: {result['answer']}\n")
            print(f"📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['section']} (relevance: {source['relevance']:.2f})")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")