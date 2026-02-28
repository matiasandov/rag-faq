# Quick Start Guide - Car RAG Assistant

## 🚀 Setup (5 minutes)

### 1. Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama Server
```bash
# In a separate terminal, keep this running
ollama serve
```

## 🎯 Usage

### Option A: Web UI (Recommended)

```bash
streamlit run app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Click "Initialize RAG System" in sidebar
3. Click "Connect to Ollama" in sidebar
4. Upload PDF or use existing collection
5. Start asking questions!

### Option B: Command Line

```bash
# Process a PDF
python cli.py process-pdf accesorios.pdf

# Ask a single question
python cli.py query "¿Cuántos puntos AOC DELL?"

# Interactive mode
python cli.py interactive

# Show stats
python cli.py stats
```

## 📝 Example Questions

```
¿Cuántos puntos se otorgan en el programa AOC DELL?
¿Cuál es el proceso de gestión de accesorios?
¿Quién es el responsable del proceso de AOC?
Explica las fases del proceso de homologación
¿Qué áreas están involucradas en el proceso?
```

## 🔧 Common Issues

### "Could not connect to Ollama"
→ Make sure `ollama serve` is running in another terminal

### "Model not found"
→ Run `ollama pull llama3.2`

### Slow responses
→ Try smaller model: `ollama pull phi3`

### Out of memory
→ Reduce n_results in query settings

## 📊 Tips

- **First query is slow**: Model needs to load
- **Use GPU**: Ollama auto-detects, much faster
- **Multilingual**: Use `paraphrase-multilingual-MiniLM-L12-v2` embedding
- **Better answers**: Increase n_results (more context)
- **Faster**: Decrease n_results, use smaller model

## 🎓 Next Steps

1. Experiment with different models
2. Adjust chunk sizes for your documents
3. Try different embedding models
4. Fine-tune temperature for your use case

## 📞 Need Help?

Check the full README.md for detailed troubleshooting!
