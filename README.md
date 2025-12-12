
# RAG Demo

A hands-on demo of a full Retrieval-Augmented Generation (RAG) system. It shows how to build a smart app that lets you ask questions about your own documents in natural language. The whole pipeline is here—from loading docs to answering questions conversationally, with chat history so follow-up questions make sense. It's got a nice interface and uses modern tools for LLMs and vector search. Great starting point if you're looking to build something production-like.

 Repo Structure

- **app.py**  
  The main Streamlit app—handles the UI, user sessions, ingestion, retrieval, and showing responses.

- **ingestionPipeline.py**  
  Takes care of loading docs from the "docs" folder, splitting them into chunks, creating embeddings, and saving everything to the vector database.

- **retrevialPipeline.py** (note: typo in the original—probably meant "retrieval")  
  Handles querying: turns your question into an embedding, searches for relevant chunks, and pulls the best matches.

- **historyAwareGeneration.py**  
  Keeps track of chat history and rewrites questions with context, so the system understands follow-ups and feels more like a real conversation.

- **docs/**  
  Drop your PDF/text files here—they'll get indexed when you run ingestion.

 Getting Started

1. Install the dependencies (list below).

2. Put your documents in the docs/ folder.

3. Run the ingestion pipeline to process and index them.

4. Fire up the app and start asking questions!

 Tech Stack

- LLM for generation and understanding
- Vector DB for fast similarity searches
- Embeddings for semantic matching
- Built-in memory for conversational context

 Dependencies

Just run these:

```bash
pip install langchain langchain-community langchain-ollama chromadb pypdf python-dotenv streamlit sentence-transformers
```

Key ones:
- LangChain (core + community + Ollama integration)
- ChromaDB for the vector store
- PyPDF for handling PDFs
- Sentence-transformers (though it pulls from Ollama here)
- Streamlit for the simple web UI
- python-dotenv for config

 Models

**LLM:**  
deepseek-r1:1.5b – A lightweight local model via Ollama. Runs fast, doesn't need much hardware, and works well for Q&A and chat.

**Embeddings:**  
nomic-embed-text – Solid embeddings from Ollama, perfect for retrieval tasks.

 Requirements

**Minimum:**
- Python 3.9+
- 8GB RAM (to run the 1.5B model comfortably)
- ~5GB disk space
- Windows/Linux/macOS

**Must-have:** Ollama installed (get it from ollama.ai)

Pull the models:

```bash
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
```

**Recommended:** 16GB RAM, decent CPU, SSD—makes everything snappier with bigger doc sets.

Best part: Fully local—no API keys, no internet needed after setup, everything stays on your machine for privacy.

 Setup

1. Grab Ollama from https://ollama.ai and install it.

2. Pull the models (commands above).

3. Optional: Add a .env file if you want to tweak things, like:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

That's it—drop in docs, ingest, and query away!
