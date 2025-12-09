import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your company files.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store with Ollama embeddings"""
    print("Creating embeddings using Ollama and storing in ChromaDB...")

    try:
        # Initialize Ollama embeddings with explicit base_url
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",  # Don't include :latest
            base_url="http://localhost:11434"  # Explicit Ollama server URL
        )

        # Test the embedding model first
        print("Testing embedding model...")
        test_embedding = embedding_model.embed_query("test")
        print(f"✅ Embedding model working! Dimension: {len(test_embedding)}")

    except Exception as e:
        print(f"❌ Error initializing Ollama embeddings: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull the model: ollama pull nomic-embed-text")
        print("3. Test it: ollama run nomic-embed-text")
        print("4. Check Ollama is on port 11434: curl http://localhost:11434")
        raise

    print("--- Creating vector store ---")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("--- Finished creating vector store ---")
        print(f"Vector store created and saved to {persist_directory}")
        return vectorstore

    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        raise


def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline with Ollama Embeddings ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. Loading existing store...")

        try:
            embedding_model = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )

            vectorstore = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
            return vectorstore

        except Exception as e:
            print(f"❌ Error loading existing vector store: {e}")
            print("You may need to delete the db folder and recreate it.")
            raise

    print("Persistent directory does not exist. Initializing vector store...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store with Ollama embeddings
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()