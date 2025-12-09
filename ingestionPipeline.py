import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time

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


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    # Using RecursiveCharacterTextSplitter for better chunking
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\n✅ Created {len(chunks)} chunks")

    if chunks:
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Preview: {chunk.page_content[:200]}...")
            print("-" * 50)

        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store with Ollama embeddings - batch processing"""
    print(f"\nCreating embeddings for {len(chunks)} chunks using Ollama...")

    try:
        # Initialize Ollama embeddings
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
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
        print("3. Restart Ollama if needed")
        raise

    print("\n--- Creating vector store with batch processing ---")

    # Process in batches to avoid overwhelming Ollama
    BATCH_SIZE = 50  # Process 50 chunks at a time
    total_chunks = len(chunks)

    try:
        # Initialize ChromaDB
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print(f"Processing {total_chunks} chunks in batches of {BATCH_SIZE}...")

        # Process chunks in batches
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")

            try:
                # Add batch to vector store
                vectorstore.add_documents(batch)
                print("✅ Done")

                # Small delay to prevent overwhelming Ollama
                if i + BATCH_SIZE < total_chunks:
                    time.sleep(0.5)

            except Exception as batch_error:
                print(f"❌ Error in batch {batch_num}")
                print(f"Error: {batch_error}")
                print("Retrying with smaller batch...")

                # Retry with smaller batches
                for doc in batch:
                    try:
                        vectorstore.add_documents([doc])
                        print(".", end="", flush=True)
                    except Exception as e:
                        print(f"❌ Failed to add document: {e}")
                print()

        print("\n--- Finished creating vector store ---")
        print(f"✅ Vector store created with {vectorstore._collection.count()} documents")
        print(f"Saved to: {persist_directory}")

        return vectorstore

    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}")
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
            print("Delete the db folder and recreate: rm -rf db")
            raise

    print("Persistent directory does not exist. Initializing vector store...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

    # Step 3: Create vector store with batched processing
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n" + "=" * 70)
    print("✅ Ingestion complete! Your documents are now ready for RAG queries.")
    print("=" * 70)

    return vectorstore


if __name__ == "__main__":
    main()