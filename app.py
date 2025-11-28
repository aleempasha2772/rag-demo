from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM as Ollama

print("All packages installed successfully!")

# Load and process documents
loader = PyPDFLoader("Aleem_Pasha_Java_Springboot.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Print sample content to verify loading worked
print("\nSample chunk content:")
print(chunks[0].page_content[:200] + "...")

# Initialize models
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = Ollama(model="deepseek-r1:1.5b")
    print("Models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {e}")
    exit()

# Create vector store
vector_store = Chroma.from_documents(chunks, embeddings)
print("Vector store created successfully")


# Simple Q&A function
def ask_question(question):
    try:
        print(f"  Searching for: '{question}'")

        # Use similarity search directly
        relevant_docs = vector_store.similarity_search(question, k=2)

        print(f"  Found {len(relevant_docs)} relevant documents")

        if not relevant_docs:
            return "No relevant information found in the documents."

        # Combine context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create prompt
        prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: """

        # Get response from LLM
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error processing question: {str(e)}"


# Test the system
print("\n--- Testing RAG System ---")
questions = [
    "What is Aleem Pasha's phone number?",
    "What are Aleem Pasha's technical skills?",
    "What projects has he worked on at Tata Consultancy Services?",
    "What is his educational background?",
]

for question in questions:
    print(f"\nQ: {question}")
    answer = ask_question(question)
    print(f"A: {answer}")