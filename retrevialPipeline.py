from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv



#loading the environment
load_dotenv()
persist_directory="db/chroma_db"
embedding_model = OllamaEmbeddings(model="nomic-embed-text",base_url="http://localhost:11434")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microsoft pay to acquire GitHub?"


retriever = db.as_retriever(search_kwargs={"k": 5})

relevent_docs  = retriever.invoke(query)
print(f"User Query: {query}")
print("--- Context ---")
for i,doc in enumerate(relevent_docs,1):
    print(f"Document {i}: {doc}")


combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevent_docs])}


Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

#create a chat model
model = ChatOllama(
    model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)
# Display the full result and content only
print("\n--- Generated Response ---")
print("Full result:")
print(result)
print("Content only:")
print(result.content)