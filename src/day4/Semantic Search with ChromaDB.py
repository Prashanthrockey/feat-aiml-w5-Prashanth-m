
import chromadb
import requests

client = chromadb.PersistentClient(path="./day4_chroma_data")

collection = client.get_or_create_collection(
    name="day4_documents",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "Python is used for programming.",
    "Machine learning learns patterns from data.",
    "Deep learning uses neural networks.",
    "Ollama runs language models locally.",
    "ChromaDB stores vector embeddings.",
    "LangGraph builds stateful AI workflows.",
    "CrewAI coordinates AI agents.",
    "MLflow tracks machine learning experiments.",
    "Ragas evaluates retrieval augmented generation.",
    "MLOps manages machine learning in production.",
    "Embeddings represent text as numerical vectors.",
    "Cosine similarity compares vector directions.",
    "Vector databases support similarity search.",
    "RAG retrieves relevant information for an LLM.",
    "System prompts guide language model behavior.",
    "Python dictionaries store key-value pairs.",
    "APIs allow software systems to communicate.",
    "PDF files can contain text and images.",
    "Metadata describes information about a document.",
    "Local LLMs can process prompts on-device."
]

categories = [
    "python", "ml", "ml", "ollama", "database",
    "framework", "framework", "mlops", "evaluation",
    "mlops", "embeddings", "search", "database",
    "rag", "prompting", "python", "api", "documents",
    "database", "ollama"
]


def embed(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


embeddings = [embed(doc) for doc in documents]

collection.upsert(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": category}
        for category in categories
    ]
)

print("Documents stored:", collection.count())

query = "How can AI find text with a similar meaning?"

results = collection.query(
    query_embeddings=[embed(query)],
    n_results=3
)

print("\nTop 3 similar documents:")
for doc in results["documents"][0]:
    print("-", doc)

filtered = collection.query(
    query_embeddings=[embed(query)],
    n_results=3,
    where={"category": "database"}
)

print("\nDatabase category results:")
for doc in filtered["documents"][0]:
    print("-", doc)
