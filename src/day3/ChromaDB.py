
import chromadb
import requests

client = chromadb.PersistentClient(path="./chroma_data")

collection = client.get_or_create_collection(
    name="w5d3_documents",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "Python is a popular programming language.",
    "Machine learning learns patterns from data.",
    "Deep learning uses neural networks.",
    "Ollama runs language models locally.",
    "ChromaDB stores and searches vector embeddings.",
    "LangGraph helps build stateful AI workflows.",
    "CrewAI coordinates AI agents.",
    "MLflow tracks machine learning experiments.",
    "Ragas evaluates retrieval augmented generation.",
    "MLOps manages machine learning in production.",
    "Embeddings represent text as numerical vectors.",
    "Cosine similarity compares vector directions.",
    "A vector database supports similarity search.",
    "Retrieval augmented generation uses retrieved context.",
    "A system prompt guides model behavior.",
    "Python dictionaries store key-value pairs.",
    "APIs allow software systems to communicate.",
    "PDF files can contain text and images.",
    "Metadata describes additional information about data.",
    "Local LLMs can help keep prompts on-device."
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
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


ids = [f"doc_{i}" for i in range(len(documents))]

embeddings = [embed(doc) for doc in documents]

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": category}
        for category in categories
    ]
)

print("Documents stored:", collection.count())

query = "How can I search text by meaning?"
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
