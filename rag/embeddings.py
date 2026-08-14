from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from functools import lru_cache
from config.settings import settings
from rich.console import Console

console = Console()

# Initialize the embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence transformer model for embeddings (cached after first load)"""
    console.print("[blue]Loading embedding model...[/blue]")
    return SentenceTransformer(EMBEDDING_MODEL)


def get_chroma_client():
    """Get ChromaDB client with persistent storage"""
    chroma_dir = settings.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir)
    )
    return client


def get_or_create_collection(collection_name: str = "adas_knowledge"):
    """Get or create a ChromaDB collection"""
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "ADAS reports and domain knowledge"}
    )

    return collection


def add_document_to_rag(content: str, doc_id: str, metadata: dict | None = None):
    """Add a document to the RAG vector database"""
    if metadata is None:
        metadata = {}

    collection = get_or_create_collection()
    model = get_embedding_model()

    # Generate embedding
    embedding = model.encode(content).tolist()

    collection.add(
        documents=[content],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[metadata]
    )

    console.print(f"[green]Document added to RAG: {doc_id}[/green]")


def search_rag(query: str, n_results: int = 3) -> list:
    """Search the RAG database for relevant documents"""
    collection = get_or_create_collection()
    model = get_embedding_model()

    # Check if collection has documents
    if collection.count() == 0:
        console.print("[yellow]RAG database is empty[/yellow]")
        return []

    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count())
    )

    documents = results.get('documents', [[]])[0]
    console.print(
        f"[green]RAG found {len(documents)} relevant documents[/green]")

    return documents
