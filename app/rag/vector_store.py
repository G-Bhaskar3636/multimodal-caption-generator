import chromadb

from rag.embedder import embed_text

DB_PATH = "./chroma_store"
COLLECTION_NAME = "caption_history"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def add_caption(entry_id: str, caption: str, metadata: dict) -> None:
    collection = _get_collection()
    vector = embed_text(caption)

    collection.add(
        ids=[entry_id],
        embeddings=[vector],
        documents=[caption],
        metadatas=[metadata],
    )


def search(query: str, top_k: int = 5) -> list[dict]:
    collection = _get_collection()
    query_vector = embed_text(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    matches = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({"caption": doc, "metadata": meta, "distance": dist})

    return matches