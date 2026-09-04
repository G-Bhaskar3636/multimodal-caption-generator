"""
Conversation memory for the agent — a SEPARATE persistent vector store
(agent_chroma_store) from RAG's caption-history store (chroma_store).
RAG remembers what images/captions existed; Memory remembers what was
discussed in past conversation turns.
"""

import uuid
from datetime import datetime, timezone

import chromadb

from rag.embedder import embed_text  # reuse the same embedding model

DB_PATH = "./agent_chroma_store"
COLLECTION_NAME = "agent_memory"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def store_turn(session_id: str, user_question: str, final_answer: str) -> None:
    """Save one conversation turn (question + answer) into memory."""
    collection = _get_collection()
    combined_text = f"Q: {user_question}\nA: {final_answer}"
    vector = embed_text(combined_text)

    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[vector],
        documents=[combined_text],
        metadatas=[{
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    )


def recall_relevant_turns(query: str, session_id: str, top_k: int = 3) -> list[str]:
    """
    Retrieve past conversation turns relevant to the current query, scoped
    to this session. Returns a list of "Q: ... A: ..." strings.
    """
    collection = _get_collection()
    query_vector = embed_text(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"session_id": session_id},
    )

    documents = results.get("documents", [[]])[0]
    return documents