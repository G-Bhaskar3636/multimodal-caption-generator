import uuid
from datetime import datetime, timezone

from rag.vector_store import add_caption, search


def store_caption(
    english_caption: str,
    translated_caption: str,
    language: str,
    media_type: str = "image",
) -> str:
    entry_id = str(uuid.uuid4())

    add_caption(
        entry_id=entry_id,
        caption=english_caption,
        metadata={
            "translated_caption": translated_caption,
            "language": language,
            "media_type": media_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return entry_id


def search_history(query: str, top_k: int = 5) -> list[dict]:
    return search(query, top_k=top_k)