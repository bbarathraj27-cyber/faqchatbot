import logging
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.services.chunking_service import Chunk

logger = logging.getLogger(__name__)

_pc = None
try:
    _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    logger.info("Pinecone client initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize Pinecone client: {e}. Vector search will be disabled.")


def ensure_index_exists() -> None:
    if not _pc:
        logger.warning("Pinecone not available, skipping index creation")
        return
    try:
        existing_indexes = [index.name for index in _pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            _pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT),
            )
    except Exception as e:
        logger.warning(f"Could not ensure index exists: {e}")


def _get_index():
    if not _pc:
        raise RuntimeError("Pinecone is not available")
    return _pc.Index(settings.PINECONE_INDEX_NAME)


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]], topics_per_chunk: list[list[str]] | None = None) -> None:
    if not _pc:
        logger.warning("Pinecone not available, skipping chunk upsert")
        return
    
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    try:
        index = _get_index()
        vectors = []
        skipped = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if all(v == 0.0 for v in embedding):
                skipped += 1
                continue
            topics = topics_per_chunk[i] if topics_per_chunk else []
            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "source_url": chunk.source_url,
                    "question": chunk.question or "",
                    "topics": topics,
                },
            })

        if skipped:
            logger.warning(f"Skipped {skipped} chunks with zero embeddings (not stored)")

        batch_size = 100
        stored = 0
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size]
            try:
                index.upsert(vectors=batch)
                stored += len(batch)
            except Exception as e:
                logger.warning(f"Could not upsert batch {start}-{start+len(batch)}: {e}")

        logger.info(f"Stored {stored}/{len(chunks)} chunks in Pinecone")
    except Exception as e:
        logger.warning(f"Could not upsert chunks to Pinecone: {e}")


def query_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    if not _pc:
        logger.warning("Pinecone not available, returning empty results")
        return []
    
    try:
        index = _get_index()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        return [
            {
                "text": match.metadata.get("text", ""),
                "source_url": match.metadata.get("source_url", ""),
                "question": match.metadata.get("question", ""),
                "topics": match.metadata.get("topics", []),
                "score": match.score,
            }
            for match in results.matches
        ]
    except Exception as e:
        logger.warning(f"Could not query similar chunks: {e}")
        return []


def query_by_topics(topics: list[str], top_k: int = 5) -> list[dict]:
    """
    Fetches chunks tagged with any of the given topics (metadata filter),
    used for the KAG-driven part of retrieval alongside vector search.
    """
    if not _pc or not topics:
        return []

    try:
        index = _get_index()
        # Pinecone requires a query vector even for metadata-only filtering,
        # so we use a zero vector and rely purely on the filter.
        zero_vector = [0.0] * settings.EMBEDDING_DIMENSION

        results = index.query(
            vector=zero_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"topics": {"$in": topics}},
        )

        return [
            {
                "text": match.metadata.get("text", ""),
                "source_url": match.metadata.get("source_url", ""),
                "question": match.metadata.get("question", ""),
                "topics": match.metadata.get("topics", []),
                "score": match.score,
            }
            for match in results.matches
        ]
    except Exception as e:
        logger.warning(f"Could not query by topics: {e}")
        return []

def delete_chunks_by_source(source_url: str) -> int:
    if not _pc:
        return 0
    try:
        index = _get_index()
        zero_vector = [0.0] * settings.EMBEDDING_DIMENSION
        results = index.query(vector=zero_vector, top_k=10000, include_metadata=False,
                               filter={"source_url": {"$eq": source_url}})
        ids_to_delete = [match.id for match in results.matches]
        if not ids_to_delete:
            return 0
        index.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception as e:
        logger.warning(f"Could not delete chunks for source {source_url}: {e}")
        return 0


def delete_all_chunks() -> None:
    if not _pc:
        return
    try:
        index = _get_index()
        index.delete(delete_all=True)
    except Exception as e:
        logger.warning(f"Could not delete all chunks: {e}")


def get_index_stats() -> dict:
    if not _pc:
        return {"total_vectors": 0, "estimated_storage_mb": 0.0}
    try:
        index = _get_index()
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        bytes_per_vector = (settings.EMBEDDING_DIMENSION * 4) + 200
        estimated_mb = round((total_vectors * bytes_per_vector) / (1024 * 1024), 2)
        return {"total_vectors": total_vectors, "estimated_storage_mb": estimated_mb}
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
        return {"total_vectors": 0, "estimated_storage_mb": 0.0}