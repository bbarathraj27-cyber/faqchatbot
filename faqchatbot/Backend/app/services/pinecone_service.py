import logging
from pinecone import Pinecone, ServerlessSpec
import asyncio

from app.config import settings
from app.services.chunking_service import Chunk

logger = logging.getLogger(__name__)

_pc = None
_index = None  # cached Index client - created lazily, reused across calls
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


async def async_ensure_index_exists() -> None:
    await asyncio.to_thread(ensure_index_exists)


def _get_index():
    global _index
    if not _pc:
        raise RuntimeError("Pinecone is not available")
    if _index is None:
        _index = _pc.Index(settings.PINECONE_INDEX_NAME)
    return _index


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


async def async_upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]], topics_per_chunk: list[list[str]] | None = None) -> None:
    await asyncio.to_thread(upsert_chunks, chunks, embeddings, topics_per_chunk)


def query_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    if not _pc:
        logger.warning("Pinecone not available, returning empty results")
        return []
    
    try:
        index = _get_index()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        return [
            {
                "id": match.id,
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


async def async_query_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    return await asyncio.to_thread(query_similar_chunks, query_embedding, top_k)


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
                "id": match.id,
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


async def async_query_by_topics(topics: list[str], top_k: int = 5) -> list[dict]:
    return await asyncio.to_thread(query_by_topics, topics, top_k)


def get_chunks_by_source(source_url: str, top_k: int = 10000) -> list[dict]:
    """
    Fetches every chunk currently stored for a given source_url, including
    its text, so callers (e.g. the verification agent) can diff it against
    freshly scraped content without needing to keep raw text elsewhere.
    """
    if not _pc:
        return []
    try:
        index = _get_index()
        zero_vector = [0.0] * settings.EMBEDDING_DIMENSION
        results = index.query(
            vector=zero_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"source_url": {"$eq": source_url}},
        )
        return [
            {
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "question": match.metadata.get("question", ""),
            }
            for match in results.matches
        ]
    except Exception as e:
        logger.warning(f"Could not fetch chunks for source {source_url}: {e}")
        return []


async def async_get_chunks_by_source(source_url: str, top_k: int = 10000) -> list[dict]:
    return await asyncio.to_thread(get_chunks_by_source, source_url, top_k)


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


async def async_delete_chunks_by_source(source_url: str) -> int:
    return await asyncio.to_thread(delete_chunks_by_source, source_url)


def delete_chunks_by_ids(chunk_ids: list[str]) -> int:
    """
    Deletes an explicit list of vector IDs, as opposed to
    delete_chunks_by_source which deletes everything currently matching a
    source_url filter. Used by the re-ingest/correction flow: capture the
    OLD chunk IDs before upserting the new ones, then delete precisely
    those old IDs afterward - since the new chunks share the same
    source_url, a post-upsert delete_chunks_by_source(url) would also wipe
    the vectors that were just written.
    """
    if not _pc or not chunk_ids:
        return 0
    try:
        index = _get_index()
        index.delete(ids=chunk_ids)
        return len(chunk_ids)
    except Exception as e:
        logger.warning(f"Could not delete chunks by id: {e}")
        return 0


async def async_delete_chunks_by_ids(chunk_ids: list[str]) -> int:
    return await asyncio.to_thread(delete_chunks_by_ids, chunk_ids)


def delete_all_chunks() -> None:
    if not _pc:
        return
    try:
        index = _get_index()
        index.delete(delete_all=True)
    except Exception as e:
        logger.warning(f"Could not delete all chunks: {e}")


async def async_delete_all_chunks() -> None:
    await asyncio.to_thread(delete_all_chunks)


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


async def async_get_index_stats() -> dict:
    return await asyncio.to_thread(get_index_stats)