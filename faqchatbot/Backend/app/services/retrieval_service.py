import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def run_with_timeout(func, timeout_seconds: float, default=None, *args, **kwargs):
    """Run a callable with a hard timeout and return a fallback value on timeout."""
    if timeout_seconds <= 0:
        return default

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            future.cancel()
            logger.warning("Timed out after %.2fs", timeout_seconds)
            return default


def retrieve_context(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve context with timeout-based fallback to avoid long hangs."""
    from app.services.embedding_service import embed_text
    from app.services.neo4j_service import extract_topics
    from app.services.pinecone_service import query_similar_chunks, query_by_topics, search_by_text

    query_embedding = run_with_timeout(embed_text, 6.0, [], question)

    if not query_embedding or all(v == 0.0 for v in query_embedding):
        logger.warning("Embedding failed or timed out; using fallback retrieval")

        try:
            topics = run_with_timeout(extract_topics, 2.0, [], question)
            if topics:
                matches = run_with_timeout(query_by_topics, 4.0, [], topics, top_k)
                if matches:
                    return matches[:top_k]
        except Exception as exc:
            logger.warning("Topics-based fallback failed: %s", exc)

        try:
            matches = run_with_timeout(search_by_text, 4.0, [], question, top_k)
            if matches:
                return matches[:top_k]
        except Exception as exc:
            logger.warning("Text-based Pinecone fallback failed: %s", exc)

        return []

    matches = run_with_timeout(query_similar_chunks, 6.0, [], query_embedding, top_k)
    if not matches:
        logger.warning("Vector search returned no matches; attempting text fallback")
        try:
            matches = run_with_timeout(search_by_text, 4.0, [], question, top_k)
            if matches:
                return matches[:top_k]
        except Exception as exc:
            logger.warning("Text fallback after empty vector results failed: %s", exc)

    return matches[:top_k]
