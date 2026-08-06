import json
import hashlib
import logging
import asyncio

import redis
import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# Keep the existing synchronous client for callers that expect sync APIs.
redis_client = None
try:
    if settings.REDIS_URL:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    else:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            ssl=True,
            decode_responses=True,
        )
    redis_client.ping()
    logger.info("Redis cache connected successfully (sync client)")
except Exception as e:
    logger.warning(f"Could not connect to Redis cache (sync): {e}. Caching will be disabled for sync paths.")
    redis_client = None

# Async client is created lazily on first use so importing this module
# doesn't require Redis to be reachable.
_async_redis_client: aioredis.Redis | None = None


def _make_key(question: str) -> str:
    normalized = question.strip().lower()
    return "faq:" + hashlib.sha256(normalized.encode()).hexdigest()


def make_signature(chunk_ids: list[str]) -> str:
    """
    Fingerprint of the exact grounding chunks used to build an answer.
    If a re-ingest adds, removes, or replaces any of these chunks, the
    signature for the same question will differ on the next request —
    that's the freshness check question_api.py uses to decide whether
    a cached answer is still valid or must be regenerated.
    """
    normalized = sorted(set(chunk_ids))
    joined = "|".join(normalized)
    return hashlib.sha256(joined.encode()).hexdigest()


def get_cached_answer(question: str) -> dict | None:
    """
    Returns the full cached record (answer + sources + signature), or None.
    Caller is responsible for comparing the signature against fresh
    retrieval results to decide whether the cached answer is still valid.
    """
    if not redis_client:
        return None
    try:
        key = _make_key(question)
        cached = redis_client.get(key)
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.warning(f"Cache retrieval failed: {e}")
        return None


def set_cached_answer(question: str, answer_data: dict, signature: str) -> None:
    if not redis_client:
        return
    try:
        key = _make_key(question)
        payload = {**answer_data, "signature": signature}
        redis_client.setex(key, settings.CACHE_TTL_SECONDS, json.dumps(payload))
    except Exception as e:
        logger.warning(f"Cache storage failed: {e}")

def extend_cached_answer_ttl(question: str, extra_seconds: int) -> bool:
    """
    Called when a cached answer gets a 👍. Refreshes its Redis TTL so a
    verified-good answer sticks around longer instead of expiring on the
    normal CACHE_TTL_SECONDS schedule. Returns True if a cached entry was
    found and extended, False otherwise (e.g. nothing cached for this
    question, or it already expired).
    """
    if not redis_client:
        return False
    try:
        key = _make_key(question)
        if not redis_client.exists(key):
            return False
        redis_client.expire(key, extra_seconds)
        return True
    except Exception as e:
        logger.warning(f"Cache TTL extension failed: {e}")
        return False


def purge_cached_answer(question: str) -> bool:
    """
    Called when a cached answer gets a 👎. Deletes it from Redis immediately
    so the next person asking the same question gets a freshly regenerated
    answer instead of the same disliked one. Returns True if a cached entry
    was found and removed, False otherwise.
    """
    if not redis_client:
        return False
    try:
        key = _make_key(question)
        deleted = redis_client.delete(key)
        return deleted > 0
    except Exception as e:
        logger.warning(f"Cache purge failed: {e}")
        return False


def _make_correction_key(raw_question: str) -> str:
    normalized = raw_question.strip().lower()
    return "faq:correction:" + hashlib.sha256(normalized.encode()).hexdigest()


def get_cached_correction(raw_question: str) -> dict | None:
    """
    Returns the cached {"corrected": ..., "topics": [...]} result of
    correct_and_extract_topics() for this exact raw question, or None.

    Keyed on the RAW (uncorrected) question, unlike the answer cache which
    is keyed on the corrected question - the point is to avoid needing a
    Groq call just to find out what the corrected question is.
    """
    if not redis_client:
        return None
    try:
        key = _make_correction_key(raw_question)
        cached = redis_client.get(key)
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.warning(f"Correction cache retrieval failed: {e}")
        return None


def set_cached_correction(raw_question: str, corrected: str, topics: list[str]) -> None:
    if not redis_client:
        return
    try:
        key = _make_correction_key(raw_question)
        payload = {"corrected": corrected, "topics": topics}
        redis_client.setex(key, settings.CACHE_TTL_SECONDS, json.dumps(payload))
    except Exception as e:
        logger.warning(f"Correction cache storage failed: {e}")


async def _ensure_async_client() -> aioredis.Redis | None:
    global _async_redis_client
    if _async_redis_client is not None:
        return _async_redis_client
    try:
        if settings.REDIS_URL:
            _async_redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        else:
            _async_redis_client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                ssl=True,
                decode_responses=True,
            )
        # health check
        await _async_redis_client.ping()
        logger.info("Redis cache connected successfully (async client)")
        return _async_redis_client
    except Exception as e:
        logger.warning(f"Could not initialize async Redis client: {e}")
        _async_redis_client = None
        return None


async def async_get_cached_answer(question: str) -> dict | None:
    client = await _ensure_async_client()
    if not client:
        return None
    try:
        key = _make_key(question)
        cached = await client.get(key)
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.warning(f"Async cache retrieval failed: {e}")
        return None


async def async_set_cached_answer(question: str, answer_data: dict, signature: str) -> None:
    client = await _ensure_async_client()
    if not client:
        return
    try:
        key = _make_key(question)
        payload = {**answer_data, "signature": signature}
        await client.setex(key, settings.CACHE_TTL_SECONDS, json.dumps(payload))
    except Exception as e:
        logger.warning(f"Async cache storage failed: {e}")


async def async_extend_cached_answer_ttl(question: str, extra_seconds: int) -> bool:
    client = await _ensure_async_client()
    if not client:
        return False
    try:
        key = _make_key(question)
        if not await client.exists(key):
            return False
        await client.expire(key, extra_seconds)
        return True
    except Exception as e:
        logger.warning(f"Async cache TTL extension failed: {e}")
        return False


async def async_purge_cached_answer(question: str) -> bool:
    client = await _ensure_async_client()
    if not client:
        return False
    try:
        key = _make_key(question)
        deleted = await client.delete(key)
        return deleted > 0
    except Exception as e:
        logger.warning(f"Async cache purge failed: {e}")
        return False


async def async_get_cached_correction(raw_question: str) -> dict | None:
    client = await _ensure_async_client()
    if not client:
        return None
    try:
        key = _make_correction_key(raw_question)
        cached = await client.get(key)
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.warning(f"Async correction cache retrieval failed: {e}")
        return None


async def async_set_cached_correction(raw_question: str, corrected: str, topics: list[str]) -> None:
    client = await _ensure_async_client()
    if not client:
        return
    try:
        key = _make_correction_key(raw_question)
        payload = {"corrected": corrected, "topics": topics}
        await client.setex(key, settings.CACHE_TTL_SECONDS, json.dumps(payload))
    except Exception as e:
        logger.warning(f"Async correction cache storage failed: {e}")


# Convenience sync wrappers that call the async APIs when used from
# synchronous contexts. Avoid calling these from an active event loop.
def async_to_sync(coro, /, *args, **kwargs):
    return asyncio.run(coro(*args, **kwargs))


def get_cached_answer_sync(question: str) -> dict | None:
    return async_to_sync(async_get_cached_answer, question)


def set_cached_answer_sync(question: str, answer_data: dict, signature: str) -> None:
    return async_to_sync(async_set_cached_answer, question, answer_data, signature)


def get_cached_correction_sync(raw_question: str) -> dict | None:
    return async_to_sync(async_get_cached_correction, raw_question)


def set_cached_correction_sync(raw_question: str, corrected: str, topics: list[str]) -> None:
    return async_to_sync(async_set_cached_correction, raw_question, corrected, topics)