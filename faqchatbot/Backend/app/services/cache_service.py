import json
import hashlib
import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

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
    logger.info("Redis cache connected successfully")
except Exception as e:
    logger.warning(f"Could not connect to Redis cache: {e}. Caching will be disabled.")
    redis_client = None


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