import logging
import time
import threading
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)
_client = None
try:
    _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize Gemini client: {e}. Embeddings will be unavailable.")

# --- Global pacing: ensures a minimum gap between ANY two embedding calls,
#     regardless of which page/thread triggered them ---
_last_call_lock = threading.Lock()
_last_call_time = 0.0
MIN_SECONDS_BETWEEN_CALLS = 4.0  # tune based on your actual free-tier RPM limit


def _wait_for_rate_slot():
    global _last_call_time
    with _last_call_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        _last_call_time = time.time()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not _client:
        logger.warning("Gemini client not available, returning zero embeddings")
        return [[0.0] * 1536 for _ in texts]

    all_embeddings: list[list[float]] = []
    batch_size = 100
    max_retries = 4

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embedded = False

        for attempt in range(max_retries):
            _wait_for_rate_slot()
            try:
                result = _client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=1536),
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                embedded = True
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    wait_time = 20 * (attempt + 1)
                    logger.warning(
                        f"Rate limited on batch {i}-{i+len(batch)}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Could not embed batch {i}-{i+len(batch)}: {e}")
                    break

        if not embedded:
            logger.warning(
                f"Batch {i}-{i+len(batch)} failed after retries. Using zero embeddings (will be skipped on upsert)."
            )
            all_embeddings.extend([[0.0] * 1536 for _ in batch])

    return all_embeddings
