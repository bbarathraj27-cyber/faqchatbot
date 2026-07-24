import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types
from app.config import settings
from app.services.gemini_key_pool import gemini_key_pool

logger = logging.getLogger(__name__)

BATCH_RETRY_BUDGET_SECONDS = 300.0  # 5 minutes

# NOTE: batches now run concurrently across the Gemini key pool (see
# embed_texts below), so there's no fixed inter-batch sleep anymore.
# Pacing within a single key is still enforced per-call by
# gemini_key_pool.wait_for_rate_slot().


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def _embed_one_batch(batch: list[str], batch_label: str) -> list[list[float]]:
    """Embeds a single batch, retrying across the key pool within the
    per-batch time budget. Returns zero-vectors for the batch if every
    attempt fails."""
    start_time = time.time()
    attempt = 0

    while time.time() - start_time < BATCH_RETRY_BUDGET_SECONDS:
        attempt += 1
        client, label = gemini_key_pool.get_client()
        if client is None:
            break

        gemini_key_pool.wait_for_rate_slot(label)
        try:
            result = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=1536),
            )
            gemini_key_pool.mark_success(label)
            return [e.values for e in result.embeddings]
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                gemini_key_pool.mark_rate_limited(label)
                elapsed = time.time() - start_time
                logger.warning(
                    f"{label} rate limited on batch {batch_label} "
                    f"(attempt {attempt}, {elapsed:.0f}s/{BATCH_RETRY_BUDGET_SECONDS:.0f}s "
                    f"of retry budget used), rotating to next key"
                )
                continue
            else:
                logger.error(f"Could not embed batch {batch_label}: {e}")
                break

    logger.error(
        f"Batch {batch_label} ({len(batch)} chunks) failed after "
        f"{BATCH_RETRY_BUDGET_SECONDS:.0f}s of retrying across all available keys. "
        f"Using zero embeddings — these chunks will be SKIPPED on upsert."
    )
    return [[0.0] * 1536 for _ in batch]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not gemini_key_pool.available():
        logger.warning("No Gemini API keys available, returning zero embeddings")
        return [[0.0] * 1536 for _ in texts]

    batch_size = 100
    batch_starts = list(range(0, len(texts), batch_size))
    results: list[list[list[float]] | None] = [None] * len(batch_starts)

    max_workers = max(1, len(gemini_key_pool._entries))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, start in enumerate(batch_starts):
            batch = texts[start:start + batch_size]
            label = f"{start}-{start + len(batch)}"
            future = executor.submit(_embed_one_batch, batch, label)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    all_embeddings: list[list[float]] = []
    for batch_embeddings in results:
        all_embeddings.extend(batch_embeddings)
    return all_embeddings