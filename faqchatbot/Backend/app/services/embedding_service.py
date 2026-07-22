import logging
import time
import threading
from google.genai import types
from app.config import settings
from app.services.gemini_key_pool import gemini_key_pool

logger = logging.getLogger(__name__)

# How long we're willing to keep retrying a single batch before giving up
# on it entirely. This is a *time* budget, not an attempt-count cap — the
# old fixed "4 attempts" cap gave up too early whenever a key got
# rate-limited again right after its cooldown ended (our local cooldown
# clock doesn't always line up with Gemini's real per-minute quota
# window), silently dropping real content as zero-embeddings. Retrying by
# wall-clock time means we keep trying for as long as it's reasonable to,
# regardless of how many individual attempts that takes.
BATCH_RETRY_BUDGET_SECONDS = 300.0  # 5 minutes

# Proactive pacing: a fixed pause between consecutive batch calls,
# independent of any rate-limit reaction. Reactive backoff (above) only
# kicks in *after* we already got a 429 — this pause exists so we avoid
# tripping the limit in the first place. Free-tier embedding quotas can be
# much stricter on newly created/unverified accounts than the general
# published numbers suggest, so this errs conservative rather than
# assuming a generous ceiling.
PACING_SECONDS_BETWEEN_BATCHES = 5.0


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not gemini_key_pool.available():
        logger.warning("No Gemini API keys available, returning zero embeddings")
        return [[0.0] * 1536 for _ in texts]

    all_embeddings: list[list[float]] = []
    batch_size = 100
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        embedded = False
        start_time = time.time()
        attempt = 0

        while time.time() - start_time < BATCH_RETRY_BUDGET_SECONDS:
            attempt += 1
            client, label = gemini_key_pool.get_client()
            if client is None:
                # No keys configured at all — nothing to retry against.
                break

            gemini_key_pool.wait_for_rate_slot(label)
            try:
                result = client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=1536),
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                gemini_key_pool.mark_success(label)
                embedded = True
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    gemini_key_pool.mark_rate_limited(label)
                    elapsed = time.time() - start_time
                    logger.warning(
                        f"{label} rate limited on batch {i}-{i+len(batch)} "
                        f"(attempt {attempt}, {elapsed:.0f}s/{BATCH_RETRY_BUDGET_SECONDS:.0f}s "
                        f"of retry budget used), rotating to next key"
                    )
                    # get_client() already blocks/waits when every key is in
                    # cooldown, so looping again naturally paces us rather
                    # than hammering the API — no extra sleep needed here.
                    continue
                else:
                    # Not a rate-limit error — retrying the same way won't
                    # help, so stop this batch immediately instead of
                    # burning the retry budget on a non-transient failure.
                    logger.error(f"Could not embed batch {i}-{i+len(batch)} using {label}: {e}")
                    break

        if not embedded:
            logger.error(
                f"Batch {i}-{i+len(batch)} ({len(batch)} chunks) failed after "
                f"{BATCH_RETRY_BUDGET_SECONDS:.0f}s of retrying across all available keys. "
                f"Using zero embeddings — these chunks will be SKIPPED on upsert and are "
                f"MISSING from the index. Consider re-running ingestion for this source "
                f"later, or increasing BATCH_RETRY_BUDGET_SECONDS if this recurs."
            )
            all_embeddings.extend([[0.0] * 1536 for _ in batch])

        # Proactive pacing: pause between batches regardless of whether
        # this one succeeded, failed, or hit rate limits. No need to pause
        # after the final batch.
        if batch_num < total_batches - 1:
            time.sleep(PACING_SECONDS_BETWEEN_BATCHES)

    return all_embeddings