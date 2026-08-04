"""
Verification service — the "correctness check" half of the verification
agent (see verification_agent.py for the looping/scheduling half).

For a single Source row, this:
  1. Re-scrapes the live webpage right now.
  2. Cleans + chunks it exactly the way the original ingestion pipeline did,
     so the comparison is apples-to-apples.
  3. Pulls whatever is currently stored in Pinecone for that source_url.
  4. Compares the two on word-similarity. If they're close enough, nothing
     changes (no wasted API calls). If they've drifted — the page changed,
     the old scrape was botched, whatever — it treats that as "incorrect"
     and re-ingests: re-embeds the fresh chunks with Gemini, re-extracts
     topics with Groq, upserts the corrected chunks into Pinecone, and only
     then deletes the stale vectors that used to represent this source.

Similarity is intentionally cheap (difflib on word tokens) rather than an
LLM call, so checking 230+ sources doesn't itself burn the API budget the
agent is trying to protect. The LLM/embedding calls only happen for
sources that actually need correcting.
"""

import difflib
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import ObjectDeletedError, StaleDataError

from app.models.source_model import Source
from app.services.scraper_service import scrape_faq_page
from app.services.cleaning_service import clean_text
from app.services.chunking_service import chunk_qa_pair, chunk_long_text, Chunk
from app.services.embedding_service import embed_texts
from app.services.neo4j_service import link_topics_from_texts_batch
from app.services.pinecone_service import (
    get_chunks_by_source,
    delete_chunks_by_ids,
    upsert_chunks,
)

logger = logging.getLogger(__name__)

# Below this word-similarity ratio, the stored content is considered stale
# / incorrect and gets re-ingested. 0.90 tolerates trivial whitespace/typo
# level noise but catches real content drift (updated docs, changed FAQ
# answers, a page that now 404s and fell back to different boilerplate).
SIMILARITY_THRESHOLD = 0.90


def _normalize_words(text: str) -> list[str]:
    return text.lower().split()


def compute_similarity(old_text: str, new_text: str) -> float:
    """Word-level similarity ratio between 0.0 (nothing alike) and 1.0
    (identical). Order-sensitive on purpose — reshuffled boilerplate should
    still score high, but truncated/replaced content should not."""
    if not old_text and not new_text:
        return 1.0
    if not old_text or not new_text:
        return 0.0
    old_words = _normalize_words(old_text)
    new_words = _normalize_words(new_text)
    matcher = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    return matcher.ratio()


def _rescrape_and_chunk(url: str) -> list[Chunk]:
    """Re-runs scrape -> clean -> chunk for a single URL, mirroring
    ingest_website's per-page logic exactly."""
    items = scrape_faq_page(url)
    chunks: list[Chunk] = []
    for item in items:
        cleaned = clean_text(item.text)
        if not cleaned:
            continue
        if item.question:
            chunks.extend(chunk_qa_pair(question=item.question, answer=cleaned, source_url=item.source_url))
        else:
            chunks.extend(chunk_long_text(cleaned, item.source_url))
    return chunks


def _safe_commit(db: Session, url: str) -> bool:
    """
    Commits the session, tolerating the case where the Source row was
    deleted out from under us (e.g. someone hit "Delete" in the admin UI
    while this check was mid-flight).

    On StaleDataError / ObjectDeletedError: rolls back so the session is
    usable again for the *next* source in the pass, and returns False.
    On success: returns True.

    Without this, a single deleted-while-checking source poisons the
    SQLAlchemy session (PendingRollbackError) for every source checked
    after it in the same pass, and even logging the failure can itself
    throw (touching an expired attribute like source.url triggers a
    reload against a session that's no longer usable).
    """
    try:
        db.commit()
        return True
    except (StaleDataError, ObjectDeletedError) as e:
        db.rollback()
        logger.warning(f"[verify] {url} was deleted or modified concurrently, skipping: {e}")
        return False


def verify_source(source: Source, db: Session, similarity_threshold: float = SIMILARITY_THRESHOLD, correct_on_drift: bool = True) -> dict:
    """
    Checks one Source against its live webpage and corrects it in place if
    the stored content has drifted. Always attempts to commit an updated
    last_checked_at / last_check_result / last_similarity_score on the
    Source row, even on failure, so the admin UI reflects reality.

    correct_on_drift: when False, a detected drift is reported (result=
    "drifted") but NOT acted on - no re-scrape-driven re-ingest, no Groq/
    Gemini calls. Use this for lightweight callers (like the bulk-scrape
    resume sanity check) that just want a quick "did this change" signal
    and can't afford to block on a potentially several-minutes-long
    correction cycle (topic extraction + re-embedding, both of which retry
    for a while if the key pools are still rate-limited). The background
    self-healing verification agent is the right place for actual
    corrections - it runs independently and isn't blocking anything.

    Returns a small result dict for the caller (agent loop) to aggregate.
    """
    # Capture the URL up front, before any commit/rollback can expire this
    # attribute — touching source.url *after* a failed commit is what
    # previously triggered the cascading PendingRollbackError.
    url = source.url
    now = datetime.now(timezone.utc)

    try:
        fresh_chunks = _rescrape_and_chunk(url)
    except Exception as e:
        logger.warning(f"[verify] Could not re-scrape {url}: {e}")
        source.last_checked_at = now
        source.last_check_result = "error"
        if not _safe_commit(db, url):
            return {"url": url, "result": "skipped", "detail": "source deleted during re-scrape"}
        return {"url": url, "result": "error", "detail": str(e)}

    stored_chunks = get_chunks_by_source(url)

    fresh_text = "\n".join(c.text for c in fresh_chunks)
    stored_text = "\n".join(c["text"] for c in stored_chunks)

    similarity = compute_similarity(stored_text, fresh_text)
    source.last_checked_at = now
    source.last_similarity_score = similarity

    if not fresh_chunks:
        # Page now has nothing scrapable (dead link, blocked, redirected to
        # an empty page, etc). Don't nuke existing good data over a
        # transient fetch problem — just flag it for a human to look at.
        source.last_check_result = "error"
        if not _safe_commit(db, url):
            return {"url": url, "result": "skipped", "detail": "source deleted before commit"}
        logger.warning(f"[verify] {url} returned no scrapable content on re-check")
        return {"url": url, "result": "error", "detail": "no content on re-scrape"}

    if similarity >= similarity_threshold:
        source.last_check_result = "match"
        if not _safe_commit(db, url):
            return {"url": url, "result": "skipped", "detail": "source deleted before commit"}
        return {"url": url, "result": "match", "similarity": similarity}

    # --- Drift detected ---
    if not correct_on_drift:
        source.last_check_result = "drifted"
        if not _safe_commit(db, url):
            return {"url": url, "result": "skipped", "detail": "source deleted before commit"}
        logger.info(
            f"[verify] {url} drifted (similarity={similarity:.2f} < {similarity_threshold}) - "
            f"correction skipped (correct_on_drift=False), will be caught by the next full "
            f"verification pass instead"
        )
        return {"url": url, "result": "drifted", "similarity": similarity}

    logger.info(
        f"[verify] {url} drifted (similarity={similarity:.2f} < {similarity_threshold}), re-ingesting"
    )
    try:
        # Capture the OLD chunk IDs up front and delete them only AFTER the
        # new chunks are successfully embedded and upserted. Deleting first
        # (the old behavior) meant an embedding/upsert failure partway
        # through left this source with zero chunks instead of
        # stale-but-present ones. Deleting by explicit old IDs afterward
        # (rather than delete_chunks_by_source(url) again) matters too: the
        # new chunks share the same source_url, so a source_url-filtered
        # delete run after the upsert would also wipe the vectors that were
        # just written.
        old_chunk_ids = [c["id"] for c in stored_chunks]

        topics_per_chunk = link_topics_from_texts_batch(
            [c.text for c in fresh_chunks],
            [c.source_url for c in fresh_chunks],
        )
        embeddings = embed_texts([c.text for c in fresh_chunks])
        upsert_chunks(fresh_chunks, embeddings, topics_per_chunk)

        deleted = delete_chunks_by_ids(old_chunk_ids)

        source.chunks_created = len(fresh_chunks)
        source.status = "success"
        source.last_check_result = "corrected"
        source.last_corrected_at = now
        source.correction_count = (source.correction_count or 0) + 1

        if not _safe_commit(db, url):
            return {
                "url": url,
                "result": "skipped",
                "detail": "source deleted during correction (new vectors were still written)",
            }

        logger.info(
            f"[verify] Corrected {url}: removed {deleted} stale vectors, "
            f"stored {len(fresh_chunks)} fresh chunks"
        )
        return {
            "url": url,
            "result": "corrected",
            "similarity": similarity,
            "old_chunks_removed": deleted,
            "new_chunks_stored": len(fresh_chunks),
        }
    except Exception as e:
        logger.error(f"[verify] Failed to correct {url}: {e}")
        source.last_check_result = "error"
        if not _safe_commit(db, url):
            return {"url": url, "result": "skipped", "detail": "source deleted during failed correction"}
        return {"url": url, "result": "error", "detail": str(e)}