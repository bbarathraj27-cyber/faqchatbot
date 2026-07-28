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
     and re-ingests: deletes the stale vectors, re-embeds the fresh chunks
     with Gemini, re-extracts topics with Groq, and upserts the corrected
     chunks back into Pinecone.

Similarity is intentionally cheap (difflib on word tokens) rather than an
LLM call, so checking 230+ sources doesn't itself burn the API budget the
agent is trying to protect. The LLM/embedding calls only happen for
sources that actually need correcting.
"""

import difflib
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.source_model import Source
from app.services.scraper_service import scrape_faq_page
from app.services.cleaning_service import clean_text
from app.services.chunking_service import chunk_qa_pair, chunk_long_text, Chunk
from app.services.embedding_service import embed_texts
from app.services.neo4j_service import link_topics_from_texts_batch
from app.services.pinecone_service import (
    get_chunks_by_source,
    delete_chunks_by_source,
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


def verify_source(source: Source, db: Session, similarity_threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """
    Checks one Source against its live webpage and corrects it in place if
    the stored content has drifted. Always commits an updated
    last_checked_at / last_check_result / last_similarity_score on the
    Source row, even on failure, so the admin UI reflects reality.

    Returns a small result dict for the caller (agent loop) to aggregate.
    """
    url = source.url
    now = datetime.now(timezone.utc)

    try:
        fresh_chunks = _rescrape_and_chunk(url)
    except Exception as e:
        logger.warning(f"[verify] Could not re-scrape {url}: {e}")
        source.last_checked_at = now
        source.last_check_result = "error"
        db.commit()
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
        db.commit()
        logger.warning(f"[verify] {url} returned no scrapable content on re-check")
        return {"url": url, "result": "error", "detail": "no content on re-scrape"}

    if similarity >= similarity_threshold:
        source.last_check_result = "match"
        db.commit()
        return {"url": url, "result": "match", "similarity": similarity}

    # --- Drift detected: correct it ---
    logger.info(
        f"[verify] {url} drifted (similarity={similarity:.2f} < {similarity_threshold}), re-ingesting"
    )
    try:
        deleted = delete_chunks_by_source(url)

        topics_per_chunk = link_topics_from_texts_batch(
            [c.text for c in fresh_chunks],
            [c.source_url for c in fresh_chunks],
        )
        embeddings = embed_texts([c.text for c in fresh_chunks])
        upsert_chunks(fresh_chunks, embeddings, topics_per_chunk)

        source.chunks_created = len(fresh_chunks)
        source.status = "success"
        source.last_check_result = "corrected"
        source.last_corrected_at = now
        source.correction_count = (source.correction_count or 0) + 1
        db.commit()

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
        db.commit()
        return {"url": url, "result": "error", "detail": str(e)}