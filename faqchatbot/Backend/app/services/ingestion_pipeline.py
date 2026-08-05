import logging
import time
from app.services.scraper_service import scrape_multiple_pages, crawl_website
from app.services.cleaning_service import clean_text
from app.services.chunking_service import chunk_qa_pair, chunk_long_text, Chunk
from app.services.embedding_service import embed_texts
from app.services.pinecone_service import ensure_index_exists, upsert_chunks
from app.services.neo4j_service import link_topics_from_texts_batch
from app.services import control_signal

logger = logging.getLogger(__name__)


def ingest_website(
    urls: list[str],
    question_selector: str | None = None,
    answer_selector: str | None = None,
    crawl: bool = False,
    max_pages: int = 20,
    extract_topics: bool = True,
) -> dict:
    """
    Checked at every major step boundary (not a param anymore - uses
    control_signal.raise_if_abort_requested(), the SAME mechanism
    embedding_service already uses for its retry loops, and the SAME
    exception (control_signal.AbortRequested) that _run_bulk_scrape in
    admin_api.py already catches around this whole call). No separate
    should_abort callback needed - control_signal picks up the right job_id
    from the thread-local set by _run_bulk_scrape via set_current_job().

    Every checkpoint here sits BEFORE a write step (Neo4j topic links,
    Pinecone upsert), so if AbortRequested fires, nothing new has been
    written for this page yet at that point - the caller safely retries
    the whole page from scratch on resume, same as before.

    Each major step is timed and logged at INFO level (e.g. "Scrape took
    12.3s", "Embedding took 187.4s for 42 chunks") so a page that stalls
    for a long time shows exactly which step it stalled in, instead of
    just sitting silent - this is what makes a 5-minute Gemini retry
    budget (BATCH_RETRY_BUDGET_SECONDS in embedding_service.py) or a Groq
    cooldown stack visible in the logs instead of looking like a hang.
    """
    control_signal.raise_if_abort_requested()
    ensure_index_exists()

    control_signal.raise_if_abort_requested()
    t0 = time.time()
    if crawl:
        scraped_items = crawl_website(urls, question_selector, answer_selector, max_pages=max_pages)
    else:
        scraped_items = scrape_multiple_pages(urls, question_selector, answer_selector)
    logger.info(f"[timing] Scrape took {time.time() - t0:.1f}s for {len(scraped_items)} item(s)")

    control_signal.raise_if_abort_requested()
    t0 = time.time()
    all_chunks: list[Chunk] = []
    for item in scraped_items:
        cleaned = clean_text(item.text)
        if not cleaned:
            continue
        if item.question:
            all_chunks.extend(chunk_qa_pair(question=item.question, answer=cleaned, source_url=item.source_url))
        else:
            all_chunks.extend(chunk_long_text(cleaned, item.source_url))
    logger.info(f"[timing] Cleaning+chunking took {time.time() - t0:.1f}s, produced {len(all_chunks)} chunk(s)")

    if not all_chunks:
        return {
            "pages_scraped": len(scraped_items), "chunks_created": 0,
            "topics_extracted": 0, "total_words": 0,
            "status": "no content found - check your CSS selectors",
        }

    control_signal.raise_if_abort_requested()
    all_topics_seen = set()
    if extract_topics:
        # Batched + parallelized: groups chunks into batches of ~15 per LLM
        # call (instead of one call per chunk) and runs those batches
        # concurrently across however many Groq keys are configured.
        t0 = time.time()
        topics_per_chunk = link_topics_from_texts_batch(
            [chunk.text for chunk in all_chunks],
            [chunk.source_url for chunk in all_chunks],
        )
        logger.info(f"[timing] Topic extraction (Groq) took {time.time() - t0:.1f}s for {len(all_chunks)} chunk(s)")
        for topics in topics_per_chunk:
            all_topics_seen.update(topics)
    else:
        topics_per_chunk = [[] for _ in all_chunks]

    control_signal.raise_if_abort_requested()
    total_words = sum(len(chunk.text.split()) for chunk in all_chunks)
    texts = [chunk.text for chunk in all_chunks]
    t0 = time.time()
    embeddings = embed_texts(texts)  # already raises AbortRequested internally mid-retry
    logger.info(f"[timing] Embedding (Gemini) took {time.time() - t0:.1f}s for {len(texts)} chunk(s)")

    control_signal.raise_if_abort_requested()
    t0 = time.time()
    upsert_chunks(all_chunks, embeddings, topics_per_chunk)
    logger.info(f"[timing] Pinecone upsert took {time.time() - t0:.1f}s")

    return {
        "pages_scraped": len(scraped_items), "chunks_created": len(all_chunks),
        "topics_extracted": len(all_topics_seen), "total_words": total_words,
        "status": "success",
    }