"""
Backfills Neo4j topic extraction for sources that were bulk-ingested with
extract_topics=False (see app/api/admin_api.py's _run_bulk_scrape).

Why this exists: running topic extraction inline during bulk ingestion of
~18.8k URLs burns through all configured Groq keys' daily token quota
(200k TPD each) long before the crawl finishes, after which every remaining
page just cycles through rate-limit retries doing no real ingestion work.
Content is now ingested WITHOUT topics first (fast, reliable), and this
script fills in the Neo4j topic graph afterward at a sustainable,
resumable pace.

Safe to stop (Ctrl+C) and re-run at any time -- it always picks up sources
where topics_backfilled is still False, and marks each one True only after
its topics are successfully linked.

Usage:
    cd Backend
    python migrate_add_topics_backfilled_column.py   # once, if not already run
    python backfill_topics.py                        # run once
    python backfill_topics.py --loop                 # run continuously,
                                                       # sleeping between passes
"""

import argparse
import logging
import time

from app.database import SessionLocal
from app.models.source_model import Source
from app.services.pinecone_service import get_chunks_by_source
from app.services.neo4j_service import link_topics_from_texts_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Small batch of SOURCES per pass (each source can itself contain many
# chunks, and neo4j_service already groups chunks into TOPIC_BATCH_SIZE=15
# per Groq call) -- keeps a single pass well within a reasonable slice of
# the daily token budget so this can run alongside other Groq usage
# (chatbot answering live traffic) without starving it.
SOURCES_PER_PASS = 20

# Pause between passes when running with --loop, so this trickles along
# throughout the day instead of front-loading all Groq usage into a burst.
SLEEP_BETWEEN_PASSES_SECONDS = 60


def run_one_pass() -> int:
    """Backfills topics for up to SOURCES_PER_PASS sources. Returns how many
    were processed (0 means nothing left to do)."""
    db = SessionLocal()
    try:
        pending = (
            db.query(Source)
            .filter(Source.status == "success", Source.topics_backfilled.is_(False))
            .order_by(Source.id)
            .limit(SOURCES_PER_PASS)
            .all()
        )
        if not pending:
            return 0

        for source in pending:
            chunks = get_chunks_by_source(source.url)
            if not chunks:
                # Nothing in Pinecone for this URL (e.g. it failed silently
                # or was later deleted) -- mark it backfilled anyway so it
                # doesn't get retried forever.
                logger.info(f"No chunks found for {source.url}, marking backfilled (nothing to do)")
                source.topics_backfilled = True
                db.commit()
                continue

            texts = [c["text"] for c in chunks]
            urls = [source.url] * len(texts)
            try:
                link_topics_from_texts_batch(texts, urls)
                source.topics_backfilled = True
                db.commit()
                logger.info(f"Backfilled topics for {source.url} ({len(texts)} chunks)")
            except Exception as e:
                db.rollback()
                logger.warning(f"Failed to backfill topics for {source.url}: {e}")
                # Leave topics_backfilled False -- picked up again next pass.

        return len(pending)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run continuously instead of a single pass")
    args = parser.parse_args()

    if not args.loop:
        count = run_one_pass()
        print(f"Processed {count} source(s).")
        return

    print("Running continuously. Ctrl+C to stop.")
    while True:
        count = run_one_pass()
        if count == 0:
            print(f"Nothing left to backfill. Sleeping {SLEEP_BETWEEN_PASSES_SECONDS}s...")
        else:
            print(f"Processed {count} source(s). Sleeping {SLEEP_BETWEEN_PASSES_SECONDS}s...")
        time.sleep(SLEEP_BETWEEN_PASSES_SECONDS)


if __name__ == "__main__":
    main()