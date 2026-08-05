from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, func

from app.database import Base


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, default="pending")  # pending, success, failed
    chunks_created = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

    # --- Verification agent bookkeeping ---
    # Last time this source was re-checked against the live webpage.
    last_checked_at = Column(DateTime, nullable=True)
    # Last time a mismatch was found and the stored chunks were corrected.
    last_corrected_at = Column(DateTime, nullable=True)
    # Outcome of the most recent check: "match", "corrected", "error", or
    # null if it has never been checked by the agent yet.
    last_check_result = Column(String, nullable=True)
    # Similarity ratio (0-1) between the live page content and what was
    # stored, from the most recent check.
    last_similarity_score = Column(Float, nullable=True)
    # How many times the agent has re-corrected this source over its life.
    correction_count = Column(Integer, default=0)

    # Set once backfill_topics.py has run Neo4j topic extraction for this
    # source. Bulk ingestion (admin_api.py) runs with extract_topics=False,
    # so every source starts out False here and gets picked up later at a
    # throttled rate that respects Groq's daily token quota.
    topics_backfilled = Column(Boolean, default=False)