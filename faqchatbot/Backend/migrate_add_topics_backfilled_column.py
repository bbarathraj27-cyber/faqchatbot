"""
One-time migration: adds a `topics_backfilled` column to the `sources`
table, used by backfill_topics.py to track which sources still need their
Neo4j topic graph populated (since bulk ingestion now runs with
extract_topics=False -- see app/api/admin_api.py -- to avoid burning
through Groq's daily token quota mid-crawl).

Usage:
    cd Backend
    python migrate_add_topics_backfilled_column.py
"""

from sqlalchemy import create_engine, text
from app.config import settings

STATEMENTS = [
    "ALTER TABLE sources ADD COLUMN IF NOT EXISTS topics_backfilled BOOLEAN DEFAULT FALSE",
]


def main():
    engine = create_engine(settings.DATABASE_URL)
    with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"Running: {stmt}")
            conn.execute(text(stmt))
    print("Migration complete.")


if __name__ == "__main__":
    main()