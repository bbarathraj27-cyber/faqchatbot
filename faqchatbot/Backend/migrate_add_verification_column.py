"""
One-time migration: adds the verification-agent columns to the existing
`sources` table.

Base.metadata.create_all() (used in app/main.py on startup) only creates
tables that don't exist yet — it will NOT add new columns to a table that's
already there. Since your `sources` table already has ~230 rows, run this
once against your database before starting the app with the verification
agent enabled.

Usage:
    cd Backend
    python migrate_add_verification_columns.py
"""

from sqlalchemy import create_engine, text
from app.config import settings

STATEMENTS = [
    "ALTER TABLE sources ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMP",
    "ALTER TABLE sources ADD COLUMN IF NOT EXISTS last_corrected_at TIMESTAMP",
    "ALTER TABLE sources ADD COLUMN IF NOT EXISTS last_check_result VARCHAR",
    "ALTER TABLE sources ADD COLUMN IF NOT EXISTS last_similarity_score FLOAT",
    "ALTER TABLE sources ADD COLUMN IF NOT EXISTS correction_count INTEGER DEFAULT 0",
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