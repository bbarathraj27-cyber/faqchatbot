"""
One-time migration: adds the render_js column to the existing
`scrape_jobs` table.

Base.metadata.create_all() (used in app/main.py on startup) only creates
tables that don't exist yet — it will NOT add new columns to a table that's
already there. Run this once against your database before starting the
app with JS-rendering support enabled for bulk/range scrape jobs.

Usage:
    cd Backend
    python migrate_add_render_js_column.py
"""

from sqlalchemy import create_engine, text
from app.config import settings

STATEMENTS = [
    "ALTER TABLE scrape_jobs ADD COLUMN IF NOT EXISTS render_js BOOLEAN DEFAULT FALSE",
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