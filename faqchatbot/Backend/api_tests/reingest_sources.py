import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models.source_model import Source
from app.services.ingestion_pipeline import ingest_website
from app.services.pinecone_service import delete_chunks_by_source


def reingest_sources(urls: list[str], extract_topics: bool = False) -> list[dict]:
    results = []
    for url in urls:
        delete_chunks_by_source(url)
        result = ingest_website(urls=[url], extract_topics=extract_topics)
        results.append({"url": url, **result})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-ingest selected sources with the latest chunking logic")
    parser.add_argument("urls", nargs="+", help="One or more source URLs to re-ingest")
    parser.add_argument("--with-topics", action="store_true", help="Extract topics during re-ingest")
    args = parser.parse_args()

    results = reingest_sources(args.urls, extract_topics=args.with_topics)
    for result in results:
        print(result)
