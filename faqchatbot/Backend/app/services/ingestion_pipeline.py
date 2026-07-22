import time
from app.services.scraper_service import scrape_multiple_pages, crawl_website
from app.services.cleaning_service import clean_text
from app.services.chunking_service import chunk_qa_pair, chunk_long_text, Chunk
from app.services.embedding_service import embed_texts
from app.services.pinecone_service import ensure_index_exists, upsert_chunks
from app.services.neo4j_service import link_topics_from_text

def ingest_website(
    urls: list[str],
    question_selector: str | None = None,
    answer_selector: str | None = None,
    crawl: bool = False,
    max_pages: int = 20,
    extract_topics: bool = True,
) -> dict:
    ensure_index_exists()
    if crawl:
        scraped_items = crawl_website(urls, question_selector, answer_selector, max_pages=max_pages)
    else:
        scraped_items = scrape_multiple_pages(urls, question_selector, answer_selector)
    all_chunks: list[Chunk] = []
    for item in scraped_items:
        cleaned = clean_text(item.text)
        if not cleaned:
            continue
        if item.question:
            all_chunks.extend(chunk_qa_pair(question=item.question, answer=cleaned, source_url=item.source_url))
        else:
            all_chunks.extend(chunk_long_text(cleaned, item.source_url))

    if not all_chunks:
        return {
            "pages_scraped": len(scraped_items), "chunks_created": 0,
            "topics_extracted": 0, "total_words": 0,
            "status": "no content found - check your CSS selectors",
        }
    topics_per_chunk: list[list[str]] = []
    all_topics_seen = set()
    if extract_topics:
        for chunk in all_chunks:
            topics = link_topics_from_text(chunk.text, chunk.source_url)
            topics_per_chunk.append(topics)
            all_topics_seen.update(topics)
    else:
        topics_per_chunk = [[] for _ in all_chunks]
    total_words = sum(len(chunk.text.split()) for chunk in all_chunks)
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_texts(texts)
    upsert_chunks(all_chunks, embeddings, topics_per_chunk)
    return {
        "pages_scraped": len(scraped_items), "chunks_created": len(all_chunks),
        "topics_extracted": len(all_topics_seen), "total_words": total_words,
        "status": "success",
    }