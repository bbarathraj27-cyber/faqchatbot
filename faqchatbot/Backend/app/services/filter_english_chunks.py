"""
Filter retrieved RAG chunks to English-only.

Drops any chunk whose stored 'text' is not detected as English, so the
chatbot only ever answers using English-language source content.
"""

from langdetect import detect, LangDetectException


def detect_lang(text: str) -> str:
    text = text.strip()
    if len(text) < 20:
        return "en"  # too short to reliably detect — assume fine, don't discard
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def filter_english_chunks(chunks: list[dict]) -> list[dict]:
    """
    Given a list of chunk dicts (each with a 'text' key, as returned by
    pinecone_service.query_similar_chunks / query_by_topics / etc.),
    return only the ones detected as English.
    """
    english_chunks = []
    for chunk in chunks:
        lang = detect_lang(chunk.get("text", ""))
        if lang == "en":
            english_chunks.append(chunk)
        else:
            import logging
            logging.getLogger(__name__).info(
                f"[SKIPPED - {lang}] {chunk.get('source_url', 'unknown')}"
            )
    return english_chunks