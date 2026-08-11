import re
from dataclasses import dataclass, field
from typing import Optional
import uuid

from app.config import settings
from app.services.tokenizer_utils import encode as _encode, decode as _decode, count_tokens as _token_count


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    question: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-caps text to max_tokens, decoding back to a string."""
    tokens = _encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _decode(tokens[:max_tokens])

def _normalize_chunk_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    return text.replace("\n", " ")


def _split_into_segments(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sections = re.split(r"\n\s*\n", text)
    segments: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if re.match(r"^(#{1,6})\s+", section):
            segments.append(section)
        else:
            segments.append(section)
    return segments


def chunk_qa_pair(question: str, answer: str, source_url: str) -> list[Chunk]:
    combined_text = f"Q: {question}\nA: {answer}"

    if _token_count(combined_text) > settings.CHUNK_SIZE_TOKENS * 1.5:
        return chunk_long_text(combined_text, source_url)

    normalized = _normalize_chunk_text(combined_text)
    return [Chunk(chunk_id=str(uuid.uuid4()), text=normalized, source_url=source_url, question=question)]


def _split_oversized_segment(segment: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard-splits a single segment into token-sized pieces when it's already
    bigger than chunk_size on its own (e.g. a raw-text-fallback scrape with no
    blank-line paragraph breaks, so _split_into_segments returns it as one
    giant segment). Without this, that segment flows through the merge loop
    below untouched and becomes one oversized chunk - which is exactly how a
    17k-word page turned into a single ~95KB chunk that blew past Pinecone's
    40KB-per-vector metadata limit and silently failed to upsert."""
    tokens = _encode(segment)
    if len(tokens) <= chunk_size:
        return [segment]

    pieces = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        piece_tokens = tokens[start : start + chunk_size]
        if not piece_tokens:
            break
        pieces.append(_decode(piece_tokens))
        if start + chunk_size >= len(tokens):
            break
    return pieces


def chunk_long_text(text: str, source_url: str) -> list[Chunk]:
    tokens = _encode(text)
    chunk_size = settings.CHUNK_SIZE_TOKENS
    overlap = settings.CHUNK_OVERLAP_TOKENS

    if chunk_size <= overlap:
        raise ValueError("CHUNK_SIZE_TOKENS must be greater than CHUNK_OVERLAP_TOKENS")

    chunks: list[Chunk] = []
    raw_segments = _split_into_segments(_decode(tokens))
    if not raw_segments:
        return []

    segments: list[str] = []
    for seg in raw_segments:
        segments.extend(_split_oversized_segment(seg, chunk_size, overlap))

    current = ""
    for segment in segments:
        candidate = f"{current}\n\n{segment}".strip() if current else segment
        if _token_count(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(Chunk(chunk_id=str(uuid.uuid4()), text=_normalize_chunk_text(current), source_url=source_url))
            current = segment
        else:
            current = segment

    if current:
        chunks.append(Chunk(chunk_id=str(uuid.uuid4()), text=_normalize_chunk_text(current), source_url=source_url))

    return chunks