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


def chunk_long_text(text: str, source_url: str) -> list[Chunk]:
    tokens = _encode(text)
    chunk_size = settings.CHUNK_SIZE_TOKENS
    overlap = settings.CHUNK_OVERLAP_TOKENS

    if chunk_size <= overlap:
        raise ValueError("CHUNK_SIZE_TOKENS must be greater than CHUNK_OVERLAP_TOKENS")

    chunks: list[Chunk] = []
    segments = _split_into_segments(_decode(tokens))
    if not segments:
        return []

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