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

def chunk_qa_pair(question: str, answer: str, source_url: str) -> list[Chunk]:
    combined_text = f"Q: {question}\nA: {answer}"

    if _token_count(combined_text) > settings.CHUNK_SIZE_TOKENS * 1.5:
        return chunk_long_text(combined_text, source_url)

    return [Chunk(chunk_id=str(uuid.uuid4()), text=combined_text, source_url=source_url, question=question)]


def chunk_long_text(text: str, source_url: str) -> list[Chunk]:
    tokens = _encode(text)
    chunk_size = settings.CHUNK_SIZE_TOKENS
    overlap = settings.CHUNK_OVERLAP_TOKENS

    if chunk_size <= overlap:
        raise ValueError("CHUNK_SIZE_TOKENS must be greater than CHUNK_OVERLAP_TOKENS")

    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _decode(chunk_tokens)

        chunks.append(Chunk(chunk_id=str(uuid.uuid4()), text=chunk_text, source_url=source_url))

        if end == len(tokens):
            break
        start = end - overlap

    return chunks