"""
Shared tiktoken encoder for the whole app.
"""

import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


def encode(text: str) -> list[int]:
    return _encoding.encode(text)


def decode(tokens: list[int]) -> str:
    return _encoding.decode(tokens)


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncates text to at most max_tokens tokens. No-ops if already under budget."""
    if max_tokens <= 0:
        return ""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding.decode(tokens[:max_tokens])