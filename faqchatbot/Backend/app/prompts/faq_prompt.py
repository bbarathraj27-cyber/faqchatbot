import tiktoken

# Same encoding chunking_service.py uses, so token counts here are
# consistent with how chunks were sized during ingestion.
_encoding = tiktoken.get_encoding("cl100k_base")

# Tokenization optimization: cap how many context tokens we send to the
# LLM per answer, instead of blindly concatenating every retrieved chunk.
# _retrieve() in question_api.py can return up to 5 vector matches + 3
# graph matches = up to 8 chunks (~450 tokens max each), which could push
# 3,600+ tokens of context into every single answer-generation call with
# no cap. This budget keeps cost/latency predictable and stops a future
# change to top_k from silently ballooning prompt size.
MAX_CONTEXT_TOKENS = 3000

SYSTEM_PROMPT = """You are a helpful FAQ assistant. Answer the user's question using ONLY the
context provided below. If the answer isn't in the context, say you don't know
rather than guessing. Keep answers clear and concise.

Context:
{context}
"""


def _build_context_text(context_chunks: list[dict]) -> str:
    """
    Adds chunks (already ranked by relevance from _retrieve) in order until
    the token budget is used up. If a chunk would only partially fit, it's
    truncated to fit exactly rather than dropped outright, so we never
    exceed MAX_CONTEXT_TOKENS but still use the available budget fully.
    """
    pieces: list[str] = []
    tokens_used = 0

    for c in context_chunks:
        piece = f"[Source: {c['source_url']}]\n{c['text']}"
        piece_tokens = _encoding.encode(piece)
        remaining = MAX_CONTEXT_TOKENS - tokens_used
        if remaining <= 0:
            break
        if len(piece_tokens) > remaining:
            piece = _encoding.decode(piece_tokens[:remaining])
            pieces.append(piece)
            tokens_used += remaining
            break
        pieces.append(piece)
        tokens_used += len(piece_tokens)

    return "\n\n".join(pieces)


def build_prompt(question: str, context_chunks: list[dict]) -> list[dict]:
    """
    Builds the messages array sent to the LLM, injecting retrieved
    context chunks (from Pinecone) into the system prompt so answers
    are grounded in your actual FAQ content. Context is capped at
    MAX_CONTEXT_TOKENS (see _build_context_text).
    """
    context_text = _build_context_text(context_chunks)

    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)},
        {"role": "user", "content": question},
    ]