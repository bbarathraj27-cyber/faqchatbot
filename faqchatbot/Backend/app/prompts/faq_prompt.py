from app.services.tokenizer_utils import encode as _encode, decode as _decode

# Tokenization optimization: cap how many context tokens we send to the
# LLM per answer, instead of blindly concatenating every retrieved chunk.
# _retrieve() in question_api.py can return up to 5 vector matches + 3
# graph matches = up to 8 chunks (~450 tokens max each), which could push
# 3,600+ tokens of context into every single answer-generation call with
# no cap. This budget keeps cost/latency predictable and stops a future
# change to top_k from silently ballooning prompt size.
MAX_CONTEXT_TOKENS = 1500


SYSTEM_PROMPT = """You are a friendly TensorFlow FAQ assistant. Use ONLY the context provided below to answer documentation questions clearly and concisely.

Instructions:
- If the user asks a TensorFlow or API question, answer using only the context.
- If the user says a greeting or starts a casual conversation (for example: "hi", "hello", "hey", "how are you"), respond politely and helpfully, then gently steer them back to asking a TensorFlow documentation question.
- If the user asks something unrelated or the context does not contain the answer, respond with exactly: "I don't know based on the available documentation. Please ask another TensorFlow question or provide more detail."
- Always respond in English, no matter what language the question or context is in.
- If the context contains non-English text, translate the relevant parts into English yourself before using it in your answer.

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
        piece_tokens = _encode(piece)
        remaining = MAX_CONTEXT_TOKENS - tokens_used
        if remaining <= 0:
            break
        if len(piece_tokens) > remaining:
            piece = _decode(piece_tokens[:remaining])
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