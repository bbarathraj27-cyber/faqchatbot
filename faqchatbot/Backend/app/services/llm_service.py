import json
import logging
import time
from app.config import settings
from app.prompts.faq_prompt import build_prompt
from app.services.groq_key_pool import chat_completion, CHAT_RETRY_BUDGET_SECONDS
from app.services.cache_service import get_cached_correction, set_cached_correction

logger = logging.getLogger(__name__)

# Tokenization optimization: cap how much of the raw user question we ever
# send to the LLM. Questions are normally short (a sentence or two), so
# this never affects real usage - it only protects against a pathological
# input (e.g. someone pasting a huge block of text into the question field)
# silently inflating every downstream call: correct_and_extract_topics,
# embed_text (via the corrected question), and generate_answer.
MAX_QUESTION_CHARS = 1000


def _cap_question(question: str) -> str:
    return question[:MAX_QUESTION_CHARS]


class AnswerGenerationError(Exception):
    """Raised when the LLM call itself fails (network/API error), as opposed
    to the LLM successfully responding with 'I don't know'. Callers use this
    to distinguish a genuine failure (should NOT be cached) from a real,
    grounded-or-not answer (safe to cache)."""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    try:
        messages = build_prompt(question, context_chunks)
        return chat_completion(messages, model=settings.LLM_MODEL, temperature=0.2)
    except Exception as e:
        logger.error(f"Could not generate answer from LLM: {e}")
        raise AnswerGenerationError(str(e)) from e


def correct_question_spelling(question: str, deadline: float | None = None) -> str:
    """
    Uses the LLM to fix typos/spelling in the user's question before
    embedding + search, without changing its meaning.

    `deadline` lets a caller (e.g. correct_and_extract_topics' fallback
    path) share a single retry budget across multiple chat_completion()
    calls instead of each call getting its own independent budget.
    """
    try:
        question = _cap_question(question)
        content = chat_completion(
            [
                {
                    "role": "system",
                    "content": "Fix any spelling or typing mistakes in the user's question. "
                                "Keep the meaning and technical terms (like TensorFlow, Keras, TFX) exactly as intended. "
                                "Return ONLY the corrected question, nothing else — no explanation, no quotes."
                },
                {"role": "user", "content": question},
            ],
            model=settings.LLM_MODEL,
            temperature=0.0,
            deadline=deadline,
        )
        corrected = content.strip()
        return corrected if corrected else question
    except Exception as e:
        logger.warning(f"Could not correct question spelling: {e}")
        return question


_CORRECT_AND_EXTRACT_SYSTEM_PROMPT = """You do two things to the user's question:
1. Fix any spelling/typing mistakes without changing its meaning. Keep technical
   terms (like TensorFlow, Keras, TFX) exactly as intended.
2. Extract 2 to 4 short topic keywords (1-3 words each) representing what the
   question is about.

Return ONLY raw JSON in this exact shape, nothing else - no explanation, no
markdown fences:
{"corrected": "...", "topics": ["...", "..."]}
"""


def correct_and_extract_topics(question: str) -> tuple[str, list[str]]:
    """
    Combines correct_question_spelling + extract_topics into a single Groq
    call. These were previously two separate sequential LLM round-trips on
    every question (cache hit or miss); this cuts it to one.

    Before making any Groq call at all, this checks a correction cache
    (keyed on the raw question) so that a repeated raw question - even one
    that never previously produced a cached answer, e.g. because retrieval
    came up empty - doesn't pay for another Groq round-trip just to
    re-derive the same corrected question + topics.

    Falls back to (original question, []) on any failure so callers behave
    the same as before if this call fails - topics degrade gracefully since
    KAG retrieval already handles an empty topic list.

    On the fallback path (combined call fails), correct_question_spelling()
    and extract_topics() share a single deadline instead of each getting
    its own independent CHAT_RETRY_BUDGET_SECONDS budget - otherwise the
    combined call + the two fallback calls can each retry for up to 30s,
    stacking to ~90s+ worst-case latency on one request.
    """
    raw_question = question

    cached = get_cached_correction(raw_question)
    if cached is not None:
        return cached.get("corrected") or raw_question, cached.get("topics") or []

    deadline = time.time() + CHAT_RETRY_BUDGET_SECONDS

    try:
        capped_question = _cap_question(question)
        content = chat_completion(
            [
                {"role": "system", "content": _CORRECT_AND_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": capped_question},
            ],
            model=settings.LLM_MODEL,
            temperature=0.0,
            deadline=deadline,
        )
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        corrected = str(parsed.get("corrected") or "").strip() or capped_question
        topics = [str(t).strip().lower() for t in parsed.get("topics", []) if str(t).strip()][:4]
        set_cached_correction(raw_question, corrected, topics)
        return corrected, topics
    except Exception as e:
        logger.warning(f"Could not correct+extract in one call, falling back to two calls: {e}")
        corrected = correct_question_spelling(question, deadline=deadline)
        from app.services.neo4j_service import extract_topics
        topics = extract_topics(corrected, deadline=deadline)
        set_cached_correction(raw_question, corrected, topics)
        return corrected, topics