import json
import logging
from app.config import settings
from app.prompts.faq_prompt import build_prompt
from app.services.groq_key_pool import chat_completion

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


def correct_question_spelling(question: str) -> str:
    """
    Uses the LLM to fix typos/spelling in the user's question before
    embedding + search, without changing its meaning.
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

    Falls back to (original question, []) on any failure so callers behave
    the same as before if this call fails - topics degrade gracefully since
    KAG retrieval already handles an empty topic list.
    """
    try:
        question = _cap_question(question)
        content = chat_completion(
            [
                {"role": "system", "content": _CORRECT_AND_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            model=settings.LLM_MODEL,
            temperature=0.0,
        )
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        corrected = str(parsed.get("corrected") or "").strip() or question
        topics = [str(t).strip().lower() for t in parsed.get("topics", []) if str(t).strip()][:4]
        return corrected, topics
    except Exception as e:
        logger.warning(f"Could not correct+extract in one call, falling back to two calls: {e}")
        corrected = correct_question_spelling(question)
        from app.services.neo4j_service import extract_topics
        topics = extract_topics(corrected)
        return corrected, topics