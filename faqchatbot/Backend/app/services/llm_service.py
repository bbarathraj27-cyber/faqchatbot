import json
import logging
import time
from app.config import settings
from app.prompts.faq_prompt import build_prompt
from app.services.groq_key_pool import chat_completion as _groq_chat_completion, CHAT_RETRY_BUDGET_SECONDS
from app.services.gemini_llm_service import gemini_chat_completion
from app.services.cache_service import get_cached_correction, set_cached_correction

logger = logging.getLogger(__name__)


def chat_completion(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    deadline: float | None = None,
    json_mode: bool = False,
) -> tuple[str, str]:
    try:
        content = _groq_chat_completion(messages, model=model, temperature=temperature, deadline=deadline)
        return content, "groq"
    except Exception as e:
        if not settings.LLM_FALLBACK_TO_GEMINI:
            raise
        logger.warning(f"Groq failed on all keys ({e}), falling back to Gemini ({settings.GEMINI_LLM_MODEL})")
        content = gemini_chat_completion(messages, temperature=temperature, deadline=deadline, json_mode=json_mode)
        return content, "gemini_fallback"

MAX_QUESTION_CHARS = 1000


def _cap_question(question: str) -> str:
    return question[:MAX_QUESTION_CHARS]


class AnswerGenerationError(Exception):
    """Raised when the LLM call itself fails (network/API error)."""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    try:
        messages = build_prompt(question, context_chunks)
        content, provider = chat_completion(messages, model=settings.LLM_MODEL, temperature=0.2)
        if provider != "groq":
            logger.info(f"generate_answer served by provider={provider}")
        return content
    except Exception as e:
        logger.error(f"Could not generate answer from LLM: {e}")
        raise AnswerGenerationError(str(e)) from e


def correct_question_spelling(question: str, deadline: float | None = None) -> str:
    try:
        question = _cap_question(question)
        content, provider = chat_completion(
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
        if provider != "groq":
            logger.info(f"correct_question_spelling served by provider={provider}")
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
    raw_question = question

    cached = get_cached_correction(raw_question)
    if cached is not None:
        return cached.get("corrected") or raw_question, cached.get("topics") or []

    deadline = time.time() + CHAT_RETRY_BUDGET_SECONDS

    try:
        capped_question = _cap_question(question)
        content, provider = chat_completion(
            [
                {"role": "system", "content": _CORRECT_AND_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": capped_question},
            ],
            model=settings.LLM_MODEL,
            temperature=0.0,
            deadline=deadline,
            json_mode=True,
        )
        if provider != "groq":
            logger.info(f"correct_and_extract_topics served by provider={provider}")
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