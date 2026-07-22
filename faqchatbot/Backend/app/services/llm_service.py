import logging
from openai import OpenAI
from app.config import settings
from app.prompts.faq_prompt import build_prompt

logger = logging.getLogger(__name__)
_client = OpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


class AnswerGenerationError(Exception):
    """Raised when the LLM call itself fails (network/API error), as opposed
    to the LLM successfully responding with 'I don't know'. Callers use this
    to distinguish a genuine failure (should NOT be cached) from a real,
    grounded-or-not answer (safe to cache)."""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    try:
        messages = build_prompt(question, context_chunks)
        response = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Could not generate answer from LLM: {e}")
        # Previously this returned a fake "answer" string, which meant callers
        # (question_api.ask_question) couldn't tell a real answer apart from a
        # failure and ended up caching the failure for 24h. Raise instead so
        # the caller can handle/skip-caching this case explicitly.
        raise AnswerGenerationError(str(e)) from e


def correct_question_spelling(question: str) -> str:
    """
    Uses the LLM to fix typos/spelling in the user's question before
    embedding + search, without changing its meaning.
    """
    try:
        response = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Fix any spelling or typing mistakes in the user's question. "
                                "Keep the meaning and technical terms (like TensorFlow, Keras, TFX) exactly as intended. "
                                "Return ONLY the corrected question, nothing else — no explanation, no quotes."
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        corrected = response.choices[0].message.content.strip()
        return corrected if corrected else question
    except Exception as e:
        logger.warning(f"Could not correct question spelling: {e}")
        return question