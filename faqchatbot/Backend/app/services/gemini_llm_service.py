"""
Gemini chat-completion fallback, used ONLY when Groq (openai/gpt-oss-20b)
fails on every key in its pool. Normal traffic never touches this file.
"""

import logging
import time

from google.genai import types

from app.config import settings
from app.services.gemini_llm_key_pool import gemini_llm_key_pool

logger = logging.getLogger(__name__)

GEMINI_CHAT_RETRY_BUDGET_SECONDS = 20.0


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[str]]:
    system_instruction = None
    user_parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_instruction = content if system_instruction is None else f"{system_instruction}\n\n{content}"
        else:
            user_parts.append(content)
    return system_instruction, user_parts


def gemini_chat_completion(
    messages: list[dict],
    temperature: float = 0.0,
    deadline: float | None = None,
    json_mode: bool = False,
) -> str:
    if not gemini_llm_key_pool.available():
        raise RuntimeError("No Gemini LLM fallback keys configured")

    system_instruction, user_parts = _to_gemini_contents(messages)
    contents = "\n\n".join(p for p in user_parts if p)

    config_kwargs = {"temperature": temperature}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    min_end_time = time.time() + GEMINI_CHAT_RETRY_BUDGET_SECONDS
    end_time = max(deadline, min_end_time) if deadline is not None else min_end_time
    last_exc: Exception | None = None

    while time.time() < end_time:
        client, label = gemini_llm_key_pool.get_client()
        if client is None:
            break

        gemini_llm_key_pool.wait_for_rate_slot(label)
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_LLM_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            gemini_llm_key_pool.mark_success(label)
            return (response.text or "").strip()
        except Exception as e:
            last_exc = e
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                gemini_llm_key_pool.mark_rate_limited(label)
                continue
            else:
                raise

    raise last_exc or RuntimeError("Gemini LLM fallback failed with no keys available")