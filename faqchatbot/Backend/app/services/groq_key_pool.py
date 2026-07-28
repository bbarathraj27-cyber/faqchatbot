"""
Round-robin pool of Groq API keys for LLM calls (spelling+topic extraction,
batch topic extraction, answer generation).

Mirrors gemini_key_pool.py exactly: instead of hammering a single API key
(and its single free-tier quota, which is quite tight for Groq - as low as
6,000-8,000 tokens/minute on openai/gpt-oss-20b) until it starts returning
429s, we spread calls across several keys and put a key into a temporary
"cooldown" the moment it gets rate-limited, so the next call automatically
tries a different key instead of retrying the same exhausted one.

Usage:
    from app.services.groq_key_pool import groq_key_pool
    client, key_label = groq_key_pool.get_client()
    ... use client (OpenAI-compatible client pointed at Groq's base_url) ...
    groq_key_pool.mark_rate_limited(key_label)   # only on a 429/rate-limit error
    groq_key_pool.mark_success(key_label)         # on a successful call
"""

import logging
import threading
import time
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Base cooldown after a key first gets rate-limited.
COOLDOWN_SECONDS = 10.0
# If the same key gets rate-limited again shortly after a previous cooldown,
# back off harder instead of retrying it on the same losing cadence.
# Cooldown doubles per consecutive failure, capped here.
MAX_COOLDOWN_SECONDS = 20.0

# Minimum gap between two calls made with the SAME key, so a burst of
# requests that all happen to land on one key still gets paced instead of
# firing back-to-back.
MIN_SECONDS_BETWEEN_CALLS = 1.0


@dataclass
class _KeyEntry:
    label: str          # short id for logging, e.g. "groq-key-1" (never logs the real key)
    client: "OpenAI"
    cooldown_until: float = 0.0
    last_call_time: float = 0.0
    consecutive_failures: int = 0


class GroqKeyPool:
    def __init__(self, api_keys: list[str]):
        self._lock = threading.Lock()
        self._entries: list[_KeyEntry] = []
        self._cursor = 0

        for i, key in enumerate(api_keys):
            key = key.strip()
            if not key:
                continue
            try:
                client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
                self._entries.append(_KeyEntry(label=f"groq-key-{i+1}", client=client))
            except Exception as e:
                logger.warning(f"Could not initialize Groq client for groq-key-{i+1}: {e}")

        if self._entries:
            logger.info(f"Groq key pool initialized with {len(self._entries)} key(s)")
        else:
            logger.warning("Groq key pool has no usable keys - LLM calls will fail")

    def available(self) -> bool:
        return len(self._entries) > 0

    def key_count(self) -> int:
        return len(self._entries)

    def get_client(self):
        """
        Returns (client, label) for the next key that isn't currently in
        cooldown, round-robin style. If every key is in cooldown, returns
        the one that will free up soonest anyway (better to try and fail
        fast than to block the whole request pipeline).
        """
        with self._lock:
            if not self._entries:
                return None, None

            n = len(self._entries)
            now = time.time()

            for _ in range(n):
                entry = self._entries[self._cursor]
                self._cursor = (self._cursor + 1) % n
                if entry.cooldown_until <= now:
                    wait = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - entry.last_call_time))
                    if wait > 0:
                        time.sleep(wait)
                    entry.last_call_time = time.time()
                    return entry.client, entry.label

            soonest = min(self._entries, key=lambda e: e.cooldown_until)
            wait = max(0.0, soonest.cooldown_until - now)
            logger.warning(f"All Groq keys in cooldown, waiting {wait:.1f}s on {soonest.label}")
            time.sleep(wait)
            soonest.last_call_time = time.time()
            return soonest.client, soonest.label

    def mark_rate_limited(self, label: str):
        with self._lock:
            for entry in self._entries:
                if entry.label == label:
                    entry.consecutive_failures += 1
                    cooldown = min(
                        COOLDOWN_SECONDS * (2 ** (entry.consecutive_failures - 1)),
                        MAX_COOLDOWN_SECONDS,
                    )
                    entry.cooldown_until = time.time() + cooldown
                    logger.warning(
                        f"{label} rate-limited (consecutive failure #{entry.consecutive_failures}), "
                        f"cooling down for {cooldown:.0f}s"
                    )
                    return

    def mark_success(self, label: str):
        """Call after a successful LLM call so backoff doesn't stay escalated forever."""
        with self._lock:
            for entry in self._entries:
                if entry.label == label:
                    entry.consecutive_failures = 0
                    return


def _resolve_keys() -> list[str]:
    if settings.GROQ_API_KEYS.strip():
        return [k for k in settings.GROQ_API_KEYS.split(",") if k.strip()]
    if settings.GROQ_API_KEY.strip():
        return [settings.GROQ_API_KEY]
    return []


groq_key_pool = GroqKeyPool(_resolve_keys())


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "rate_limit" in msg.lower() or "RESOURCE_EXHAUSTED" in msg


# How long we're willing to keep retrying a single chat completion (rotating
# across keys) before giving up. Mirrors BATCH_RETRY_BUDGET_SECONDS in
# embedding_service.py, but shorter - a user-facing question shouldn't hang
# for 5 minutes waiting on an LLM call the way a background ingest batch can.
CHAT_RETRY_BUDGET_SECONDS = 30.0


def chat_completion(messages: list[dict], model: str, temperature: float = 0.0) -> str:
    """
    Shared entry point for every Groq chat completion in the app. Rotates
    across the key pool and retries on rate-limit errors within a time
    budget, so callers (llm_service.py, neo4j_service.py) don't each need
    their own retry/rotation logic.

    Raises the last exception if every attempt within the retry budget
    fails - callers decide how to handle that (AnswerGenerationError,
    fall back to original text, etc).
    """
    if not groq_key_pool.available():
        raise RuntimeError("No Groq API keys configured")

    start_time = time.time()
    last_exc: Exception | None = None

    while time.time() - start_time < CHAT_RETRY_BUDGET_SECONDS:
        client, label = groq_key_pool.get_client()
        if client is None:
            break
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            groq_key_pool.mark_success(label)
            return response.choices[0].message.content
        except Exception as e:
            last_exc = e
            if is_rate_limit_error(e):
                groq_key_pool.mark_rate_limited(label)
                continue  # get_client() paces/rotates us to the next key
            else:
                # Non-rate-limit error - retrying the same way won't help.
                raise

    raise last_exc or RuntimeError("Groq chat completion failed with no keys available")