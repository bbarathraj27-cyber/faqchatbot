"""
Round-robin pool of Groq API keys for LLM calls (spelling+topic extraction,
batch topic extraction, answer generation).

Mirrors gemini_key_pool.py exactly: instead of hammering a single API key
(and its single free-tier quota, which is quite tight for Groq - as low as
6,000-8,000 tokens/minute on openai/gpt-oss-20b) until it starts returning
429s, we spread calls across several keys and put a key into a temporary
"cooldown" the moment it gets rate-limited, so the next call automatically
tries a different key instead of retrying the same exhausted one.

Also handles genuinely INVALID keys (401 invalid_api_key) separately from
rate limits: an invalid key is not a "try again later" situation, so instead
of cooling down and retrying it, we mark it permanently disabled for this
process and immediately move on to the next key in the pool. This means a
single bad key in GROQ_API_KEYS no longer breaks every request - only calls
would fail if EVERY key in the pool is bad/invalid.

Usage:
    from app.services.groq_key_pool import groq_key_pool
    client, key_label = groq_key_pool.get_client()
    ... use client (OpenAI-compatible client pointed at Groq's base_url) ...
    groq_key_pool.mark_rate_limited(key_label)   # only on a 429/rate-limit error
    groq_key_pool.mark_invalid(key_label)        # only on a 401/invalid_api_key error
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
    disabled: bool = False   # set True permanently once a key is confirmed invalid (401)


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
        return any(not e.disabled for e in self._entries)

    def key_count(self) -> int:
        return len(self._entries)

    def active_key_count(self) -> int:
        """Number of keys that are NOT permanently disabled (still usable)."""
        return sum(1 for e in self._entries if not e.disabled)

    def get_client(self):
        """
        Returns (client, label) for the next key that isn't currently in
        cooldown and isn't permanently disabled, round-robin style. If every
        active key is in cooldown, returns the one that will free up soonest
        anyway (better to try and fail fast than to block the whole request
        pipeline). Returns (None, None) if every key is permanently disabled.
        """
        with self._lock:
            active = [e for e in self._entries if not e.disabled]
            if not active:
                return None, None

            n = len(self._entries)
            now = time.time()

            for _ in range(n):
                entry = self._entries[self._cursor]
                self._cursor = (self._cursor + 1) % n
                if entry.disabled:
                    continue
                if entry.cooldown_until <= now:
                    wait = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - entry.last_call_time))
                    if wait > 0:
                        time.sleep(wait)
                    entry.last_call_time = time.time()
                    return entry.client, entry.label

            soonest = min(active, key=lambda e: e.cooldown_until)
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

    def mark_invalid(self, label: str):
        """
        Call when a key comes back with a genuine 401 invalid_api_key error.
        Unlike a rate limit, this will never resolve itself by waiting, so we
        permanently disable the key for the rest of this process instead of
        cooling it down and trying it again later.
        """
        with self._lock:
            for entry in self._entries:
                if entry.label == label:
                    entry.disabled = True
                    logger.warning(
                        f"{label} is invalid (401) - permanently disabling this key for "
                        f"this process. {self.active_key_count()}/{len(self._entries)} "
                        f"key(s) still active."
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


def is_invalid_key_error(e: Exception) -> bool:
    msg = str(e)
    return "401" in msg or "invalid_api_key" in msg.lower()


# How long we're willing to keep retrying a single chat completion (rotating
# across keys) before giving up. Mirrors BATCH_RETRY_BUDGET_SECONDS in
# embedding_service.py, but shorter - a user-facing question shouldn't hang
# for 5 minutes waiting on an LLM call the way a background ingest batch can.
CHAT_RETRY_BUDGET_SECONDS = 5.0


def chat_completion(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    deadline: float | None = None,
) -> str:
    """
    Shared entry point for every Groq chat completion in the app. Rotates
    across the key pool and retries on rate-limit errors within a time
    budget, so callers (llm_service.py, neo4j_service.py) don't each need
    their own retry/rotation logic.

    `deadline` is an optional absolute time.time() cutoff. Pass this when a
    caller makes several chat_completion() calls as part of one logical
    request (e.g. a combined call that falls back to two separate calls) so
    the whole chain shares a single CHAT_RETRY_BUDGET_SECONDS budget instead
    of each call getting its own independent 30s budget stacked on top of
    the others. If omitted, defaults to "now + CHAT_RETRY_BUDGET_SECONDS"
    exactly as before.

    Raises the last exception if every attempt within the retry budget
    fails - callers decide how to handle that (AnswerGenerationError,
    fall back to original text, etc).
    """
    if not groq_key_pool.available():
        raise RuntimeError("No valid Groq API keys configured (all keys missing or invalid)")

    end_time = deadline if deadline is not None else time.time() + CHAT_RETRY_BUDGET_SECONDS
    last_exc: Exception | None = None

    while time.time() < end_time:
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
            if is_invalid_key_error(e):
                groq_key_pool.mark_invalid(label)
                continue  # try the next key immediately, this one is permanently done
            elif is_rate_limit_error(e):
                groq_key_pool.mark_rate_limited(label)
                continue  # get_client() paces/rotates us to the next key
            else:
                # Some other error (bad request, network issue, etc.) -
                # retrying the same way won't help.
                raise

    raise last_exc or RuntimeError("Groq chat completion failed with no keys available")