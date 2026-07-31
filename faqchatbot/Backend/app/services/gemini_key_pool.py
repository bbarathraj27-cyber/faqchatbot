"""
Round-robin pool of Gemini API keys for the embedding service.

Mirrors the same idea as the Groq LLMKeyPool: instead of hammering a single
API key (and its single free-tier quota) until it starts returning
RESOURCE_EXHAUSTED / 429, we spread calls across several keys and put a key
into a temporary "cooldown" the moment it gets rate-limited, so the next
request automatically tries a different key instead of retrying the same
exhausted one.

Usage:
    from app.services.gemini_key_pool import gemini_key_pool
    client, key_label = gemini_key_pool.get_client()
    ... use client ...
    gemini_key_pool.mark_rate_limited(key_label)   # only on a 429/RESOURCE_EXHAUSTED
    gemini_key_pool.mark_success(key_label)         # on a successful call
"""

import logging
import threading
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

# Base cooldown after a key first gets rate-limited.
COOLDOWN_SECONDS = 60.0
# If the same key gets rate-limited again shortly after a previous cooldown
# (i.e. our 60s local cooldown clock didn't line up with Gemini's real quota
# reset window), back off harder instead of retrying it on the same losing
# cadence. Cooldown doubles per consecutive failure, capped here.
MAX_COOLDOWN_SECONDS = 300.0

MIN_SECONDS_BETWEEN_CALLS = 10.0


@dataclass
class _KeyEntry:
    label: str          # short id for logging, e.g. "key-1" (never logs the real key)
    client: "genai.Client"
    cooldown_until: float = 0.0
    last_call_time: float = 0.0
    consecutive_failures: int = 0


class GeminiKeyPool:
    def __init__(self, api_keys: list[str]):
        self._lock = threading.Lock()
        self._entries: list[_KeyEntry] = []
        self._cursor = 0

        for i, key in enumerate(api_keys):
            key = key.strip()
            if not key:
                continue
            try:
                client = genai.Client(api_key=key)
                self._entries.append(_KeyEntry(label=f"key-{i+1}", client=client))
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client for key-{i+1}: {e}")

        if self._entries:
            logger.info(f"Gemini key pool initialized with {len(self._entries)} key(s)")
        else:
            logger.warning("Gemini key pool has no usable keys — embeddings will be unavailable")

    def available(self) -> bool:
        return len(self._entries) > 0

    def key_count(self) -> int:
        return len(self._entries)

    def get_client(self):
        """
        Returns (client, label) for the next key that isn't currently in
        cooldown, round-robin style. If every key is in cooldown, waits for
        whichever one frees up soonest — but the wait happens OUTSIDE the
        lock so it doesn't serialize every other thread trying to grab a
        client at the same time.
        """
        wait = 0.0
        soonest_client = None
        soonest_label = None

        with self._lock:
            if not self._entries:
                return None, None

            n = len(self._entries)
            now = time.time()

            # First pass: find a key that's not in cooldown.
            for _ in range(n):
                entry = self._entries[self._cursor]
                self._cursor = (self._cursor + 1) % n
                if entry.cooldown_until <= now:
                    return entry.client, entry.label

            # All keys are cooling down — pick whichever recovers soonest.
            soonest = min(self._entries, key=lambda e: e.cooldown_until)
            wait = max(0.0, soonest.cooldown_until - now)
            soonest_client, soonest_label = soonest.client, soonest.label

        # Sleep OUTSIDE the lock so other threads aren't blocked behind us.
        if wait > 0:
            logger.warning(f"All Gemini keys in cooldown, waiting {wait:.1f}s on {soonest_label}")
            time.sleep(wait)
        return soonest_client, soonest_label

    def wait_for_rate_slot(self, label: str):
        """
        Simple per-key pacing so we don't fire calls on the same key faster
        than MIN_SECONDS_BETWEEN_CALLS. The wait itself happens outside the
        lock so concurrent workers on DIFFERENT keys never block each other,
        and workers on the SAME key only wait as long as actually needed.
        """
        wait = 0.0
        with self._lock:
            entry = next((e for e in self._entries if e.label == label), None)
            if not entry:
                return
            now = time.time()
            wait = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - entry.last_call_time))
            # Reserve the slot immediately so a second thread checking right
            # behind us sees the updated last_call_time, not the stale one.
            entry.last_call_time = now + wait

        if wait > 0:
            logger.debug("%s waiting %.2fs for rate slot", label, wait)
            time.sleep(wait)

    def mark_rate_limited(self, label: str):
        with self._lock:
            for entry in self._entries:
                if entry.label == label:
                    entry.consecutive_failures += 1
                    # Exponential backoff: 60s, 120s, 240s, capped at
                    # MAX_COOLDOWN_SECONDS. Resets to the base once the key
                    # succeeds again (see mark_success).
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
        """Call after a successful embed call so backoff doesn't stay escalated forever."""
        with self._lock:
            for entry in self._entries:
                if entry.label == label:
                    entry.consecutive_failures = 0
                    return


def _resolve_keys() -> list[str]:
    if settings.GEMINI_API_KEYS.strip():
        return [k for k in settings.GEMINI_API_KEYS.split(",") if k.strip()]
    if settings.GEMINI_API_KEY.strip():
        return [settings.GEMINI_API_KEY]
    return []


gemini_key_pool = GeminiKeyPool(_resolve_keys())

# ---------------------------------------------------------------------------
# Chat completion (used as a FALLBACK when the Groq key pool is exhausted —
# see groq_key_pool.chat_completion). Reuses the same key pool / cooldown
# machinery as embed_text, just pointed at generate_content instead of
# embed_content.
# ---------------------------------------------------------------------------

CHAT_FALLBACK_RETRY_BUDGET_SECONDS = 5.0


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _messages_to_gemini(messages: list[dict]) -> tuple[str | None, list["types.Content"]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for m in messages:
        role = m.get("role")
        text = m.get("content", "")
        if role == "system":
            system_parts.append(text)
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=[types.Part(text=text)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    deadline: float | None = None,
) -> str:
    if not gemini_key_pool.available():
        raise RuntimeError("No valid Gemini API keys configured (all keys missing)")

    model = model or settings.GEMINI_LLM_MODEL
    system_instruction, contents = _messages_to_gemini(messages)

    end_time = deadline if deadline is not None else time.time() + CHAT_FALLBACK_RETRY_BUDGET_SECONDS
    last_exc: Exception | None = None

    while time.time() < end_time:
        client, label = gemini_key_pool.get_client()
        if client is None:
            break
        gemini_key_pool.wait_for_rate_slot(label)
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
            )
            gemini_key_pool.mark_success(label)
            return response.text
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e):
                gemini_key_pool.mark_rate_limited(label)
                continue
            else:
                raise

    raise last_exc or RuntimeError("Gemini chat completion failed with no keys available")