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
from dataclasses import dataclass, field

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)

# Base cooldown after a key first gets rate-limited.
COOLDOWN_SECONDS = 60.0
# If the same key gets rate-limited again shortly after a previous cooldown
# (i.e. our 60s local cooldown clock didn't line up with Gemini's real quota
# reset window), back off harder instead of retrying it on the same losing
# cadence. Cooldown doubles per consecutive failure, capped here.
MAX_COOLDOWN_SECONDS = 300.0


@dataclass
class _KeyEntry:
    label: str          # short id for logging, e.g. "key-1" (never logs the real key)
    client: "genai.Client"
    cooldown_until: float = 0.0
    last_call_time: float = 0.0
    consecutive_failures: int = 0


MIN_SECONDS_BETWEEN_CALLS = 2.0


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

    def get_client(self):
        """
        Returns (client, label) for the next key that isn't currently in
        cooldown, round-robin style. If every key is in cooldown, returns
        the one that will free up soonest anyway (better to try and fail
        fast than to block ingestion entirely).
        """
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
            logger.warning(f"All Gemini keys in cooldown, waiting {wait:.1f}s on {soonest.label}")
            time.sleep(wait)
            return soonest.client, soonest.label

    def wait_for_rate_slot(self, label: str):
        with self._lock:
            entry = next((e for e in self._entries if e.label == label), None)
            if not entry:
                return
            now = time.time()
            wait = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - entry.last_call_time))
            if wait > 0:
                logger.debug("%s waiting %.2fs for rate slot", label, wait)
                time.sleep(wait)
            entry.last_call_time = time.time()

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