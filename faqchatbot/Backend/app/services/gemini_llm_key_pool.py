"""
Round-robin pool of Gemini API keys, dedicated to the LLM chat FALLBACK
path only (used when Groq is fully exhausted). This is intentionally a
separate pool from gemini_key_pool.py (which is dedicated to embeddings),
so a heavy ingestion run hammering the embedding pool can never starve out
a live chat fallback that happens to need a key at the same moment.

Key resolution order:
    1. settings.GEMINI_LLM_API_KEYS   (dedicated fallback keys, if you set them)
    2. settings.GEMINI_API_KEYS       (reuse the embedding pool's keys)
    3. settings.GEMINI_API_KEY        (single-key fallback)
"""

import logging
import threading
import time
from dataclasses import dataclass

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 60.0
MAX_COOLDOWN_SECONDS = 300.0
MIN_SECONDS_BETWEEN_CALLS = 2.0


@dataclass
class _KeyEntry:
    label: str
    client: "genai.Client"
    cooldown_until: float = 0.0
    last_call_time: float = 0.0
    consecutive_failures: int = 0


class GeminiLLMKeyPool:
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
                self._entries.append(_KeyEntry(label=f"llm-key-{i+1}", client=client))
            except Exception as e:
                logger.warning(f"Could not initialize Gemini LLM client for llm-key-{i+1}: {e}")

        if self._entries:
            logger.info(f"Gemini LLM (fallback) key pool initialized with {len(self._entries)} key(s)")
        else:
            logger.warning("Gemini LLM (fallback) key pool has no usable keys - Gemini fallback will be unavailable")

    def available(self) -> bool:
        return len(self._entries) > 0

    def key_count(self) -> int:
        return len(self._entries)

    def get_client(self):
        wait = 0.0
        soonest_client = None
        soonest_label = None

        with self._lock:
            if not self._entries:
                return None, None

            n = len(self._entries)
            now = time.time()

            for _ in range(n):
                entry = self._entries[self._cursor]
                self._cursor = (self._cursor + 1) % n
                if entry.cooldown_until <= now:
                    return entry.client, entry.label

            soonest = min(self._entries, key=lambda e: e.cooldown_until)
            wait = max(0.0, soonest.cooldown_until - now)
            soonest_client, soonest_label = soonest.client, soonest.label

        if wait > 0:
            logger.warning(f"All Gemini LLM keys in cooldown, waiting {wait:.1f}s on {soonest_label}")
            time.sleep(wait)
        return soonest_client, soonest_label

    def wait_for_rate_slot(self, label: str):
        wait = 0.0
        with self._lock:
            entry = next((e for e in self._entries if e.label == label), None)
            if not entry:
                return
            now = time.time()
            wait = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - entry.last_call_time))
            entry.last_call_time = now + wait

        if wait > 0:
            logger.debug("%s waiting %.2fs for rate slot", label, wait)
            time.sleep(wait)

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
        with self._lock:
            for entry in self._entries:
                if entry.label == label:
                    entry.consecutive_failures = 0
                    return


def _resolve_keys() -> list[str]:
    if settings.GEMINI_LLM_API_KEYS.strip():
        return [k for k in settings.GEMINI_LLM_API_KEYS.split(",") if k.strip()]
    if settings.GEMINI_API_KEYS.strip():
        return [k for k in settings.GEMINI_API_KEYS.split(",") if k.strip()]
    if settings.GEMINI_API_KEY.strip():
        return [settings.GEMINI_API_KEY]
    return []


gemini_llm_key_pool = GeminiLLMKeyPool(_resolve_keys())