"""
In-memory pause/cancel signaling for background scrape jobs.

Why this exists: relying only on the database (Neon Postgres) for
pause_requested / cancel_requested has two problems:

1. If Neon is unreachable (free-tier quota, cold start, etc.) the whole
   background job can crash on a DB read/write, taking the pause/cancel
   machinery down with it.
2. Even when Neon is reachable, a "Pause" click can sit for a long time if
   the scraper is in the middle of a long time.sleep() (e.g. a Gemini/Groq
   key-pool cooldown wait) - the DB was never checked mid-sleep, so Pause
   only takes effect after that sleep finally finishes.

This module keeps the pause/cancel flags in a plain in-process dict, keyed
by job_id, so:
- Checking/setting a flag is instant and never touches the network/DB.
- Long sleeps in the key pools can be broken into short increments that
  re-check the flag, so Pause can interrupt a stuck cooldown wait in under
  a second instead of minutes.

The database is still updated too (scrape_job_service persists status there
for the UI / job history / resumability), but it's treated as a "bonus"
write that must never be allowed to block or crash the actual pause/cancel
behavior. This module is the source of truth while a job is running.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class AbortRequested(Exception):
    """Raised from inside a long-running retry loop (embedding batches, LLM
    calls) the moment pause/cancel is detected mid-retry. A plain sleep
    interruption isn't enough for loops that retry immediately on failure
    (e.g. embedding_service's 5-minute batch retry budget) - without this,
    such a loop keeps retrying at full speed and never notices the pause
    request until it naturally finishes or exhausts its retry budget.
    Callers that catch this should treat it as "work was interrupted, not
    failed" - don't count it as a real error."""
    pass



_lock = threading.Lock()
_signals: dict[str, dict] = {}  # job_id -> {"pause": bool, "cancel": bool}

# Tracks which job_id the CURRENT THREAD is working on, so deeply nested
# calls (key-pool cooldown sleeps, buried inside ingest_website) can check
# the right job's signal without threading job_id through every function
# signature in the ingestion pipeline. Each background scrape job runs
# start-to-finish on a single worker thread, so thread-local storage is
# safe here.
_thread_local = threading.local()


def set_current_job(job_id: str | None) -> None:
    """Call at the start (and end, with None) of a background job's work on
    this thread, so nested code can find the right job without job_id being
    passed explicitly."""
    _thread_local.job_id = job_id


def get_current_job() -> str | None:
    return getattr(_thread_local, "job_id", None)


def clear_signals(job_id: str) -> None:
    """Reset both flags for a job. Call this unconditionally at the start of
    every run (fresh start AND resume) as a belt-and-suspenders guard
    against a stale flag surviving from a previous run of the same job_id."""
    with _lock:
        _signals[job_id] = {"pause": False, "cancel": False}


def request_pause(job_id: str) -> None:
    with _lock:
        _signals.setdefault(job_id, {"pause": False, "cancel": False})
        _signals[job_id]["pause"] = True


def request_cancel(job_id: str) -> None:
    with _lock:
        _signals.setdefault(job_id, {"pause": False, "cancel": False})
        _signals[job_id]["cancel"] = True


def get_signal(job_id: str) -> dict:
    with _lock:
        return dict(_signals.get(job_id, {"pause": False, "cancel": False}))


def is_abort_requested(job_id: str | None = None) -> bool:
    """True if pause or cancel has been requested for the given job (or the
    current thread's job, if job_id is omitted). Used by the key pools to
    decide whether to bail out of a cooldown sleep early."""
    jid = job_id if job_id is not None else get_current_job()
    if not jid:
        return False
    sig = get_signal(jid)
    return sig["pause"] or sig["cancel"]


def raise_if_abort_requested(job_id: str | None = None) -> None:
    """Like is_abort_requested, but raises AbortRequested instead of
    returning a bool. Use this INSIDE retry loops (not just before/after a
    sleep) so a pause/cancel breaks the loop immediately instead of letting
    it keep retrying until its own budget runs out."""
    if is_abort_requested(job_id):
        jid = job_id if job_id is not None else get_current_job()
        raise AbortRequested(f"pause/cancel requested for job {jid}")


def interruptible_sleep(seconds: float, job_id: str | None = None, poll_interval: float = 0.5) -> None:
    """Like time.sleep(seconds), but wakes up early the moment pause/cancel
    is requested for the relevant job. Safe to call with job_id=None from
    deep inside the key pools - it picks up whichever job the current
    thread is running via set_current_job(). If no job is associated with
    this thread (e.g. a normal foreground request, not a bulk scrape), it
    just behaves like a plain time.sleep()."""
    if seconds <= 0:
        return
    jid = job_id if job_id is not None else get_current_job()
    if not jid:
        time.sleep(seconds)
        return
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        if is_abort_requested(jid):
            logger.info(
                "interruptible_sleep aborted early for job %s (%.1fs of wait skipped)",
                jid, remaining,
            )
            return
        time.sleep(min(poll_interval, remaining))