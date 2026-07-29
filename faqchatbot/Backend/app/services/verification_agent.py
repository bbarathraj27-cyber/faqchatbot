"""
Background verification agent.

Continuously loops over every row in the `sources` table (the list of
ingested webpages you see in the admin tab), re-checks each one against
its live page via verification_service.verify_source, and corrects any
that have drifted. After a full pass over every source, it sleeps for a
cooldown period (default 2 days) before starting the next pass. This runs
forever in a single background thread for the lifetime of the process.

Usage (wired up in app/main.py on startup):

    from app.services.verification_agent import verification_agent
    verification_agent.start()

Status is exposed in-memory (mirrors job_tracker.py's pattern) and can be
read from the admin API without touching the DB:

    from app.services.verification_agent import verification_agent
    verification_agent.get_status()

A manual "run now" is also supported, e.g. from an admin button, without
disturbing the 2-day schedule of the background loop — see run_once_now().
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from app.database import SessionLocal
from app.models.source_model import Source
from app.services.verification_service import verify_source

logger = logging.getLogger(__name__)

# 2 days, in seconds, between the end of one full pass and the start of the
# next. Kept as a module constant (not settings) since this is an internal
# operational knob, not something that needs to vary per environment — but
# it's trivially overridable below if that changes.
COOLDOWN_SECONDS = 2 * 24 * 60 * 60

# Small pause between sources within a single pass. The Gemini/Groq key
# pools already rate-limit + cool down their own calls, so this isn't load
# protection — it's just to keep a 230+ source pass from being one
# unbroken CPU/network burst and to make the "currently checking" status
# meaningful to watch in the UI.
PER_SOURCE_PAUSE_SECONDS = 1.0


class VerificationAgent:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._wake_event = threading.Event()  # lets "run now" skip the cooldown wait
        self._status = {
            "running_pass": False,
            "enabled": False,
            "last_pass_started_at": None,
            "last_pass_finished_at": None,
            "next_pass_at": None,
            "current_url": None,
            "sources_checked": 0,
            "sources_total": 0,
            "corrected_this_pass": 0,
            "errors_this_pass": 0,
            "last_pass_summary": None,
        }

    # ---------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------- #

    def start(self):
        """Starts the background loop thread. Safe to call once at app
        startup; calling it twice is a no-op so an accidental double-import
        (e.g. under a reloader) doesn't spawn two loops."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.info("Verification agent already running, skipping start()")
                return
            self._status["enabled"] = True
            self._thread = threading.Thread(target=self._loop_forever, name="verification-agent", daemon=True)
            self._thread.start()
            logger.info("Verification agent background thread started")

    def run_once_now(self):
        """Wakes the loop immediately to run a pass right away, without
        waiting out the current cooldown. If a pass is already running this
        is a no-op (the loop is already doing the work)."""
        self._wake_event.set()

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # ---------------------------------------------------------------- #
    # Internals
    # ---------------------------------------------------------------- #

    def _seconds_until_due(self) -> float:
        """
        Checks the DB for the most recent last_checked_at across all sources
        to figure out how long ago the last full pass actually finished, so a
        server restart doesn't wipe out the cooldown and trigger an immediate
        re-run. Returns 0 if a pass is due now (or none has ever run).
        """
        db = SessionLocal()
        try:
            latest = db.query(Source.last_checked_at).order_by(
                Source.last_checked_at.desc()
            ).first()
            if not latest or not latest[0]:
                return 0.0
            last_checked = latest[0]
            if last_checked.tzinfo is None:
                last_checked = last_checked.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_checked).total_seconds()
            return max(0.0, COOLDOWN_SECONDS - elapsed)
        finally:
            db.close()

    def _loop_forever(self):
        # On startup, don't blindly run immediately — check if a pass
        # finished recently (survives server restarts) and wait out the rest
        # of the cooldown first if so.
        initial_wait = self._seconds_until_due()
        if initial_wait > 0:
            next_run = datetime.now(timezone.utc) + timedelta(seconds=initial_wait)
            with self._lock:
                self._status["next_pass_at"] = next_run.isoformat()
            logger.info(f"Verification agent: recent pass detected, waiting {initial_wait/3600:.1f}h before next pass")
            self._wake_event.clear()
            deadline = time.time() + initial_wait
            while time.time() < deadline:
                if self._wake_event.wait(timeout=30):
                    break

        while True:
            try:
                self._run_full_pass()
            except Exception as e:
                logger.exception(f"Verification agent pass crashed, will retry after cooldown: {e}")

            next_run = datetime.now(timezone.utc) + timedelta(seconds=COOLDOWN_SECONDS)
            with self._lock:
                self._status["next_pass_at"] = next_run.isoformat()
            logger.info(f"Verification agent sleeping until {next_run.isoformat()} (2-day cooldown)")

            # Sleep in short increments so run_once_now() can cut the wait
            # short instead of blocking a plain time.sleep(2 days).
            self._wake_event.clear()
            deadline = time.time() + COOLDOWN_SECONDS
            while time.time() < deadline:
                if self._wake_event.wait(timeout=30):
                    logger.info("Verification agent woken early via run_once_now()")
                    break

    def _run_full_pass(self):
        db = SessionLocal()
        try:
            sources = db.query(Source).order_by(Source.id.asc()).all()
            total = len(sources)
            started_at = datetime.now(timezone.utc)

            with self._lock:
                self._status.update({
                    "running_pass": True,
                    "last_pass_started_at": started_at.isoformat(),
                    "sources_checked": 0,
                    "sources_total": total,
                    "corrected_this_pass": 0,
                    "errors_this_pass": 0,
                    "current_url": None,
                })

            logger.info(f"Verification agent starting full pass over {total} source(s)")

            corrected = 0
            errors = 0
            for i, source in enumerate(sources):
                with self._lock:
                    self._status["current_url"] = source.url

                try:
                    result = verify_source(source, db)
                    if result["result"] == "corrected":
                        corrected += 1
                    elif result["result"] == "error":
                        errors += 1
                except Exception as e:
                    logger.error(f"[verify] Unhandled error checking {source.url}: {e}")
                    errors += 1
                    db.rollback()

                with self._lock:
                    self._status["sources_checked"] = i + 1
                    self._status["corrected_this_pass"] = corrected
                    self._status["errors_this_pass"] = errors

                time.sleep(PER_SOURCE_PAUSE_SECONDS)

            finished_at = datetime.now(timezone.utc)
            summary = (
                f"Checked {total} source(s): {corrected} corrected, "
                f"{errors} error(s), {total - corrected - errors} already up to date."
            )
            with self._lock:
                self._status.update({
                    "running_pass": False,
                    "last_pass_finished_at": finished_at.isoformat(),
                    "current_url": None,
                    "last_pass_summary": summary,
                })
            logger.info(f"Verification agent finished pass: {summary}")
        finally:
            db.close()


verification_agent = VerificationAgent()