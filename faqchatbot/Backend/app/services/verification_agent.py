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
        # Set = not paused, work proceeds. Cleared = paused, the per-source
        # loop in _run_full_pass blocks on this until resume() is called.
        # Starts set (not paused) so a fresh process behaves as before.
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._status = {
            "running_pass": False,
            "enabled": False,
            "paused": False,
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

    def pause(self):
        """
        Pauses work as soon as possible. If a pass is mid-run, the
        currently-in-flight source finishes (verify_source isn't
        interrupted partway through - that would risk leaving a source in
        a half-corrected state), then the loop blocks before starting the
        next source until resume() is called. If no pass is running,
        pausing just prevents the next pass from starting when the
        cooldown ends.
        """
        with self._lock:
            self._pause_event.clear()
            self._status["paused"] = True
        logger.info("Verification agent paused")

    def resume(self):
        """Resumes a paused agent - the loop picks back up exactly where it
        left off (same pass, same source list, same counters)."""
        with self._lock:
            self._pause_event.set()
            self._status["paused"] = False
        logger.info("Verification agent resumed")

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
        # Only used to pull the initial source list. Deliberately NOT kept
        # open for the whole pass - see the per-source session below.
        db = SessionLocal()
        try:
            # Only source IDs are needed to drive the loop - the actual
            # Source row is re-fetched fresh (in its own short-lived
            # session) right before each verify_source() call below.
            source_ids = [row[0] for row in db.query(Source.id).order_by(Source.id.asc()).all()]
        finally:
            db.close()

        total = len(source_ids)
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
        for i, source_id in enumerate(source_ids):
            # Blocks here (no timeout) if pause() was called, until
            # resume() sets the event again. Checked BEFORE starting
            # the next source, never mid-verify_source(), so a paused
            # agent never leaves a source half-corrected.
            self._pause_event.wait()

            # A fresh session per source, opened and closed right around
            # the single verify_source() call. Deliberately NOT one session
            # held for the whole (potentially hours-long, 230+ source)
            # pass: Neon/Postgres drops idle connections, and a session
            # checked out for the entire pass never returns to the pool
            # between sources, so pool_pre_ping never gets a chance to
            # catch and replace a connection that died in between. Short-
            # lived sessions keep each connection's lifetime well inside
            # any idle timeout and let pre-ping do its job on every source.
            db = SessionLocal()
            url = None
            try:
                source = db.get(Source, source_id)
                if source is None:
                    # Deleted between listing IDs and getting here.
                    continue
                url = source.url

                with self._lock:
                    self._status["current_url"] = url

                result = verify_source(source, db)
                if result["result"] == "corrected":
                    corrected += 1
                elif result["result"] == "error":
                    errors += 1
            except Exception as e:
                # `url` was captured (if we got that far) before anything
                # that could have broken the session, so logging it here
                # never risks touching an expired/detached ORM attribute
                # against a dead connection - that's what previously
                # turned a single transient error into a whole-pass crash.
                logger.error(f"[verify] Unhandled error checking {url or f'source id={source_id}'}: {e}")
                errors += 1
            finally:
                db.close()

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


verification_agent = VerificationAgent()