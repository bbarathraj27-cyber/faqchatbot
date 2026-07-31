import time
import uuid
import threading

_jobs: dict[str, dict] = {}
_lock = threading.Lock()

# In-memory job tracker for ingestion progress (see admin_api.py). Since
# _jobs never had any cleanup, a long-running server accumulates one entry
# per ingest job forever - a slow but real memory leak. Jobs older than
# this TTL are purged the next time a job is created or updated, so completed
# (or abandoned/never-polled) jobs eventually get reclaimed without needing
# a background sweep thread.
JOB_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def _purge_expired_locked() -> None:
    """Caller must hold _lock. Drops any job whose last update is older than
    JOB_TTL_SECONDS."""
    now = time.time()
    expired = [
        job_id
        for job_id, job in _jobs.items()
        if now - job.get("_updated_at", now) > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        del _jobs[job_id]


def create_job(total_pages: int, urls: list[str] | None = None) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _purge_expired_locked()
        _jobs[job_id] = {
            "status": "running",
            "total_pages": total_pages,
            "pages_done": 0,
            "chunks_created": 0,
            "total_words": 0,
            "current_url": None,
            "error": None,
            "resume_index": 0,
            "_urls": urls or [],
            "_pause_requested": False,
            "_updated_at": time.time(),
        }
    return job_id


def update_job(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["_updated_at"] = time.time()
        _purge_expired_locked()


def request_pause(job_id: str) -> bool:
    """Flags a running job to pause before it starts its next page. The
    background loop checks this flag between pages (never mid-page), so
    a page that's already downloading/embedding always finishes and gets
    saved before the job actually stops."""
    with _lock:
        if job_id not in _jobs:
            return False
        _jobs[job_id]["_pause_requested"] = True
        _jobs[job_id]["_updated_at"] = time.time()
        return True


def consume_pause_request(job_id: str) -> bool:
    """Checks and clears the pause flag in one step, so a stray resume
    doesn't immediately re-trigger the same pause request."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        was_requested = job.get("_pause_requested", False)
        job["_pause_requested"] = False
        return was_requested


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        # Don't leak internal bookkeeping fields to callers/API responses.
        return {k: v for k, v in job.items() if k not in ("_updated_at", "_pause_requested", "_urls")}


def get_job_urls(job_id: str) -> list[str] | None:
    """Internal use only (resume needs the original URL list back)."""
    with _lock:
        job = _jobs.get(job_id)
        return job.get("_urls") if job else None