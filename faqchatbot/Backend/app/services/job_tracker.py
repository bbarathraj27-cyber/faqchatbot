import uuid
import threading

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job(total_pages: int) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "total_pages": total_pages,
            "pages_done": 0,
            "chunks_created": 0,
            "total_words": 0,
            "current_url": None,
            "error": None,
        }
    return job_id


def update_job(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)