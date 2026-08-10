import json
import logging
import uuid

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.job_model import ScrapeJob
from app.services import control_signal

logger = logging.getLogger(__name__)

AUTO_PAUSE_AFTER_CONSECUTIVE_ERRORS = 3


def create_job(
    urls: list[str],
    question_selector: str | None = None,
    answer_selector: str | None = None,
    render_js: bool = False,
) -> ScrapeJob:
    db = SessionLocal()
    try:
        job = ScrapeJob(
            id=str(uuid.uuid4()),
            urls_json=json.dumps(urls),
            question_selector=question_selector,
            answer_selector=answer_selector,
            render_js=render_js,
            status="running",
            total_pages=len(urls),
            current_index=0,
            chunks_created=0,
            total_words=0,
            pause_requested=False,
            consecutive_errors=0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        control_signal.clear_signals(job.id)
        return job
    finally:
        db.close()


def get_job_dict(job_id: str) -> dict | None:
    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job:
            return None
        return _to_dict(job)
    finally:
        db.close()


def list_resumable_jobs() -> list[dict]:
    db = SessionLocal()
    try:
        jobs = (
            db.query(ScrapeJob)
            .filter(ScrapeJob.status == "paused")
            .order_by(ScrapeJob.updated_at.desc())
            .limit(10)
            .all()
        )
        return [_to_dict(j) for j in jobs]
    finally:
        db.close()


def recover_orphaned_jobs() -> int:
    db = SessionLocal()
    try:
        orphaned = db.query(ScrapeJob).filter(ScrapeJob.status == "running").all()
        for job in orphaned:
            job.status = "paused"
            job.current_url = None
            job.pause_requested = False
            control_signal.clear_signals(job.id)
            note = "Recovered after a server restart - hit Resume to continue from where it stopped."
            job.last_error = f"{job.last_error} | {note}" if job.last_error else note
        db.commit()
        return len(orphaned)
    finally:
        db.close()


def get_control_signal(job_id: str) -> dict | None:
    """Used by the bulk-scrape loop's periodic status check. Prefers the
    in-memory signal (instant, DB-independent) and falls back to DB-only
    values if the in-memory dict somehow has nothing for this job yet."""
    mem = control_signal.get_signal(job_id)
    db = SessionLocal()
    try:
        row = (
            db.query(ScrapeJob.status, ScrapeJob.pause_requested, ScrapeJob.cancel_requested)
            .filter(ScrapeJob.id == job_id)
            .first()
        )
        if not row:
            return None
        return {
            "status": row[0],
            "pause_requested": bool(row[1]) or mem["pause"],
            "cancel_requested": bool(row[2]) or mem["cancel"],
        }
    except Exception:
        logger.warning(
            "DB unreachable while checking control signal for %s; using in-memory only",
            job_id, exc_info=True,
        )
        return {"status": "running", "pause_requested": mem["pause"], "cancel_requested": mem["cancel"]}
    finally:
        db.close()


def request_pause(job_id: str) -> dict | None:
    # Instant, in-memory - takes effect even if the DB is unreachable.
    control_signal.request_pause(job_id)
    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job:
            return {"job_id": job_id, "pause_requested": True, "resumable": False}
        if job.status == "running":
            job.pause_requested = True
            db.commit()
            db.refresh(job)
        return _to_dict(job)
    except Exception:
        db.rollback()
        logger.warning(
            "Could not persist pause_requested to DB for job %s (pause still applied in-memory)",
            job_id, exc_info=True,
        )
        return {"job_id": job_id, "status": "running", "pause_requested": True, "cancel_requested": False}
    finally:
        db.close()


def request_cancel(job_id: str) -> dict | None:
    # Instant, in-memory - takes effect even if the DB is unreachable.
    control_signal.request_cancel(job_id)
    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job:
            return {"job_id": job_id, "cancel_requested": True, "resumable": False}
        if job.status == "running":
            job.cancel_requested = True
        elif job.status == "paused":
            job.status = "cancelled"
            job.cancel_requested = False
            job.pause_requested = False
            job.current_url = None
            control_signal.clear_signals(job_id)
        db.commit()
        db.refresh(job)
        return _to_dict(job)
    except Exception:
        db.rollback()
        logger.warning(
            "Could not persist cancel_requested to DB for job %s (cancel still applied in-memory)",
            job_id, exc_info=True,
        )
        return {"job_id": job_id, "status": "running", "pause_requested": False, "cancel_requested": True}
    finally:
        db.close()


def mark_resuming(job_id: str) -> ScrapeJob | None:
    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job or job.status != "paused":
            return None
        control_signal.clear_signals(job_id)
        job.status = "running"
        job.pause_requested = False
        job.cancel_requested = False
        job.consecutive_errors = 0
        job.last_error = None
        db.commit()
        db.refresh(job)
        db.expunge(job)
        return job
    finally:
        db.close()


def _to_dict(job: ScrapeJob) -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "total_pages": job.total_pages,
        "pages_done": job.current_index,
        "chunks_created": job.chunks_created,
        "total_words": job.total_words,
        "current_url": job.current_url,
        "error": job.last_error,
        "pause_requested": job.pause_requested,
        "cancel_requested": job.cancel_requested,
        "consecutive_errors": job.consecutive_errors,
        "last_resume_check": job.last_resume_check,
        "resumable": job.status == "paused",
        "cancelled": job.status == "cancelled",
    }