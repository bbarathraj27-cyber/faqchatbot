"""
DB-backed control layer for bulk scrape/import jobs.

This replaces the old pure-in-memory job_tracker for bulk scrapes: every
job is a row in `scrape_jobs`, and every meaningful state change (a page
finishing, a pause request, a resume, an auto-pause on repeated errors)
is committed immediately. That's what makes pause/resume actually work
across a browser tab close, a page refresh, or the backend process itself
restarting — there's no window where progress only exists in RAM.

Nothing here talks to Groq/Gemini/Pinecone directly; admin_api.py wires
this together with ingestion_pipeline.ingest_website for the actual work.
"""

import json
import logging
import uuid

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.job_model import ScrapeJob

logger = logging.getLogger(__name__)

# After this many consecutive per-URL failures, stop and auto-pause instead
# of ploughing through the rest of the list with (likely) every remaining
# page failing the same way — e.g. every API key in the pool is exhausted,
# or the network dropped. Progress already made is untouched; the failing
# URL is retried first on resume.
AUTO_PAUSE_AFTER_CONSECUTIVE_ERRORS = 3


def create_job(
    urls: list[str],
    question_selector: str | None = None,
    answer_selector: str | None = None,
) -> ScrapeJob:
    db = SessionLocal()
    try:
        job = ScrapeJob(
            id=str(uuid.uuid4()),
            urls_json=json.dumps(urls),
            question_selector=question_selector,
            answer_selector=answer_selector,
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
    """Paused (manually or auto) jobs that a 'Resume' button in the admin UI
    can offer, even after a full page refresh or backend restart."""
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


def request_pause(job_id: str) -> dict | None:
    """Flags the job to stop after it finishes whatever page it's currently
    on. The running background loop notices this on its next iteration —
    it does not kill anything mid-page."""
    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job:
            return None
        if job.status == "running":
            job.pause_requested = True
            db.commit()
            db.refresh(job)
        return _to_dict(job)
    finally:
        db.close()


def mark_resuming(job_id: str) -> ScrapeJob | None:
    """Flips a paused job back to running. Returns the ORM object (not a
    dict) since the caller needs the real urls/selectors to relaunch the
    background task."""
    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job or job.status != "paused":
            return None
        job.status = "running"
        job.pause_requested = False
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
        "consecutive_errors": job.consecutive_errors,
        "last_resume_check": job.last_resume_check,
        "resumable": job.status == "paused",
    }