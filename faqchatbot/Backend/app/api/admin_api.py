import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal

logger = logging.getLogger(__name__)
from app.models.source_model import Source
from app.models.job_model import ScrapeJob  # noqa: F401
from app.services.ingestion_pipeline import ingest_website
from app.services.sitemap_service import analyze_site, analyze_site_deep
from app.services.language_detection import filter_urls_by_language
from app.services.verification_agent import verification_agent
from app.services.verification_service import verify_source
from app.services import scrape_job_service

router = APIRouter(prefix="/admin", tags=["admin"])


class ScrapeRequest(BaseModel):
    urls: list[str] = Field(...)
    question_selector: str | None = Field(default=None)
    answer_selector: str | None = Field(default=None)
    crawl: bool = Field(default=False)
    max_pages: int = Field(default=20)


class ScrapeResponse(BaseModel):
    pages_scraped: int
    chunks_created: int
    topics_extracted: int
    total_words: int = 0
    status: str


class BulkScrapeRequest(BaseModel):
    urls: list[str] = Field(...)


class AnalyzeSiteRequest(BaseModel):
    url: str = Field(...)
    urls: list[str] | None = Field(default=None)


class FilterUrlsRequest(BaseModel):
    urls: list[str] = Field(...)
    languages: list[str] = Field(...)


class RangeScrapeRequest(BaseModel):
    url: str = Field(...)
    limit: int = Field(...)
    question_selector: str | None = Field(default=None)
    answer_selector: str | None = Field(default=None)


@router.post("/scrape", response_model=ScrapeResponse)
def scrape_faq_source(request: ScrapeRequest, db: Session = Depends(get_db)):
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")

    try:
        result = ingest_website(
            urls=request.urls, question_selector=request.question_selector,
            answer_selector=request.answer_selector, crawl=request.crawl, max_pages=request.max_pages,
        )
    except Exception as e:
        logger.exception("ingest_website failed")
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {e}")

    try:
        for url in request.urls:
            source = db.query(Source).filter(Source.url == url).first()
            if not source:
                source = Source(url=url)
                db.add(source)
            source.status = result["status"]
            source.chunks_created = result["chunks_created"]
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("Failed to save source status to DB")
        raise HTTPException(status_code=500, detail=f"Database write failed: {e}")

    return ScrapeResponse(**result)


@router.post("/analyze-site")
def analyze_site_endpoint(request: AnalyzeSiteRequest):
    return analyze_site(request.url)


@router.post("/analyze-site-deep")
def analyze_site_deep_endpoint(request: AnalyzeSiteRequest):
    return analyze_site_deep(request.url, urls_override=request.urls)


@router.post("/filter-urls-by-language")
def filter_urls_by_language_endpoint(request: FilterUrlsRequest):
    filtered = filter_urls_by_language(request.urls, request.languages)
    return {"urls": filtered, "count": len(filtered)}


def _resume_sanity_check(db: Session, job: ScrapeJob, urls: list[str]) -> None:
    if job.current_index <= 0:
        return
    last_url = urls[job.current_index - 1]
    source = db.query(Source).filter(Source.url == last_url).first()
    if not source:
        job.last_resume_check = f"skipped - no stored source for {last_url}"
        return
    try:
        result = verify_source(source, db)
        if result["result"] == "match":
            job.last_resume_check = f"ok - {last_url} unchanged since it was scraped"
        elif result["result"] == "corrected":
            job.last_resume_check = f"re-synced - {last_url} had drifted, corrected before continuing"
        else:
            job.last_resume_check = f"warning - could not re-verify {last_url} ({result.get('detail', 'unknown')})"
    except Exception as e:
        logger.warning("Resume sanity check failed for %s", last_url, exc_info=True)
        job.last_resume_check = f"warning - sanity check errored on {last_url}: {e}"
    db.commit()


def _run_bulk_scrape(job_id: str, is_resume: bool = False):
    db = SessionLocal()
    job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
    if not job:
        db.close()
        return
    urls = job.urls
    question_selector = job.question_selector
    answer_selector = job.answer_selector
    total_chunks = job.chunks_created
    total_words = job.total_words
    consecutive_errors = job.consecutive_errors or 0

    if is_resume:
        _resume_sanity_check(db, job, urls)

    i = job.current_index
    while i < len(urls):
        signal = scrape_job_service.get_control_signal(job_id)
        if not signal:
            db.close()
            return

        if signal["cancel_requested"]:
            job.status = "cancelled"
            job.pause_requested = False
            job.cancel_requested = False
            job.current_url = None
            db.commit()
            db.close()
            return

        if signal["pause_requested"]:
            job.status = "paused"
            job.pause_requested = False
            job.current_url = None
            db.commit()
            db.close()
            return

        url = urls[i]
        job.current_url = url
        db.commit()

        try:
            result = ingest_website(urls=[url], question_selector=question_selector, answer_selector=answer_selector)
            source = db.query(Source).filter(Source.url == url).first()
            if not source:
                source = Source(url=url)
                db.add(source)
            source.status = result["status"]
            source.chunks_created = result["chunks_created"]
            db.commit()
            total_chunks += result["chunks_created"]
            total_words += result.get("total_words", 0)
            consecutive_errors = 0
            i += 1
            job.current_index = i
            job.chunks_created = total_chunks
            job.total_words = total_words
            job.consecutive_errors = 0
            db.commit()
        except Exception as e:
            db.rollback()
            consecutive_errors += 1
            err_msg = f"Failed on {url}: {e}"
            logger.warning(err_msg)
            job.last_error = err_msg
            job.consecutive_errors = consecutive_errors
            job.chunks_created = total_chunks
            job.total_words = total_words
            if consecutive_errors >= scrape_job_service.AUTO_PAUSE_AFTER_CONSECUTIVE_ERRORS:
                job.status = "paused"
                job.pause_requested = False
                job.current_url = None
                job.last_error = (
                    f"Auto-paused after {consecutive_errors} consecutive failures "
                    f"(likely an API key or network issue) - {err_msg}"
                )
                db.commit()
                db.close()
                return
            db.commit()

    job.status = "completed"
    job.current_url = None
    db.commit()
    db.close()


@router.post("/scrape-bulk-async")
def scrape_bulk_async(request: BulkScrapeRequest, background_tasks: BackgroundTasks):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    job = scrape_job_service.create_job(urls=request.urls)
    background_tasks.add_task(_run_bulk_scrape, job.id)
    return {"job_id": job.id, "total_pages": len(request.urls)}


@router.post("/scrape-range-async")
def scrape_range_async(request: RangeScrapeRequest, background_tasks: BackgroundTasks):
    site_info = analyze_site(request.url)
    if not site_info["sitemap_found"]:
        raise HTTPException(status_code=400, detail="Could not find a sitemap for that site")
    limited_urls = site_info["all_urls"][:request.limit]
    if not limited_urls:
        raise HTTPException(status_code=400, detail="No pages found to scrape")
    job = scrape_job_service.create_job(
        urls=limited_urls,
        question_selector=request.question_selector,
        answer_selector=request.answer_selector,
    )
    background_tasks.add_task(_run_bulk_scrape, job.id)
    return {"job_id": job.id, "total_pages": len(limited_urls)}


@router.get("/scrape-status/{job_id}")
def scrape_status(job_id: str):
    job = scrape_job_service.get_job_dict(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/scrape-jobs")
def scrape_jobs():
    return scrape_job_service.list_resumable_jobs()


@router.post("/scrape-pause/{job_id}")
def scrape_pause(job_id: str):
    job = scrape_job_service.request_pause(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/scrape-cancel/{job_id}")
def scrape_cancel(job_id: str):
    job = scrape_job_service.request_cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/scrape-resume/{job_id}")
def scrape_resume(job_id: str, background_tasks: BackgroundTasks):
    job = scrape_job_service.mark_resuming(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or not paused")
    background_tasks.add_task(_run_bulk_scrape, job.id, True)
    return scrape_job_service.get_job_dict(job.id)


@router.get("/verification-status")
def verification_status():
    return verification_agent.get_status()


@router.post("/verification-run-now")
def verification_run_now():
    verification_agent.run_once_now()
    return verification_agent.get_status()


@router.post("/verification-pause")
def verification_pause():
    verification_agent.pause()
    return verification_agent.get_status()


@router.post("/verification-resume")
def verification_resume():
    verification_agent.resume()
    return verification_agent.get_status()