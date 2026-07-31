import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal

logger = logging.getLogger(__name__)
from app.models.source_model import Source
from app.services.ingestion_pipeline import ingest_website
from app.services.sitemap_service import analyze_site, analyze_site_deep
from app.services.job_tracker import (
    create_job,
    update_job,
    get_job,
    request_pause,
    consume_pause_request,
    get_job_urls,
)
from app.services.verification_agent import verification_agent

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
    return analyze_site_deep(request.url)


def _is_systemic_error(exc: Exception) -> bool:
    """
    Distinguishes a systemic failure (API key expired/invalid, rate limit,
    network/DNS down) from a single-page problem (that one page 404s, has
    unparseable content, etc).

    A single bad page should be skipped so an 18,000-page job doesn't grind
    to a halt over one broken link. A systemic failure should instead PAUSE
    the whole job on the page it was about to process (not skip it), so
    that once the key/network issue is fixed, resume retries that exact
    page instead of silently losing it.
    """
    text = str(exc).lower()
    signals = (
        "api key", "api_key_invalid", "unauthorized", "401", "403",
        "resource_exhausted", "429", "rate limit",
        "connection", "timeout", "timed out", "network",
        "name resolution", "max retries exceeded", "ssl",
    )
    return any(s in text for s in signals)


def _run_bulk_scrape(job_id: str, urls: list[str], start_index: int = 0):
    db = SessionLocal()
    job = get_job(job_id) or {}
    total_chunks = job.get("chunks_created", 0)
    total_words = job.get("total_words", 0)

    i = start_index
    try:
        while i < len(urls):
            # Check for a pause request BEFORE starting the next page, never
            # mid-page - a page that's already downloading/embedding always
            # finishes and gets saved before the job actually stops.
            if consume_pause_request(job_id):
                update_job(job_id, status="paused", resume_index=i, current_url=None)
                return

            url = urls[i]
            update_job(job_id, current_url=url, status="running")
            try:
                result = ingest_website(urls=[url], question_selector=None, answer_selector=None)
                source = db.query(Source).filter(Source.url == url).first()
                if not source:
                    source = Source(url=url)
                    db.add(source)
                source.status = result["status"]
                source.chunks_created = result["chunks_created"]
                db.commit()
                total_chunks += result["chunks_created"]
                total_words += result.get("total_words", 0)
                i += 1
                update_job(
                    job_id, pages_done=i, chunks_created=total_chunks,
                    total_words=total_words, resume_index=i, error=None,
                )
            except Exception as e:
                if _is_systemic_error(e):
                    # Pause on THIS page (don't advance i) so resume retries
                    # it instead of skipping it - the failure is likely to
                    # affect every subsequent page too (expired key, network
                    # down), so there's no point burning through the rest of
                    # the list right now.
                    logger.error(f"Systemic error on {url}, pausing job {job_id}: {e}")
                    update_job(
                        job_id, status="paused", resume_index=i, current_url=None,
                        error=f"Paused - systemic error on {url}: {e}",
                    )
                    return
                else:
                    # Page-specific issue - log it, skip just this one page,
                    # and keep going so one broken link doesn't stall an
                    # 18,000-page job.
                    logger.warning(f"Skipping {url} (page-level error): {e}")
                    i += 1
                    update_job(job_id, pages_done=i, resume_index=i, error=f"Skipped {url}: {e}")

        update_job(job_id, status="completed", current_url=None, resume_index=i)
    finally:
        db.close()


@router.post("/scrape-bulk-async")
def scrape_bulk_async(request: BulkScrapeRequest, background_tasks: BackgroundTasks):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    job_id = create_job(total_pages=len(request.urls), urls=request.urls)
    background_tasks.add_task(_run_bulk_scrape, job_id, request.urls)
    return {"job_id": job_id, "total_pages": len(request.urls)}


@router.post("/scrape-range-async")
def scrape_range_async(request: RangeScrapeRequest, background_tasks: BackgroundTasks):
    site_info = analyze_site(request.url)
    if not site_info["sitemap_found"]:
        raise HTTPException(status_code=400, detail="Could not find a sitemap for that site")
    limited_urls = site_info["all_urls"][:request.limit]
    if not limited_urls:
        raise HTTPException(status_code=400, detail="No pages found to scrape")
    job_id = create_job(total_pages=len(limited_urls), urls=limited_urls)
    background_tasks.add_task(_run_bulk_scrape, job_id, limited_urls)
    return {"job_id": job_id, "total_pages": len(limited_urls)}


@router.get("/scrape-status/{job_id}")
def scrape_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/scrape-pause/{job_id}")
def scrape_pause(job_id: str):
    """
    Requests a pause. The background loop checks this between pages (never
    mid-page), so whatever page is currently being scraped always finishes
    and gets saved first. Status flips to "paused" once the loop actually
    stops - poll /scrape-status/{job_id} to see when that's happened.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("running",):
        raise HTTPException(status_code=400, detail=f"Job is '{job['status']}', not running - nothing to pause")
    request_pause(job_id)
    return {"job_id": job_id, "pause_requested": True}


@router.post("/scrape-resume/{job_id}")
def scrape_resume(job_id: str, background_tasks: BackgroundTasks):
    """
    Resumes a paused job from exactly where it stopped - same page index,
    same URL list. Nothing already scraped gets re-processed.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "paused":
        raise HTTPException(status_code=400, detail=f"Job is '{job['status']}', not paused - nothing to resume")

    urls = get_job_urls(job_id)
    if not urls:
        raise HTTPException(status_code=400, detail="Original URL list for this job is no longer available")

    resume_index = job.get("resume_index", 0)
    update_job(job_id, status="running", error=None)
    background_tasks.add_task(_run_bulk_scrape, job_id, urls, resume_index)
    return {"job_id": job_id, "resumed_from_page": resume_index, "total_pages": len(urls)}


@router.get("/verification-status")
def verification_status():
    """Status of the background re-verification agent: whether it's mid-pass
    right now, when its last pass finished, when the next one is scheduled
    (2-day cooldown after each full pass), and a running tally for the
    current pass."""
    return verification_agent.get_status()


@router.post("/verification-run-now")
def verification_run_now():
    """Wakes the verification agent immediately instead of waiting out the
    remainder of its 2-day cooldown. If a pass is already in progress this
    just returns its current status."""
    verification_agent.run_once_now()
    return verification_agent.get_status()


@router.post("/verification-pause")
def verification_pause():
    """Pauses the agent. If a pass is mid-run, it finishes whichever source
    it's currently checking, then blocks before starting the next one."""
    verification_agent.pause()
    return verification_agent.get_status()


@router.post("/verification-resume")
def verification_resume():
    """Resumes a paused agent - it continues the same pass from wherever it
    left off."""
    verification_agent.resume()
    return verification_agent.get_status()