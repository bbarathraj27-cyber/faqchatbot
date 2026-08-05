import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from app.services import scrape_job_service, control_signal

router = APIRouter(prefix="/admin", tags=["admin"])

# How many URLs to have in flight at once during bulk ingestion.
# 10 is a moderate default that's fast without hammering Groq/Pinecone rate
# limits too hard. Tune down (5) if you see rate-limit errors, up (20+) if
# your API tiers can absorb it.
BULK_SCRAPE_CONCURRENCY = 10


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

    # Stage 1: make it clear in the UI that this page was ALREADY scraped
    # before the pause, and we're just re-verifying it hasn't changed - not
    # scraping it again from scratch. Previously job.current_url stayed None
    # for the whole sanity check (it's cleared on pause), which made a
    # resume look stuck even when it was actively re-verifying the last page.
    job.current_url = f"already scraped - re-checking it's still up to date: {last_url}"
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Could not persist resume-check progress for job %s", job.id, exc_info=True)

    source = db.query(Source).filter(Source.url == last_url).first()
    if not source:
        job.last_resume_check = f"skipped - no stored source for {last_url}"
    else:
        try:
            # correct_on_drift=False: this check exists to give a quick
            # signal before continuing, not to trigger a full re-ingest
            # (topic extraction + re-embedding) that can legitimately take
            # minutes if the key pools are still rate-limited from the
            # scraping that just happened. A real drift gets caught and
            # fixed by the separate self-healing verification agent
            # instead - it doesn't block resuming the bulk import.
            result = verify_source(source, db, correct_on_drift=False)
            if result["result"] == "match":
                job.last_resume_check = f"ok - {last_url} unchanged since it was scraped"
            elif result["result"] == "drifted":
                job.last_resume_check = (
                    f"note - {last_url} appears to have changed since it was scraped "
                    f"(similarity {result.get('similarity', 0):.2f}); the self-healing "
                    f"agent will correct it on its next pass"
                )
            else:
                job.last_resume_check = f"warning - could not re-verify {last_url} ({result.get('detail', 'unknown')})"
        except Exception as e:
            logger.warning("Resume sanity check failed for %s", last_url, exc_info=True)
            job.last_resume_check = f"warning - sanity check errored on {last_url}: {e}"

    # Stage 2: recheck is done - show an explicit transition message before
    # the main loop takes over and current_url flips to the next new page.
    # This makes it visually clear "the recheck finished, now moving on to
    # scraping" instead of jumping straight from the recheck message to a
    # new URL with nothing in between.
    job.current_url = "recheck complete - resuming scraping..."
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Could not persist resume-check result for job %s", job.id, exc_info=True)


def _run_bulk_scrape(job_id: str, is_resume: bool = False):
    # Unconditionally clear any stale pause/cancel flag before doing
    # anything else - a fresh start should never inherit a leftover signal
    # from a previous run of this same job_id, and this also protects
    # against the race where a Pause/Cancel click lands in the split
    # second between mark_resuming() and this task actually starting.
    control_signal.clear_signals(job_id)
    control_signal.set_current_job(job_id)

    db = SessionLocal()
    try:
        job = db.query(ScrapeJob).filter(ScrapeJob.id == job_id).first()
        if not job:
            return
        urls = job.urls
        question_selector = job.question_selector
        answer_selector = job.answer_selector
        total_chunks = job.chunks_created
        total_words = job.total_words
        consecutive_errors = job.consecutive_errors or 0

        if is_resume:
            _resume_sanity_check(db, job, urls)
            # Bail out cleanly if a pause/cancel arrived while the sanity
            # check itself was running, instead of barreling into the loop.
            sig = control_signal.get_signal(job_id)
            if sig["cancel"]:
                job.status = "cancelled"
                job.pause_requested = False
                job.cancel_requested = False
                job.current_url = None
                _safe_commit(db, job_id)
                return
            if sig["pause"]:
                job.status = "paused"
                job.pause_requested = False
                job.current_url = None
                _safe_commit(db, job_id)
                return

        i = job.current_index

        def _scrape_one(url: str):
            # Runs in a worker thread. Never raises -- exceptions are
            # captured and returned so the main thread does all DB writes
            # (SQLAlchemy sessions aren't thread-safe to share across threads).
            #
            # control_signal's abort-check plumbing (used by the Gemini/Groq
            # key-pool cooldown waits) is keyed off THREAD-LOCAL storage,
            # under the original assumption that one job = one thread
            # start-to-finish. With a thread pool, each task can land on a
            # different worker thread, so job_id must be (re)associated with
            # whichever thread actually runs this call -- otherwise
            # interruptible_sleep()/raise_if_abort_requested() silently see
            # no job_id and pause/cancel stops interrupting in-flight key
            # cooldowns.
            control_signal.set_current_job(job_id)
            try:
                result = ingest_website(
                    urls=[url],
                    question_selector=question_selector,
                    answer_selector=answer_selector,
                    # extract_topics=False: bulk ingestion must never depend on
                    # the Groq topic-extraction call. With ~18.8k URLs, leaving
                    # this on burns through all Groq keys' daily token quota
                    # (200k TPD each) within a small fraction of the run, after
                    # which every remaining page just cycles through
                    # rate-limit retries/cooldowns doing no real ingestion
                    # work. Topics are backfilled separately
                    # (see backfill_topics.py) at a sustainable rate after
                    # content is in.
                    extract_topics=False,
                )
                return url, result, None
            except control_signal.AbortRequested:
                # Pause/cancel fired mid-call. Not a real failure -- signal it
                # distinctly so the batch-apply loop below leaves
                # job.current_index untouched for this (and any later) URL in
                # the batch, exactly like the old sequential loop's behavior.
                return url, None, "ABORTED"
            except Exception as e:
                return url, None, e
            finally:
                # Don't leak this job_id association to whatever unrelated
                # task the pool runs on this thread next.
                control_signal.set_current_job(None)

        with ThreadPoolExecutor(max_workers=BULK_SCRAPE_CONCURRENCY) as executor:
            while i < len(urls):
                # Fast, in-memory, DB-independent check FIRST - this is what
                # makes Pause/Cancel take effect almost instantly instead of
                # waiting on a DB round-trip (or hanging if Neon is down).
                sig = control_signal.get_signal(job_id)

                if sig["cancel"]:
                    job.status = "cancelled"
                    job.pause_requested = False
                    job.cancel_requested = False
                    job.current_url = None
                    _safe_commit(db, job_id)
                    return

                if sig["pause"]:
                    job.status = "paused"
                    job.pause_requested = False
                    job.current_url = None
                    _safe_commit(db, job_id)
                    return

                # Dispatch the next batch (sliding window) concurrently.
                batch_urls = urls[i:i + BULK_SCRAPE_CONCURRENCY]
                job.current_url = (
                    f"{batch_urls[0]} (+{len(batch_urls) - 1} more in flight)"
                    if len(batch_urls) > 1 else batch_urls[0]
                )
                _safe_commit(db, job_id)

                futures = {executor.submit(_scrape_one, u): idx for idx, u in enumerate(batch_urls)}
                results_by_index = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    results_by_index[idx] = future.result()

                # Apply results back in original URL order so job.current_index
                # still means "everything before this index is done", and so
                # consecutive_errors / auto-pause behave the same as the old
                # sequential loop regardless of which URL happened to finish
                # first in the thread pool.
                pause_or_cancel_mid_batch = False
                for idx in range(len(batch_urls)):
                    url, result, error = results_by_index[idx]

                    if error == "ABORTED":
                        # Pause/cancel was mid-flight for this URL. Stop
                        # applying the rest of this batch's results -- i is
                        # left exactly where it is, so this URL (and any
                        # after it in the batch) get re-attempted from
                        # scratch on the next run/resume, matching the old
                        # sequential loop's AbortRequested handling.
                        db.rollback()
                        pause_or_cancel_mid_batch = True
                        break

                    if error is not None:
                        db.rollback()
                        consecutive_errors += 1
                        err_msg = f"Failed on {url}: {error}"
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
                            _safe_commit(db, job_id)
                            return
                        i += 1
                        job.current_index = i
                        _safe_commit(db, job_id)
                        continue

                    source = db.query(Source).filter(Source.url == url).first()
                    if not source:
                        source = Source(url=url)
                        db.add(source)
                    source.status = result["status"]
                    source.chunks_created = result["chunks_created"]
                    total_chunks += result["chunks_created"]
                    total_words += result.get("total_words", 0)
                    consecutive_errors = 0
                    i += 1
                    job.current_index = i
                    job.chunks_created = total_chunks
                    job.total_words = total_words
                    job.consecutive_errors = 0
                    _safe_commit(db, job_id)

                    # Re-check signals between URLs within the batch too, so a
                    # pause/cancel doesn't wait for the whole in-flight
                    # batch's bookkeeping to finish before being honored.
                    sig = control_signal.get_signal(job_id)
                    if sig["cancel"] or sig["pause"]:
                        pause_or_cancel_mid_batch = True
                        break

                if pause_or_cancel_mid_batch:
                    continue  # top of while loop will see the signal and exit cleanly

        job.status = "completed"
        job.current_url = None
        _safe_commit(db, job_id)
    finally:
        control_signal.set_current_job(None)
        db.close()


def _safe_commit(db: Session, job_id: str) -> None:
    """Commit, but never let a DB hiccup (e.g. Neon free-tier quota/outage)
    crash the background job silently. On failure the in-memory job state
    (and control_signal flags) are still correct - only the DB write is
    lost, and the next successful commit will catch it up."""
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("DB commit failed for job %s (continuing in-memory)", job_id, exc_info=True)


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