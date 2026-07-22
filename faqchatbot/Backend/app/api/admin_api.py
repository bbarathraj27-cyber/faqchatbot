from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.source_model import Source
from app.services.ingestion_pipeline import ingest_website
from app.services.sitemap_service import analyze_site, analyze_site_deep
from app.services.job_tracker import create_job, update_job, get_job

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
    result = ingest_website(
        urls=request.urls, question_selector=request.question_selector,
        answer_selector=request.answer_selector, crawl=request.crawl, max_pages=request.max_pages,
    )
    for url in request.urls:
        source = db.query(Source).filter(Source.url == url).first()
        if not source:
            source = Source(url=url)
            db.add(source)
        source.status = result["status"]
        source.chunks_created = result["chunks_created"]
    db.commit()
    return ScrapeResponse(**result)


@router.post("/analyze-site")
def analyze_site_endpoint(request: AnalyzeSiteRequest):
    return analyze_site(request.url)


@router.post("/analyze-site-deep")
def analyze_site_deep_endpoint(request: AnalyzeSiteRequest):
    return analyze_site_deep(request.url)


def _run_bulk_scrape(job_id: str, urls: list[str]):
    db = SessionLocal()
    total_chunks = 0
    total_words = 0
    for i, url in enumerate(urls):
        update_job(job_id, current_url=url)
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
        except Exception as e:
            update_job(job_id, error=f"Failed on {url}: {str(e)}")
        update_job(job_id, pages_done=i + 1, chunks_created=total_chunks, total_words=total_words)
    update_job(job_id, status="completed", current_url=None)
    db.close()


@router.post("/scrape-bulk-async")
def scrape_bulk_async(request: BulkScrapeRequest, background_tasks: BackgroundTasks):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    job_id = create_job(total_pages=len(request.urls))
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
    job_id = create_job(total_pages=len(limited_urls))
    background_tasks.add_task(_run_bulk_scrape, job_id, limited_urls)
    return {"job_id": job_id, "total_pages": len(limited_urls)}


@router.get("/scrape-status/{job_id}")
def scrape_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job