import os
import logging

# Must run before any app.* module grabs a logger, otherwise their INFO-level
# calls (e.g. the [robots]/[scraper] lines in robots_service.py and
# scraper_service.py) have no handler to print through and are silently
# dropped — Python's root logger defaults to WARNING with no handler at all.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.database import Base, engine, init_redis, close_redis, get_redis
from app.api import admin_api, question_api, source_api
from app.services.verification_agent import verification_agent
from app.services import scrape_job_service

logger = logging.getLogger(__name__)

try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.warning(f"Could not create database tables (DB may not be available): {e}")

app = FastAPI(title=settings.APP_NAME)

_cors_origins = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_api.router)
app.include_router(question_api.router)
app.include_router(source_api.router)


@app.on_event("startup")
async def startup():
    try:
        await init_redis()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
    try:
        recovered = scrape_job_service.recover_orphaned_jobs()
        if recovered:
            logger.info(f"Recovered {recovered} scrape job(s) left 'running' by a previous process - now resumable.")
    except Exception as e:
        logger.warning(f"Could not recover orphaned scrape jobs: {e}")
    try:
        verification_agent.start()
    except Exception as e:
        logger.warning(f"Could not start verification agent: {e}")


@app.on_event("shutdown")
async def shutdown():
    await close_redis()
    logger.info("Redis connection closed")


@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME}


@app.get("/ping-redis")
async def ping_redis():
    r = get_redis()
    await r.set("foo", "bar")
    value = await r.get("foo")
    return {"foo": value}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "..", "..", "Frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
else:
    logger.warning(f"Frontend dist folder not found at {FRONTEND_DIST}, skipping static file mount")