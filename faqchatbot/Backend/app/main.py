import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import Base, engine, init_redis, close_redis, get_redis
from app.api import admin_api, question_api, source_api
from app.services.verification_agent import verification_agent

logger = logging.getLogger(__name__)

try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.warning(f"Could not create database tables (DB may not be available): {e}")

app = FastAPI(title=settings.APP_NAME)

# CORS_ALLOWED_ORIGINS is a comma-separated list from .env (e.g.
# "https://myapp.com,https://admin.myapp.com"). Falls back to "*" only if
# unset, so local dev keeps working exactly as before. Wildcard "*" combined
# with allow_credentials=True effectively lets any origin make credentialed
# requests - fine for local dev, risky once JWT/cookie auth is in play, so
# this should be set to explicit origins before going public-facing.
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


# Serve the built frontend — MUST be the last thing added, after all API routes,
# otherwise it swallows requests meant for /question, /admin, /sources etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Backend/app
FRONTEND_DIST = os.path.join(BASE_DIR, "..", "..", "Frontend", "dist")

# ... your existing API routes above (e.g. app.include_router(...)) ...
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")