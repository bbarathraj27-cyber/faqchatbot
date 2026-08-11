"""
robots.txt compliance for the scraper.

Fetches and caches each domain's robots.txt (one fetch per domain, shared
across the whole crawl), then answers two questions before every page fetch:

  * is_allowed(url)      -> may we fetch this URL at all?
  * get_crawl_delay(url) -> did the site ask for a slower crawl rate?

Both a sync and an async API are provided since scraper_service.py has both
sync (requests) and async (httpx/Playwright) fetch paths.

We deliberately fetch robots.txt ourselves (via requests/httpx) and feed the
raw text into urllib.robotparser.RobotFileParser.parse(), rather than calling
parser.read(), because .read() does a blocking, non-configurable urlopen()
call that would (a) bypass our timeout/retry settings and (b) block the
event loop if ever awaited from async code.

KNOWN LIMITATION: Python's stdlib urllib.robotparser does not support the
"*" / "$" wildcard extensions used by many real-world robots.txt files
(e.g. "Disallow: /search*"), only literal path prefixes. Plain prefix rules
like "Disallow: /admin/" are enforced correctly; wildcarded rules are
silently ignored by the stdlib parser and will NOT be blocked. If a target
site relies heavily on wildcard rules, swap RobotFileParser for a
wildcard-aware parser such as the `protego` package (used by Scrapy).
"""

import asyncio
import logging
import threading
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import requests

from app.config import settings

logger = logging.getLogger(__name__)


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt disallows fetching a URL.

    Subclasses RuntimeError so it's caught for free by the existing
    `except (..., RuntimeError)` handlers in the crawl loops below,
    without needing to touch every call site individually.
    """

    def __init__(self, url: str):
        super().__init__(f"Blocked by robots.txt: {url}")
        self.url = url


def _robots_txt_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def _domain_key(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


# --- Sync cache -------------------------------------------------------
# None cached for a domain means "robots.txt could not be fetched" — the
# caller decides fail-open vs fail-closed via settings.SCRAPER_ROBOTS_FAIL_OPEN.
_sync_cache: dict[str, Optional[RobotFileParser]] = {}
_sync_lock = threading.Lock()


def _fetch_and_parse_sync(url: str) -> Optional[RobotFileParser]:
    robots_url = _robots_txt_url(url)
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        resp = requests.get(
            robots_url,
            timeout=settings.SCRAPER_ROBOTS_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": settings.SCRAPER_ROBOTS_USER_AGENT},
        )
        if resp.status_code == 404:
            # No robots.txt published => everything is allowed.
            parser.parse([])
            return parser
        resp.raise_for_status()
        parser.parse(resp.text.splitlines())
        return parser
    except requests.RequestException as exc:
        logger.warning(f"[robots] Could not fetch {robots_url} ({exc})")
        return None


def get_robot_parser(url: str) -> Optional[RobotFileParser]:
    if not settings.SCRAPER_RESPECT_ROBOTS_TXT:
        return None
    domain = _domain_key(url)
    with _sync_lock:
        if domain in _sync_cache:
            return _sync_cache[domain]
    parser = _fetch_and_parse_sync(url)
    with _sync_lock:
        _sync_cache[domain] = parser
    return parser


def is_allowed(url: str) -> bool:
    """True if robots.txt permits fetching this URL (or checking is
    disabled, or robots.txt couldn't be fetched and fail-open is set)."""
    if not settings.SCRAPER_RESPECT_ROBOTS_TXT:
        return True
    parser = get_robot_parser(url)
    if parser is None:
        return settings.SCRAPER_ROBOTS_FAIL_OPEN
    try:
        return parser.can_fetch(settings.SCRAPER_ROBOTS_USER_AGENT, url)
    except Exception as exc:  # malformed robots.txt shouldn't crash a crawl
        logger.warning(f"[robots] Error evaluating rules for {url} ({exc}); failing open")
        return True


def get_crawl_delay(url: str) -> Optional[float]:
    if not settings.SCRAPER_RESPECT_ROBOTS_TXT:
        return None
    parser = get_robot_parser(url)
    if parser is None:
        return None
    try:
        delay = parser.crawl_delay(settings.SCRAPER_ROBOTS_USER_AGENT)
        return float(delay) if delay is not None else None
    except Exception:
        return None


# --- Async cache --------------------------------------------------------
_async_cache: dict[str, Optional[RobotFileParser]] = {}
_async_domain_locks: dict[str, asyncio.Lock] = {}
_async_locks_guard = asyncio.Lock()


async def _fetch_and_parse_async(url: str, client: Optional[httpx.AsyncClient] = None) -> Optional[RobotFileParser]:
    robots_url = _robots_txt_url(url)
    parser = RobotFileParser()
    parser.set_url(robots_url)

    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=settings.SCRAPER_ROBOTS_FETCH_TIMEOUT_SECONDS)
        own_client = True

    try:
        resp = await client.get(robots_url, headers={"User-Agent": settings.SCRAPER_ROBOTS_USER_AGENT})
        if resp.status_code == 404:
            parser.parse([])
            return parser
        resp.raise_for_status()
        parser.parse(resp.text.splitlines())
        return parser
    except httpx.HTTPError as exc:
        logger.warning(f"[robots] Could not fetch {robots_url} ({exc})")
        return None
    finally:
        if own_client:
            await client.aclose()


async def async_get_robot_parser(url: str, client: Optional[httpx.AsyncClient] = None) -> Optional[RobotFileParser]:
    if not settings.SCRAPER_RESPECT_ROBOTS_TXT:
        return None

    domain = _domain_key(url)
    if domain in _async_cache:
        return _async_cache[domain]

    async with _async_locks_guard:
        lock = _async_domain_locks.setdefault(domain, asyncio.Lock())

    async with lock:
        if domain in _async_cache:  # re-check: another task may have just filled it
            return _async_cache[domain]
        parser = await _fetch_and_parse_async(url, client=client)
        _async_cache[domain] = parser
        return parser


async def async_is_allowed(url: str, client: Optional[httpx.AsyncClient] = None) -> bool:
    if not settings.SCRAPER_RESPECT_ROBOTS_TXT:
        return True
    parser = await async_get_robot_parser(url, client=client)
    if parser is None:
        return settings.SCRAPER_ROBOTS_FAIL_OPEN
    try:
        return parser.can_fetch(settings.SCRAPER_ROBOTS_USER_AGENT, url)
    except Exception as exc:
        logger.warning(f"[robots] Error evaluating rules for {url} ({exc}); failing open")
        return True


async def async_get_crawl_delay(url: str, client: Optional[httpx.AsyncClient] = None) -> Optional[float]:
    if not settings.SCRAPER_RESPECT_ROBOTS_TXT:
        return None
    parser = await async_get_robot_parser(url, client=client)
    if parser is None:
        return None
    try:
        delay = parser.crawl_delay(settings.SCRAPER_ROBOTS_USER_AGENT)
        return float(delay) if delay is not None else None
    except Exception:
        return None