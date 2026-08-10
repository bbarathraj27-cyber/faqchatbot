import json
import logging
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs

import requests
import httpx
import asyncio
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)

# --- Anti-bot handling: rotate a small pool of realistic desktop UAs so a
# long crawl doesn't hammer a site with one identical header on every
# request (a common, cheap bot-detection signal). ---
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]


def _build_headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


# Kept for any external code importing HEADERS directly.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FAQChatbotScraper/1.0; +https://example.com/bot-info)"
}
REQUEST_TIMEOUT_SECONDS = 15
_RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

# Common locale codes seen either as a `?hl=` / `?lang=` query param or as a
# leading path segment (e.g. tensorflow.org/ko/..., site.com/zh-cn/...).
_ENGLISH_LOCALES = {"en", "en-us", "en-gb", "en-au", "en-ca"}
_KNOWN_NON_ENGLISH_LOCALES = {
    "ko", "ja", "zh", "zh-cn", "zh-tw", "fr", "de", "es", "pt", "pt-br",
    "it", "ru", "ar", "hi", "nl", "pl", "tr", "vi", "id", "th", "sv",
    "da", "fi", "no", "cs", "el", "he", "hu", "ro", "sk", "uk",
}
_LOCALE_PATH_SEGMENT_RE = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")


def is_english_url(url: str) -> bool:
    parsed = urlparse(url)

    query = parse_qs(parsed.query)
    for param_name in ("hl", "lang", "locale"):
        values = query.get(param_name)
        if values:
            value = values[0].strip().lower()
            if value in _KNOWN_NON_ENGLISH_LOCALES:
                return False
            if value in _ENGLISH_LOCALES:
                return True

    path_segments = [seg for seg in parsed.path.split("/") if seg]
    if path_segments:
        first_segment = path_segments[0].lower()
        if first_segment in _KNOWN_NON_ENGLISH_LOCALES:
            return False
        if first_segment in _ENGLISH_LOCALES:
            return True

    return True


@dataclass
class ScrapedItem:
    source_url: str
    question: Optional[str]
    text: str


def _extract_faq_jsonld(url: str, soup: BeautifulSoup) -> list["ScrapedItem"]:
    """Pre-built scraper: many sites mark their FAQ content up with
    schema.org FAQPage JSON-LD. When present, it gives clean, pre-paired
    Q&A data directly from the page's structured data — no selector
    guessing required.
    """
    items: list[ScrapedItem] = []

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        nodes = data if isinstance(data, list) else [data]
        flat_nodes = []
        for node in nodes:
            if isinstance(node, dict) and "@graph" in node and isinstance(node["@graph"], list):
                flat_nodes.extend(node["@graph"])
            else:
                flat_nodes.append(node)

        for node in flat_nodes:
            if not isinstance(node, dict):
                continue
            if node.get("@type") != "FAQPage":
                continue
            for entry in node.get("mainEntity", []):
                if not isinstance(entry, dict) or entry.get("@type") != "Question":
                    continue
                question_text = (entry.get("name") or "").strip()
                answer_obj = entry.get("acceptedAnswer") or {}
                answer_text = (answer_obj.get("text") or "").strip() if isinstance(answer_obj, dict) else ""
                if answer_text and "<" in answer_text:
                    answer_text = BeautifulSoup(answer_text, "lxml").get_text(separator=" ", strip=True)
                if question_text and answer_text:
                    items.append(ScrapedItem(url, question_text, answer_text))

    return items


def fetch_html(url: str) -> str:
    """Sync fetch with anti-bot retry/backoff + UA rotation."""
    import time

    last_exc: Exception | None = None
    for attempt in range(settings.SCRAPER_MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=_build_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < settings.SCRAPER_MAX_RETRIES:
                wait = float(response.headers.get("Retry-After", 0)) or (
                    settings.SCRAPER_RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                )
                logger.warning(f"[scraper] {response.status_code} on {url}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < settings.SCRAPER_MAX_RETRIES:
                wait = settings.SCRAPER_RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)
    raise last_exc


async def async_fetch_html(url: str, client: httpx.AsyncClient | None = None) -> str:
    """Async HTTP fetch. Rotates UA per attempt, retries on 403/429/5xx
    with exponential backoff + jitter, honors Retry-After."""
    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True)
        own_client = True

    last_exc: Exception | None = None
    try:
        for attempt in range(settings.SCRAPER_MAX_RETRIES + 1):
            try:
                resp = await client.get(url, headers=_build_headers(), follow_redirects=True)
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < settings.SCRAPER_MAX_RETRIES:
                    wait = float(resp.headers.get("Retry-After", 0)) or (
                        settings.SCRAPER_RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                    )
                    logger.warning(
                        f"[scraper] {resp.status_code} on {url}, retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/{settings.SCRAPER_MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.text
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < settings.SCRAPER_MAX_RETRIES:
                    wait = settings.SCRAPER_RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"[scraper] Fetch error on {url} ({exc}), retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
        raise last_exc
    finally:
        if own_client:
            await client.aclose()


async def async_fetch_html_js(url: str, timeout_ms: int | None = None) -> str:
    """Fetch a page's fully-rendered DOM using headless Chromium (Playwright).
    Requires `playwright install chromium` to have been run once."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install playwright` and "
            "`playwright install chromium` to enable JS rendering."
        ) from exc

    timeout_ms = timeout_ms or settings.SCRAPER_JS_RENDER_TIMEOUT_MS
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(user_agent=random.choice(_USER_AGENTS))
            await page.goto(url, timeout=timeout_ms, wait_until="networkidle")
            html = await page.content()
            return html
        finally:
            await browser.close()


def _visible_text_length(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return len(soup.get_text(strip=True))


async def smart_fetch_html(url: str, client: httpx.AsyncClient | None = None, force_js: bool = False) -> str:
    """Static fetch first; auto-escalates to JS rendering if content looks
    like an empty SPA shell, or immediately if force_js=True."""
    if force_js and settings.SCRAPER_JS_RENDER_ENABLED:
        return await async_fetch_html_js(url)

    html = await async_fetch_html(url, client=client)

    if settings.SCRAPER_JS_RENDER_ENABLED and _visible_text_length(html) < settings.SCRAPER_JS_FALLBACK_MIN_TEXT_LENGTH:
        logger.info(f"[scraper] Thin content on {url} ({_visible_text_length(html)} chars) — retrying with JS rendering")
        try:
            return await async_fetch_html_js(url)
        except Exception as exc:
            logger.warning(f"[scraper] JS render fallback failed for {url}: {exc}; using static HTML")
            return html

    return html


def scrape_faq_page(
    url: str,
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    fallback_content_selector: str = "body",
) -> list[ScrapedItem]:
    return asyncio.run(async_scrape_faq_page(url, question_selector, answer_selector, fallback_content_selector))


async def async_scrape_faq_page(
    url: str,
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    fallback_content_selector: str = "body",
    render_js: bool = False,
) -> list[ScrapedItem]:
    """
    Extraction priority:
      1. JSON-LD FAQPage (structured data)
      2. question_selector/answer_selector CSS pairing
      3. Fallback: dump fallback_content_selector's visible text
    """
    html = await smart_fetch_html(url, force_js=render_js)
    soup = BeautifulSoup(html, "lxml")

    items: list[ScrapedItem] = _extract_faq_jsonld(url, soup)

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    if not items and question_selector and answer_selector:
        questions = soup.select(question_selector)
        answers = soup.select(answer_selector)

        if questions and answers and len(questions) == len(answers):
            for q_tag, a_tag in zip(questions, answers):
                question_text = q_tag.get_text(strip=True)
                answer_text = a_tag.get_text(separator=" ", strip=True)
                if question_text and answer_text:
                    items.append(ScrapedItem(url, question_text, answer_text))

    if not items:
        container = soup.select_one(fallback_content_selector) or soup
        text = container.get_text(separator="\n", strip=True)
        if text:
            items.append(ScrapedItem(url, None, text))

    return items


def scrape_multiple_pages(
    urls: list[str],
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    render_js: bool = False,
) -> list[ScrapedItem]:
    return asyncio.run(async_scrape_multiple_pages(urls, question_selector, answer_selector, render_js=render_js))


async def async_scrape_multiple_pages(
    urls: list[str],
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    render_js: bool = False,
) -> list[ScrapedItem]:
    all_items: list[ScrapedItem] = []
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for url in urls:
            if not is_english_url(url):
                logger.info(f"[scraper] Skipping non-English URL: {url}")
                continue
            try:
                items = await async_scrape_faq_page(url, question_selector, answer_selector, render_js=render_js)
                all_items.extend(items)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning(f"[scraper] Failed to scrape {url}: {exc}")
            await asyncio.sleep(settings.SCRAPER_MIN_DELAY_SECONDS)  # politeness delay
    return all_items


def _same_domain(base_url: str, candidate_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(candidate_url).netloc


def _extract_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full_url = urljoin(base_url, href)
        full_url = full_url.split("#")[0]
        if not _same_domain(base_url, full_url):
            continue
        if not is_english_url(full_url):
            continue
        links.append(full_url)
    return links


def crawl_website(
    start_urls: list[str],
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    max_pages: int = 20,
    render_js: bool = False,
) -> list[ScrapedItem]:
    import time

    visited: set[str] = set()
    queue: deque[str] = deque(u for u in dict.fromkeys(start_urls) if is_english_url(u))
    queued: set[str] = set(queue)
    all_items: list[ScrapedItem] = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        queued.discard(url)
        if url in visited:
            continue
        visited.add(url)

        try:
            if render_js and settings.SCRAPER_JS_RENDER_ENABLED:
                html = asyncio.run(async_fetch_html_js(url))
            else:
                html = fetch_html(url)
                if settings.SCRAPER_JS_RENDER_ENABLED and _visible_text_length(html) < settings.SCRAPER_JS_FALLBACK_MIN_TEXT_LENGTH:
                    logger.info(f"[crawler] Thin content on {url} — retrying with JS rendering")
                    html = asyncio.run(async_fetch_html_js(url))
        except (requests.RequestException, RuntimeError) as exc:
            logger.warning(f"[crawler] Failed to fetch {url}: {exc}")
            continue

        soup = BeautifulSoup(html, "lxml")
        page_items: list[ScrapedItem] = _extract_faq_jsonld(url, soup)

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        if not page_items and question_selector and answer_selector:
            questions = soup.select(question_selector)
            answers = soup.select(answer_selector)
            if questions and answers and len(questions) == len(answers):
                for q_tag, a_tag in zip(questions, answers):
                    question_text = q_tag.get_text(strip=True)
                    answer_text = a_tag.get_text(separator=" ", strip=True)
                    if question_text and answer_text:
                        page_items.append(ScrapedItem(url, question_text, answer_text))

        if not page_items:
            text = soup.get_text(separator="\n", strip=True)
            if text:
                page_items.append(ScrapedItem(url, None, text))

        all_items.extend(page_items)
        logger.info(f"[crawler] Scraped ({len(visited)}/{max_pages}): {url}")
        time.sleep(settings.SCRAPER_MIN_DELAY_SECONDS)  # politeness delay

        try:
            new_links = _extract_links(url, html)
        except Exception as exc:
            logger.warning(f"[crawler] Failed to extract links from {url}: {exc}")
            new_links = []

        for link in new_links:
            if link not in visited and link not in queued:
                queue.append(link)
                queued.add(link)

    return all_items


async def async_crawl_website(
    start_urls: list[str],
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    max_pages: int = 20,
    render_js: bool = False,
) -> list[ScrapedItem]:
    visited: set[str] = set()
    queue: deque[str] = deque(u for u in dict.fromkeys(start_urls) if is_english_url(u))
    queued: set[str] = set(queue)
    all_items: list[ScrapedItem] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            queued.discard(url)
            if url in visited:
                continue
            visited.add(url)

            try:
                html = await smart_fetch_html(url, client=client, force_js=render_js)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning(f"[crawler] Failed to fetch {url}: {exc}")
                continue

            soup = BeautifulSoup(html, "lxml")
            page_items: list[ScrapedItem] = _extract_faq_jsonld(url, soup)

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            if not page_items and question_selector and answer_selector:
                questions = soup.select(question_selector)
                answers = soup.select(answer_selector)
                if questions and answers and len(questions) == len(answers):
                    for q_tag, a_tag in zip(questions, answers):
                        question_text = q_tag.get_text(strip=True)
                        answer_text = a_tag.get_text(separator=" ", strip=True)
                        if question_text and answer_text:
                            page_items.append(ScrapedItem(url, question_text, answer_text))

            if not page_items:
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    page_items.append(ScrapedItem(url, None, text))

            all_items.extend(page_items)
            logger.info(f"[crawler] Scraped ({len(visited)}/{max_pages}): {url}")
            await asyncio.sleep(settings.SCRAPER_MIN_DELAY_SECONDS)  # politeness delay

            try:
                new_links = _extract_links(url, html)
            except Exception as exc:
                logger.warning(f"[crawler] Failed to extract links from {url}: {exc}")
                new_links = []

            for link in new_links:
                if link not in visited and link not in queued:
                    queue.append(link)
                    queued.add(link)

    return all_items