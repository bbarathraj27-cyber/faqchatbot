import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs

import requests
import httpx
import asyncio
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FAQChatbotScraper/1.0; +https://example.com/bot-info)"
}
REQUEST_TIMEOUT_SECONDS = 15

# Common locale codes seen either as a `?hl=` / `?lang=` query param or as a
# leading path segment (e.g. tensorflow.org/ko/..., site.com/zh-cn/...).
# Anything matching one of these that isn't an English variant gets skipped
# before it's ever queued or fetched — this is what was pulling in the
# ?hl=ko / ?hl=fr / ?hl=zh-CN / ?hl=pt-BR duplicate pages you saw in RAG
# results.
_ENGLISH_LOCALES = {"en", "en-us", "en-gb", "en-au", "en-ca"}
_KNOWN_NON_ENGLISH_LOCALES = {
    "ko", "ja", "zh", "zh-cn", "zh-tw", "fr", "de", "es", "pt", "pt-br",
    "it", "ru", "ar", "hi", "nl", "pl", "tr", "vi", "id", "th", "sv",
    "da", "fi", "no", "cs", "el", "he", "hu", "ro", "sk", "uk",
}
_LOCALE_PATH_SEGMENT_RE = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")


def is_english_url(url: str) -> bool:
    """
    Returns False for URLs that clearly point at a non-English locale
    variant of a page, based on:
      1. a `hl` or `lang` query parameter set to a known non-English code
      2. a leading path segment that's a known non-English locale code
         (e.g. /ko/, /zh-cn/, /fr/...)
    Defaults to True (assume English) when neither signal is present, so
    this never blocks normal same-language site navigation.
    """
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


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


async def async_fetch_html(url: str, client: httpx.AsyncClient | None = None) -> str:
    """Async HTTP fetch using `httpx.AsyncClient`. Caller may pass a shared
    `client` for connection reuse; otherwise a temporary client is created."""
    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers=HEADERS, follow_redirects=True)
        own_client = True
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    finally:
        if own_client:
            await client.aclose()


def scrape_faq_page(
    url: str,
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    fallback_content_selector: str = "body",
) -> list[ScrapedItem]:
    """Synchronous wrapper that keeps the original API but delegates to
    the async implementation when possible (useful for gradual migration)."""
    # Run the async implementation in a new event loop. This is safe when
    # called from synchronous contexts (e.g. thread-pool workers).
    return asyncio.run(async_scrape_faq_page(url, question_selector, answer_selector, fallback_content_selector))


async def async_scrape_faq_page(
    url: str,
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
    fallback_content_selector: str = "body",
) -> list[ScrapedItem]:
    html = await async_fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    items: list[ScrapedItem] = []

    if question_selector and answer_selector:
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
) -> list[ScrapedItem]:
    """Sync wrapper kept for compatibility; delegates to the async
    implementation via `asyncio.run` so existing callers continue to work."""
    return asyncio.run(async_scrape_multiple_pages(urls, question_selector, answer_selector))


async def async_scrape_multiple_pages(
    urls: list[str],
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
) -> list[ScrapedItem]:
    all_items: list[ScrapedItem] = []
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers=HEADERS) as client:
        for url in urls:
            if not is_english_url(url):
                logger.info(f"[scraper] Skipping non-English URL: {url}")
                continue
            try:
                items = await async_scrape_faq_page(url, question_selector, answer_selector)
                all_items.extend(items)
            except httpx.HTTPError as exc:
                logger.warning(f"[scraper] Failed to scrape {url}: {exc}")
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
) -> list[ScrapedItem]:
    visited: set[str] = set()
    # Filter non-English URLs out of the seed list too, so a bad start URL
    # can't get scraped even before link-discovery filtering kicks in.
    # deque instead of list: popleft() is O(1) vs pop(0) being O(n), which
    # matters once the queue grows into the hundreds/thousands of links.
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
            html = fetch_html(url)
        except requests.RequestException as exc:
            logger.warning(f"[crawler] Failed to fetch {url}: {exc}")
            continue

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        page_items: list[ScrapedItem] = []
        if question_selector and answer_selector:
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

        try:
            # _extract_links already filters out non-English URLs internally.
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
) -> list[ScrapedItem]:
    """Async crawler: uses the async fetcher but keeps parsing synchronous
    (BeautifulSoup is CPU-bound, which is fine). This implementation is a
    straightforward async port of `crawl_website` and conservatively uses a
    single `AsyncClient` for connection reuse."""
    visited: set[str] = set()
    queue: deque[str] = deque(u for u in dict.fromkeys(start_urls) if is_english_url(u))
    queued: set[str] = set(queue)
    all_items: list[ScrapedItem] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers=HEADERS) as client:
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            queued.discard(url)
            if url in visited:
                continue
            visited.add(url)

            try:
                html = await async_fetch_html(url, client=client)
            except httpx.HTTPError as exc:
                logger.warning(f"[crawler] Failed to fetch {url}: {exc}")
                continue

            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            page_items: list[ScrapedItem] = []
            if question_selector and answer_selector:
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