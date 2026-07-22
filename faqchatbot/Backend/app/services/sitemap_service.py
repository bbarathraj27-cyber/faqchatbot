import random
import httpx
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from app.services.neo4j_service import extract_topics

WORDS_PER_CHUNK_ESTIMATE = 225
BYTES_PER_CHUNK_ESTIMATE = 10 * 1024


def _fetch_sitemap_urls(sitemap_url: str, seen: set[str] | None = None) -> list[str]:
    if seen is None:
        seen = set()
    if sitemap_url in seen:
        return []
    seen.add(sitemap_url)

    resp = httpx.get(sitemap_url, timeout=15, follow_redirects=True)
    resp.raise_for_status()

    root = ElementTree.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[str] = []

    sub_sitemaps = root.findall("sm:sitemap/sm:loc", ns)
    if sub_sitemaps:
        for loc in sub_sitemaps:
            urls.extend(_fetch_sitemap_urls(loc.text.strip(), seen))
        return urls

    for loc in root.findall("sm:url/sm:loc", ns):
        urls.append(loc.text.strip())
    return urls


def analyze_site(base_url: str) -> dict:
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_url = urljoin(domain, "/sitemap.xml")

    try:
        all_urls = _fetch_sitemap_urls(sitemap_url)
    except Exception as e:
        return {
            "domain": domain, "sitemap_found": False, "error": str(e),
            "total_pages": 0, "sample_urls": [], "all_urls": [],
        }

    same_domain_urls = sorted(set(u for u in all_urls if urlparse(u).netloc == parsed.netloc))
    return {
        "domain": domain, "sitemap_found": True,
        "total_pages": len(same_domain_urls),
        "sample_urls": same_domain_urls[:20],
        "all_urls": same_domain_urls,
    }


def _fetch_page_text(url: str) -> str:
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def analyze_site_deep(base_url: str, sample_size: int = 25) -> dict:
    basic = analyze_site(base_url)
    if not basic["sitemap_found"]:
        return basic

    all_urls = basic["all_urls"]
    total_pages = len(all_urls)
    sample = random.sample(all_urls, min(sample_size, total_pages))

    total_words_in_sample = 0
    pages_sampled = 0
    pages_failed = 0
    page_previews = []

    for url in sample:
        text = _fetch_page_text(url)
        if not text:
            pages_failed += 1
            continue
        word_count = len(text.split())
        total_words_in_sample += word_count
        pages_sampled += 1
        topics = extract_topics(text)
        page_previews.append({"url": url, "word_count": word_count, "topics": topics})

    if pages_sampled == 0:
        return {"domain": basic["domain"], "sitemap_found": True,
                "total_pages_in_sitemap": total_pages, "error": "Could not read any sampled pages"}

    avg_words_per_page = total_words_in_sample / pages_sampled
    estimated_total_words = round(avg_words_per_page * total_pages)
    estimated_total_chunks = round(estimated_total_words / WORDS_PER_CHUNK_ESTIMATE)
    estimated_storage_kb = round((estimated_total_chunks * BYTES_PER_CHUNK_ESTIMATE) / 1024)

    return {
        "domain": basic["domain"], "sitemap_found": True,
        "total_pages_in_sitemap": total_pages,
        "pages_sampled": pages_sampled, "pages_failed": pages_failed,
        "avg_words_per_page": round(avg_words_per_page),
        "total_word_count": estimated_total_words,
        "estimated_total_chunks": estimated_total_chunks,
        "estimated_storage_kb": estimated_storage_kb,
        "estimated_storage_mb": round(estimated_storage_kb / 1024, 2),
        "is_estimate": True,
        "page_previews": page_previews,
    }