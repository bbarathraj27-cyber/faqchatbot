"""
Lightweight, URL-based language detection for sitemap filtering.
"""

from urllib.parse import urlparse, parse_qs

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "es-419": "Spanish (Latin America)",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "pt-br": "Portuguese (Brazil)",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "hu": "Hungarian",
    "ro": "Romanian",
    "el": "Greek",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "fa": "Persian",
    "ms": "Malay",
    "sw": "Swahili",
}

_KNOWN_CODES = set(LANGUAGE_NAMES.keys())
_LOCALE_QUERY_KEYS = ("hl", "lang", "language", "locale")


def _normalize(code: str) -> str:
    return code.strip().lower()


def detect_language(url: str) -> str:
    parsed = urlparse(url)

    query = parse_qs(parsed.query)
    for key in _LOCALE_QUERY_KEYS:
        values = query.get(key)
        if values:
            candidate = _normalize(values[0])
            if candidate in _KNOWN_CODES:
                return candidate

    segments = [s for s in parsed.path.split("/") if s]
    if segments:
        first = _normalize(segments[0])
        if first in _KNOWN_CODES:
            return first

    host_parts = parsed.netloc.split(".")
    if len(host_parts) > 2:
        sub = _normalize(host_parts[0])
        if sub in _KNOWN_CODES:
            return sub

    return "en"


def language_display_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code.upper())


def group_urls_by_language(urls: list[str]) -> list[dict]:
    buckets: dict[str, int] = {}
    for url in urls:
        code = detect_language(url)
        buckets[code] = buckets.get(code, 0) + 1

    items = [
        {"code": code, "name": language_display_name(code), "count": count}
        for code, count in buckets.items()
    ]
    items.sort(key=lambda item: (item["code"] != "en", -item["count"]))
    return items


def filter_urls_by_language(urls: list[str], languages: list[str]) -> list[str]:
    wanted = {_normalize(code) for code in languages}
    return [url for url in urls if detect_language(url) in wanted]