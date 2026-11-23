"""Web skills: fetch a page, read a news feed, search the web. All need approval.

These are the only skills that use the internet, and only when the user has
equipped them on an agent and approved the exact address or search.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from stronghouse.core.text_extract import (
    format_feed,
    format_search_results,
    html_to_text,
    parse_feed,
    parse_search_results,
)
from stronghouse.services.skills.common import SkillError

MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024
SEARCH_URL = "https://html.duckduckgo.com/html/"
_FEED_TYPES = ("rss", "atom", "xml")


class WebSkills:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def fetch_web_page(self, args: dict) -> str:
        url = _checked_url(args["url"])
        body, content_type = self._get(url)
        if any(kind in content_type for kind in _FEED_TYPES):
            return format_feed(_parse_feed(body))
        if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            return html_to_text(body)
        return body

    def read_news_feed(self, args: dict) -> str:
        url = _checked_url(args["url"])
        body, _ = self._get(url)
        return format_feed(_parse_feed(body))

    def web_search(self, args: dict) -> str:
        query = str(args["query"]).strip()
        if not query:
            raise SkillError("The search is empty.")
        try:
            response = self._http.post(SEARCH_URL, data={"q": query})
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise SkillError(f"The web search failed: {error}") from error
        return format_search_results(parse_search_results(response.text))

    def _get(self, url: str) -> tuple[str, str]:
        try:
            with self._http.stream("GET", url) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        break
                content_type = response.headers.get("content-type", "").lower()
                encoding = response.encoding or "utf-8"
        except httpx.HTTPError as error:
            raise SkillError(f"Could not fetch {url}: {error}") from error
        return b"".join(chunks).decode(encoding, errors="replace"), content_type


def _checked_url(raw: object) -> str:
    url = str(raw).strip()
    if urlparse(url).scheme not in ("http", "https"):
        raise SkillError("Only http and https addresses can be fetched.")
    return url


def _parse_feed(body: str):
    try:
        return parse_feed(body)
    except ValueError as error:
        raise SkillError(str(error)) from error
