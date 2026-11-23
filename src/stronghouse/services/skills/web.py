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
# DuckDuckGo's two no-JavaScript pages. The second is tried when the first refuses
# or answers with a "too many requests" page instead of results.
SEARCH_URLS = ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/")
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
        problem = "no results"
        for url in SEARCH_URLS:
            try:
                response = self._http.post(url, data={"q": query})
            except httpx.HTTPError as error:
                problem = str(error)
                continue
            if response.status_code != 200:
                problem = f"the search site answered {response.status_code}"
                continue
            results = parse_search_results(response.text)
            if results:
                return format_search_results(results)
        raise SkillError(
            f"The web search failed ({problem}). If you know the site, fetch its page "
            "or its RSS feed directly instead."
        )

    def _get(self, url: str) -> tuple[str, str]:
        try:
            with self._http.stream("GET", url) as response:
                if response.status_code in (401, 403, 429):
                    raise SkillError(_blocked_message(url, response.status_code))
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


def _blocked_message(url: str, status: int) -> str:
    """Tell the model why, so it can try another route instead of giving up."""
    return (
        f"{urlparse(url).netloc} refused the request ({status}). Many sites block "
        "automated readers. Try the site's RSS feed with read_news_feed (for example "
        "<site>/rss), or use web_search to find another source."
    )


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
