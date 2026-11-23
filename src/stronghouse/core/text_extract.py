"""Turn web pages, news feeds and search result pages into short plain text.

Pure functions over strings; fetching happens in services/skills/web.py.
Standard library only (html.parser, xml.etree) to keep dependencies few.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

MAX_FEED_ITEMS = 20
MAX_SEARCH_RESULTS = 8
_SUMMARY_CHARS = 200

_SKIPPED_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
    "article", "blockquote", "pre", "dt", "dd",
}  # fmt: skip


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Readable text of a web page: title first, then the body without menus or scripts."""
    collector = _TextCollector()
    collector.feed(html)
    collector.close()
    body = "".join(collector.parts)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in body.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    title = collector.title.strip()
    return f"{title}\n\n{text}" if title else text


@dataclass(frozen=True)
class FeedItem:
    title: str
    link: str
    published: str
    summary: str


def parse_feed(xml_text: str) -> list[FeedItem]:
    """Read RSS 2.0 or Atom. Raises ValueError if it is neither."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as error:
        raise ValueError(f"Not a valid news feed: {error}") from error

    items = [_rss_item(node) for node in root.iter("item")]
    if not items:
        items = [_atom_entry(node) for node in root.iter("{http://www.w3.org/2005/Atom}entry")]
    if not items and root.tag not in ("rss", "{http://www.w3.org/2005/Atom}feed"):
        raise ValueError("This address is not an RSS or Atom news feed.")
    return items[:MAX_FEED_ITEMS]


def format_feed(items: list[FeedItem]) -> str:
    if not items:
        return "The feed has no items."
    lines = []
    for number, item in enumerate(items, start=1):
        lines.append(f"{number}. {item.title}")
        if item.published:
            lines.append(f"   {item.published}")
        if item.summary:
            lines.append(f"   {item.summary}")
        if item.link:
            lines.append(f"   {item.link}")
    return "\n".join(lines)


def _rss_item(node: ElementTree.Element) -> FeedItem:
    return FeedItem(
        title=_text(node.find("title")),
        link=_text(node.find("link")),
        published=_text(node.find("pubDate")),
        summary=_short(html_to_text(_text(node.find("description")))),
    )


def _atom_entry(node: ElementTree.Element) -> FeedItem:
    atom = "{http://www.w3.org/2005/Atom}"
    link = node.find(f"{atom}link")
    summary = node.find(f"{atom}summary")
    if summary is None:
        summary = node.find(f"{atom}content")
    return FeedItem(
        title=_text(node.find(f"{atom}title")),
        link=link.get("href", "") if link is not None else "",
        published=_text(node.find(f"{atom}updated")) or _text(node.find(f"{atom}published")),
        summary=_short(html_to_text(_text(summary))),
    )


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class _DuckDuckGoParser(HTMLParser):
    """Reads results from DuckDuckGo's no-JavaScript pages (html. and lite.duckduckgo.com).

    That page's markup is not a stable API; if DuckDuckGo changes it, this
    parser returns no results instead of crashing (see ADR 0006).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self.results.append({"title": "", "url": dict(attrs).get("href", ""), "snippet": ""})
            self._field = "title"
        elif ("result__snippet" in classes or "result-snippet" in classes) and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "td", "div"):
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field and self.results:
            self.results[-1][self._field] += data


def parse_search_results(html: str) -> list[SearchResult]:
    parser = _DuckDuckGoParser()
    parser.feed(html)
    results = []
    for raw in parser.results:
        url = _unwrap_redirect(raw["url"])
        if raw["title"].strip() and url.startswith("http"):
            results.append(SearchResult(raw["title"].strip(), url, _short(raw["snippet"])))
    return results[:MAX_SEARCH_RESULTS]


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    return "\n".join(
        f"{n}. {r.title}\n   {r.url}\n   {r.snippet}".rstrip()
        for n, r in enumerate(results, start=1)
    )


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<real url>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SUMMARY_CHARS else text[: _SUMMARY_CHARS - 1] + "…"
