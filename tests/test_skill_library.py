"""The 15-skill library: catalog rules and each new skill, with no network or real desktop."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from stronghouse.core.calculator import CalculationError, calculate
from stronghouse.core.skill_catalog import SKILLS, SkillCategory
from stronghouse.core.text_extract import html_to_text, parse_feed, parse_search_results
from stronghouse.services.skills import SkillError, SkillExecutor
from stronghouse.services.skills.utility import current_datetime

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>News</title>
<item><title>Storm hits coast</title><link>https://news.test/storm</link>
<pubDate>Mon, 23 Nov 2026 08:00:00 GMT</pubDate>
<description>&lt;p&gt;Heavy &lt;b&gt;rain&lt;/b&gt;&lt;/p&gt;</description></item>
<item><title>Markets calm</title><link>https://news.test/markets</link></item>
</channel></rss>"""

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom"><title>Blog</title>
<entry><title>Release 2.0</title><link href="https://blog.test/2"/>
<updated>2026-11-20T10:00:00Z</updated><summary>New things</summary></entry></feed>"""

PAGE = """<html><head><title>Example page</title><style>p{color:red}</style></head>
<body><nav>Menu Home About</nav><h1>Welcome</h1><p>First   paragraph.</p>
<script>alert(1)</script><p>Second &amp; last.</p><footer>Copyright</footer></body></html>"""

SEARCH_PAGE = """<div class="result"><a class="result__a"
href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.theguardian.com%2Fuk&amp;rut=x">The Guardian UK</a>
<a class="result__snippet" href="#">Latest UK news and comment.</a></div>
<div class="result"><a class="result__a" href="https://example.org/">Example</a></div>"""


# ----- catalog -----


def test_library_has_fifteen_skills_across_all_categories():
    assert len(SKILLS) == 15
    assert {spec.category for spec in SKILLS.values()} == set(SkillCategory)


def test_only_skills_that_touch_nothing_outside_the_app_skip_approval():
    no_approval = {name for name, spec in SKILLS.items() if not spec.needs_approval}
    assert no_approval == {"take_note", "calculator", "current_datetime"}


def test_every_skill_has_a_handler():
    SkillExecutor(lambda *_: None)  # raises if a catalog entry has no code


# ----- pure helpers -----


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3 * 4", 14), ("(17.5 * 12) / 3", 70.0), ("2^10", 1024), ("sqrt(16)", 4.0),
     ("-5 % 3", 1), ("round(pi, 2)", 3.14), ("6 × 7", 42)],  # noqa: RUF001
)  # fmt: skip
def test_calculator(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression", ["__import__('os')", "open('x')", "1/0", "9**9**9", "a + 1", ""]
)
def test_calculator_refuses_anything_but_maths(expression):
    with pytest.raises(CalculationError):
        calculate(expression)


def test_html_to_text_keeps_content_and_drops_menus_and_scripts():
    text = html_to_text(PAGE)
    assert text.startswith("Example page")
    assert "First paragraph." in text and "Second & last." in text
    assert "alert" not in text and "Menu" not in text and "color" not in text


def test_parse_rss_and_atom():
    rss = parse_feed(RSS)
    assert [item.title for item in rss] == ["Storm hits coast", "Markets calm"]
    assert rss[0].summary == "Heavy rain"
    atom = parse_feed(ATOM)
    assert atom[0].link == "https://blog.test/2"
    with pytest.raises(ValueError):
        parse_feed("<html><body>not a feed</body></html>")


def test_search_results_unwrap_duckduckgo_redirects():
    results = parse_search_results(SEARCH_PAGE)
    assert results[0].url == "https://www.theguardian.com/uk"
    assert results[0].snippet == "Latest UK news and comment."
    assert results[1].url == "https://example.org/"


# ----- skills through the executor -----


def fake_web(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/rss":
        return httpx.Response(200, text=RSS, headers={"content-type": "application/rss+xml"})
    if request.url.host == "html.duckduckgo.com":
        assert b"q=guardian" in request.content
        return httpx.Response(200, text=SEARCH_PAGE, headers={"content-type": "text/html"})
    if request.url.path == "/page":
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})
    return httpx.Response(404)


@pytest.fixture
def executor():
    calls: list[tuple] = []

    def fake_program(argv, input_text=None):
        calls.append((argv, input_text))
        if argv[0] == "pdftotext":
            return "Rule 01: Write for the next reader\n"
        return ""

    http = httpx.Client(transport=httpx.MockTransport(fake_web))
    skills = SkillExecutor(lambda *_: None, http=http, run=fake_program)
    skills.program_calls = calls
    return skills


def test_web_skills(executor):
    feed = executor.execute(1, "read_news_feed", {"url": "https://news.test/rss"})
    assert "1. Storm hits coast" in feed and "https://news.test/storm" in feed

    page = executor.execute(1, "fetch_web_page", {"url": "https://site.test/page"})
    assert "First paragraph." in page

    feed_via_page = executor.execute(1, "fetch_web_page", {"url": "https://news.test/rss"})
    assert "Storm hits coast" in feed_via_page  # feeds are recognised either way

    results = executor.execute(1, "web_search", {"query": "guardian"})
    assert "https://www.theguardian.com/uk" in results


def test_web_skills_refuse_non_web_addresses_and_report_errors(executor):
    with pytest.raises(SkillError, match="http and https"):
        executor.execute(1, "fetch_web_page", {"url": "file:///etc/passwd"})
    with pytest.raises(SkillError, match="Could not fetch"):
        executor.execute(1, "fetch_web_page", {"url": "https://site.test/missing"})


def test_read_pdf_uses_pdftotext(executor, tmp_path: Path):
    pdf = tmp_path / "rules.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert "Rule 01" in executor.execute(1, "read_pdf", {"path": str(pdf)})
    assert executor.program_calls[-1][0][0] == "pdftotext"


def test_read_file_points_to_read_pdf_for_pdfs(executor, tmp_path: Path):
    pdf = tmp_path / "rules.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(SkillError, match="Read PDF"):
        executor.execute(1, "read_file", {"path": str(pdf)})


def test_missing_pdf_tool_gives_install_hint(tmp_path: Path):
    def missing(argv, input_text=None):
        raise SkillError(f"'{argv[0]}' is not installed.")

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(SkillError, match="pacman -S poppler"):
        SkillExecutor(lambda *_: None, run=missing).execute(1, "read_pdf", {"path": str(pdf)})


def test_desktop_skills(executor, tmp_path: Path):
    assert executor.execute(1, "copy_to_clipboard", {"text": "hello"}).startswith("Copied 5")
    assert executor.program_calls[-1] == (["wl-copy"], "hello")

    executor.execute(1, "open_with_app", {"target": "https://www.theguardian.com/uk"})
    assert executor.program_calls[-1][0] == ["xdg-open", "https://www.theguardian.com/uk"]
    with pytest.raises(SkillError, match="does not exist"):
        executor.execute(1, "open_with_app", {"target": str(tmp_path / "nope.txt")})


def test_move_file_never_overwrites(executor, tmp_path: Path):
    source = tmp_path / "a.txt"
    source.write_text("A")
    taken = tmp_path / "b.txt"
    taken.write_text("B")

    with pytest.raises(SkillError, match="already exists"):
        executor.execute(1, "move_file", {"source": str(source), "destination": str(taken)})
    assert taken.read_text() == "B"

    folder = tmp_path / "sorted"
    folder.mkdir()
    executor.execute(1, "move_file", {"source": str(source), "destination": str(folder)})
    assert (folder / "a.txt").read_text() == "A" and not source.exists()


def test_utility_skills(executor):
    assert executor.execute(1, "calculator", {"expression": "12 * 12"}) == "12 * 12 = 144"
    with pytest.raises(SkillError):
        executor.execute(1, "calculator", {"expression": "import os"})
    fixed = datetime(2026, 11, 23, 18, 30, tzinfo=UTC)
    assert current_datetime({}, now=lambda: fixed).startswith("Monday 23 November 2026, 18:30")


LITE_PAGE = """<table><tr><td><a rel="nofollow" href="https://www.theguardian.com/uk"
class="result-link">The Guardian</a></td></tr>
<tr><td class="result-snippet">Latest UK news</td></tr></table>"""


def test_lite_search_page_is_parsed():
    results = parse_search_results(LITE_PAGE)
    assert results[0].url == "https://www.theguardian.com/uk"
    assert results[0].snippet == "Latest UK news"


def blocking_web(request: httpx.Request) -> httpx.Response:
    if request.url.host == "html.duckduckgo.com":
        return httpx.Response(202, text="<html>anomaly</html>")  # bot check, no results
    if request.url.host == "lite.duckduckgo.com":
        return httpx.Response(200, text=LITE_PAGE)
    return httpx.Response(403)


def test_blocked_sites_explain_what_to_try_and_search_falls_back():
    http = httpx.Client(transport=httpx.MockTransport(blocking_web))
    skills = SkillExecutor(lambda *_: None, http=http)
    with pytest.raises(SkillError, match=r"site\.test refused the request \(403\).*RSS"):
        skills.execute(1, "fetch_web_page", {"url": "https://site.test/page"})
    assert "theguardian.com/uk" in skills.execute(1, "web_search", {"query": "guardian"})


def test_search_failure_names_the_reason():
    http = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(403)))
    skills = SkillExecutor(lambda *_: None, http=http)
    with pytest.raises(SkillError, match="answered 403"):
        skills.execute(1, "web_search", {"query": "guardian"})
