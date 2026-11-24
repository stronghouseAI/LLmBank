"""The web UI's scripts and its page must agree: a renamed id should fail here, not in the browser.

These tests read the static files only, so they need no browser and no JavaScript runtime.
"""

import re
from importlib import resources
from pathlib import Path

STATIC = Path(str(resources.files("stronghouse.ui") / "static"))
SCRIPTS = ["app.js", "guild.js", "jobs.js"]
# Ids the scripts create themselves inside rendered HTML, not present in index.html.
CREATED_AT_RUNTIME: set[str] = set()


def _page_ids() -> set[str]:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return set(re.findall(r'\bid="([^"]+)"', html))


def _script_ids(name: str) -> set[str]:
    script = (STATIC / name).read_text(encoding="utf-8")
    # $("#x"), $("#x .child"), querySelector("#x input"): the first #id in each selector.
    selectors = re.findall(r'(?:\$|querySelector(?:All)?)\(\s*["\'`]([^"\'`]+)["\'`]', script)
    return {match for sel in selectors for match in re.findall(r"#([\w-]+)", sel)}


def test_every_element_a_script_uses_exists_in_the_page():
    page = _page_ids() | CREATED_AT_RUNTIME
    for name in SCRIPTS:
        missing = _script_ids(name) - page
        assert not missing, f"{name} uses ids missing from index.html: {sorted(missing)}"


def test_page_ids_are_unique():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    ids = re.findall(r'\bid="([^"]+)"', html)
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates


def test_page_loads_every_script_and_nothing_from_the_internet():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in SCRIPTS:
        assert f'src="{name}"' in html
    # The app must work offline: no CDNs, web fonts or remote images.
    assert not re.search(r'(?:src|href)="(?:https?:)?//', html)
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "url(http" not in css and "@import" not in css


def test_colours_are_defined_only_as_theme_tokens():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    outside_tokens = re.sub(r":root[^{]*\{[^}]*\}", "", css)
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", outside_tokens)
