"""Checks on the published playground page: search metadata, static numbers.

The numbers in playground/index.html are written into the HTML so search
engines and readers without JavaScript see them. app.js re-reads them from
data/public.js; this test makes sure the two agree, so a rebuilt data file
cannot leave a stale figure in the HTML.
"""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

PLAYGROUND = Path(__file__).resolve().parent.parent / "playground"


class _Page(HTMLParser):
    """Collects meta tags, links, JSON-LD blocks and data-bind values."""

    def __init__(self):
        super().__init__()
        self.meta, self.links, self.ld, self.binds = {}, {}, [], []
        self.title, self._in, self._bind = "", None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            self.meta[a.get("name") or a.get("property")] = a.get("content", "")
        elif tag == "link":
            self.links[a.get("rel")] = a.get("href")
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._in = "ld"
            self.ld.append("")
        elif tag == "title":
            self._in = "title"
        if "data-bind" in a:
            self._bind = [a["data-bind"], a.get("data-fmt"), ""]

    def handle_endtag(self, tag):
        if tag in ("script", "title"):
            self._in = None
        if self._bind is not None:
            self.binds.append(tuple(self._bind))
            self._bind = None

    def handle_data(self, data):
        if self._in == "ld":
            self.ld[-1] += data
        elif self._in == "title":
            self.title += data
        if self._bind is not None:
            self._bind[2] += data


def _page():
    page = _Page()
    page.feed((PLAYGROUND / "index.html").read_text(encoding="utf-8"))
    return page


def _public_data():
    text = (PLAYGROUND / "data" / "public.js").read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].strip().rstrip(";"))


def test_page_has_search_and_sharing_metadata():
    page = _page()
    assert 30 <= len(page.title) <= 70
    assert 70 <= len(page.meta["description"]) <= 160
    assert page.links["canonical"] == "https://playground.vahinitech.com/"
    for key in ("og:title", "og:description", "og:image", "og:url", "twitter:card"):
        assert page.meta.get(key), key
    assert (PLAYGROUND / "og.png").exists()
    assert page.links["icon"].startswith("/site/assets/favicon")
    robots = (PLAYGROUND / "robots.txt").read_text(encoding="utf-8")
    assert "Sitemap: https://playground.vahinitech.com/sitemap.xml" in robots
    assert "<loc>https://playground.vahinitech.com/</loc>" in (
        PLAYGROUND / "sitemap.xml"
    ).read_text(encoding="utf-8")


def test_structured_data_is_valid_json():
    blocks = _page().ld
    assert len(blocks) == 1
    data = json.loads(blocks[0])
    assert data["@type"] == "WebApplication"
    assert data["url"] == "https://playground.vahinitech.com/"


def test_static_numbers_match_the_data():
    data = _public_data()
    binds = _page().binds
    assert binds, "no data-bind elements found"
    for path, fmt, shown in binds:
        value = data
        for key in path.split("."):
            value = value[key]
        expected = f"{value:.2f}%" if fmt == "pct" else str(value)
        assert shown.strip() == expected, (path, shown, expected)


def test_page_loads_nothing_from_other_hosts():
    """The served CSP allows only the page's own files."""
    html = (PLAYGROUND / "index.html").read_text(encoding="utf-8")
    for match in re.finditer(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html):
        url = match.group(1)
        if 'rel="canonical"' in match.group(0):
            continue
        assert not url.startswith(("http:", "https:", "//")), url


def test_styles_use_only_design_system_tokens():
    """Colours come from the Vahini design system, never from this repo."""
    css = (PLAYGROUND / "style.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert not re.findall(
        r"#[0-9a-fA-F]{3,8}\b|rgba?\(", css
    ), "colour literal in style.css"
    assert not re.search(r"var\(--v-[a-z0-9-]+,", css), "a design token has a fallback"
    for script in ("app.js", "charts.js", "stages.js", "write.js"):
        js = (PLAYGROUND / script).read_text(encoding="utf-8")
        assert not re.findall(
            r"[\"']#[0-9a-fA-F]{6}[\"']", js
        ), f"colour literal in {script}"
    html = (PLAYGROUND / "index.html").read_text(encoding="utf-8")
    assert html.index('href="/site/design/v1/vahini.css"') < html.index(
        'href="style.css"'
    )


def test_no_inline_style_attributes():
    """The served CSP is style-src 'self': inline style attributes are dropped."""
    html = (PLAYGROUND / "index.html").read_text(encoding="utf-8")
    assert ' style="' not in html
    for script in ("app.js", "charts.js", "stages.js", "write.js"):
        js = (PLAYGROUND / script).read_text(encoding="utf-8")
        assert not re.search(
            r"style=[\\\"']", js
        ), f"inline style attribute built in {script}"
