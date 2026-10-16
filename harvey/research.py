"""Prospect research — real, citable material for one cold email.

Used by the Writer when ``channels.email.deep_research`` is on. Reads a few
pages of the prospect's own website (about, services, work, blog, careers)
and, with a Serper key, the company's recent news and the person's public
footprint. Everything returned is raw source text with its URL, so the
writer can only claim what a source actually says. All of it is untrusted
web content: it goes into a tool-less brain call as data, never as
instructions.
"""

import logging
import re
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("harvey.research")

PAGE_HINTS = ("about", "service", "solution", "product", "work", "case", "client",
              "portfolio", "blog", "news", "press", "insight", "career", "job", "team")
MAX_PAGES = 5            # homepage + 4 inner pages
PAGE_CHARS = 2500        # per page, after boilerplate is stripped
MAX_BYTES = 2_000_000    # skip anything bigger than a normal web page
SEARCH_RESULTS = 5
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def _page_text(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href", "") for a in soup.find_all("a", href=True)]
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:PAGE_CHARS], links


async def _fetch(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        resp = await client.get(url)
    except Exception as e:
        logger.debug(f"Research fetch failed ({url[:80]}): {e}")
        return None
    if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
        return None
    if len(resp.content) > MAX_BYTES:
        return None
    return resp.text


async def read_website(website: str) -> list[tuple[str, str]]:
    """[(url, text)] for the homepage and the most useful inner pages."""
    if not website:
        return []
    if not website.startswith(("http://", "https://")):
        website = f"https://{website}"
    host = urlparse(website).netloc.lower().removeprefix("www.")
    pages: list[tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers=HEADERS) as client:
        html = await _fetch(client, website)
        if not html:
            return []
        text, links = _page_text(html)
        pages.append((website, text))

        seen, picks = {website.rstrip("/")}, []
        for href in links:
            url = urljoin(website, href).split("#")[0].rstrip("/")
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or url in seen:
                continue
            if parsed.netloc.lower().removeprefix("www.") != host:
                continue
            path = parsed.path.lower()
            hint = next((h for h in PAGE_HINTS if h in path), None)
            if hint and not any(h == hint for h, _ in picks):
                picks.append((hint, url))
                seen.add(url)
        for _, url in picks[: MAX_PAGES - 1]:
            html = await _fetch(client, url)
            if html:
                text, _ = _page_text(html)
                if len(text) > 200:
                    pages.append((url, text))
    return pages


async def search(query: str, api_key: str, recent: bool = False) -> list[dict]:
    """Serper results as [{title, snippet, date, link}]."""
    body = {"q": query, "num": SEARCH_RESULTS}
    if recent:
        body["tbs"] = "qdr:y"  # past year only
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://google.serper.dev/search", json=body,
                                     headers={"X-API-KEY": api_key})
        data = resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        logger.debug(f"Research search failed ({query[:60]}): {e}")
        return []
    return [
        {k: str(item.get(k, "")) for k in ("title", "snippet", "date", "link")}
        for item in (data.get("organic") or [])[:SEARCH_RESULTS]
        if isinstance(item, dict)
    ]


async def research_prospect(prospect, company, serper_key: str = "") -> str:
    """A plain-text research dossier, every item tagged with its source."""
    website = (company.website or company.domain) if company else ""
    sections = []

    pages = await read_website(website)
    if pages:
        sections.append("THEIR WEBSITE (verbatim text, boilerplate removed):")
        sections += [f"[{url}]\n{text}" for url, text in pages]

    if serper_key and prospect.company:
        news = await search(f'"{prospect.company}"', serper_key, recent=True)
        if news:
            sections.append("WEB RESULTS ABOUT THE COMPANY (past year):")
            sections += [f"[{r['link']}] {r['date']} {r['title']}: {r['snippet']}" for r in news]
        person = await search(f'"{prospect.full_name()}" "{prospect.company}"', serper_key)
        if person:
            sections.append("WEB RESULTS ABOUT THE PERSON (may include other people with "
                            "the same name; only trust results that also mention the company):")
            sections += [f"[{r['link']}] {r['date']} {r['title']}: {r['snippet']}" for r in person]

    return "\n\n".join(sections)
