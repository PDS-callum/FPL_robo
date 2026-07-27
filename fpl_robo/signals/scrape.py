"""Fetch Premier League–relevant headlines from allowlisted RSS feeds."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from fpl_robo.config import SIGNAL_INGEST

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsArticle:
    source: str
    title: str
    summary: str
    link: str
    published: str | None = None

    def text_blob(self, max_chars: int | None = None) -> str:
        limit = int(max_chars or SIGNAL_INGEST["max_article_chars"])
        body = f"{self.title}. {self.summary}".strip()
        body = _WS_RE.sub(" ", body)
        return body[:limit]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_html(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


def _child_text(node: ET.Element, *names: str) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return str(child.text)
        # RSS sometimes uses namespaced tags.
        for child in list(node):
            if child.tag.endswith(name) and child.text:
                return str(child.text)
    return ""


def _parse_published(raw: str) -> str | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return raw.strip() or None


def parse_rss(xml_text: str, *, source: str) -> list[NewsArticle]:
    """Parse RSS/Atom XML into articles."""
    root = ET.fromstring(xml_text)
    articles: list[NewsArticle] = []

    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for item in items:
        title = _strip_html(_child_text(item, "title"))
        summary = _strip_html(
            _child_text(item, "description", "summary", "content")
            or _child_text(item, "{http://www.w3.org/2005/Atom}summary")
        )
        link = _child_text(item, "link", "id")
        if not link:
            link_el = item.find("link")
            if link_el is not None:
                link = link_el.attrib.get("href") or (link_el.text or "")
        published = _parse_published(_child_text(item, "pubDate", "published", "updated"))
        if not title:
            continue
        articles.append(
            NewsArticle(
                source=source,
                title=title,
                summary=summary,
                link=link.strip(),
                published=published,
            )
        )
    return articles


def fetch_feed(url: str, *, source: str, timeout: float | None = None) -> list[NewsArticle]:
    timeout = float(timeout or SIGNAL_INGEST["request_timeout_seconds"])
    req = Request(
        url,
        headers={
            "User-Agent": "FPL-Robo/1.0 (+local research; contact: local)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - allowlisted feed URLs
        xml_text = resp.read().decode("utf-8", errors="replace")
    return parse_rss(xml_text, source=source)


def fetch_all_feeds(
    feeds: list[dict[str, str]] | None = None,
    *,
    max_articles: int | None = None,
    prefer_lineups: bool = False,
) -> tuple[list[NewsArticle], list[dict[str, Any]]]:
    """Fetch configured RSS feeds. Returns (articles, feed_errors)."""
    feeds = feeds or list(SIGNAL_INGEST["rss_feeds"])
    limit = int(max_articles or SIGNAL_INGEST["max_articles"])
    articles: list[NewsArticle] = []
    errors: list[dict[str, Any]] = []

    for feed in feeds:
        name = str(feed.get("name") or "feed")
        url = str(feed.get("url") or "")
        if not url:
            continue
        try:
            items = fetch_feed(url, source=name)
            articles.extend(items)
        except (URLError, TimeoutError, ET.ParseError, OSError, ValueError) as exc:
            errors.append({"source": name, "url": url, "error": str(exc)})

    # Prefer Premier League–ish titles, then newest-ish order preserved from feeds.
    def score(a: NewsArticle) -> tuple:
        blob = f"{a.title} {a.summary}".lower()
        # Strong availability keywords first.
        strong = sum(
            1
            for k in (
                "injury",
                "injured",
                "hamstring",
                "knock",
                "doubt",
                "ruled out",
                "suspended",
                "suspension",
                "ban",
                "fitness",
                "team news",
                "return to training",
                "sidelined",
            )
            if k in blob
        )
        lineup = sum(
            1
            for k in (
                "starting xi",
                "predicted xi",
                "predicted lineup",
                "predicted line-up",
                "team news",
                "line-up",
                "lineup",
                "how they line up",
                "likely xi",
                "expected xi",
            )
            if k in blob
        )
        weak = 1 if any(
            k in blob for k in ("premier league", "premiership", "fantasy")
        ) else 0
        # Deprioritise listicles / rankings that waste LLM time.
        junk = 1 if any(
            k in blob
            for k in ("best male footballers", "next generation", "100 best", "power ranking")
        ) else 0
        if prefer_lineups:
            return (junk, -lineup, -strong, -weak, a.published or "")
        return (junk, -strong, -lineup, -weak, a.published or "")

    articles = sorted(articles, key=score)
    # Dedupe by title
    seen: set[str] = set()
    unique: list[NewsArticle] = []
    for art in articles:
        key = art.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(art)
        if len(unique) >= limit:
            break

    return unique, errors


def looks_like_lineup_article(article: NewsArticle) -> bool:
    blob = f"{article.title} {article.summary}".lower()
    return any(
        k in blob
        for k in (
            "starting xi",
            "predicted xi",
            "predicted lineup",
            "predicted line-up",
            "team news",
            "line-up",
            "lineup",
            "how they line up",
            "likely xi",
            "expected xi",
        )
    )


def articles_payload(articles: list[NewsArticle], errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(articles),
        "errors": errors,
        "articles": [a.to_dict() for a in articles],
    }
