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
    body: str = ""

    def text_blob(self, max_chars: int | None = None) -> str:
        limit = int(max_chars or SIGNAL_INGEST["max_article_chars"])
        # When a body was fetched, allow a larger blob for heuristics/LLM.
        if self.body:
            limit = max(limit, int(SIGNAL_INGEST.get("article_body_max_chars") or limit))
        summary = (self.summary or "").split("The post ")[0].split("appeared first on")[0]
        parts = [self.title, summary, self.body]
        body = ". ".join(p.strip() for p in parts if p and str(p).strip())
        body = _WS_RE.sub(" ", body)
        return body[:limit]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def summary_is_thin(self) -> bool:
        clean = (self.summary or "").split("The post ")[0].split("appeared first on")[0].strip()
        return len(clean) < 160


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

    def _published_ts(a: NewsArticle) -> float:
        if not a.published:
            return 0.0
        try:
            dt = datetime.fromisoformat(a.published.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0

    # Prefer actionable PL news; break ties with newest first.
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
                "out for",
                "withdrawn",
            )
            if k in blob
        )
        transfer = sum(
            1
            for k in (
                "signs for",
                "signed for",
                "joins",
                "has joined",
                "complete signing",
                "completes move",
                "seal deal",
                "transfer",
                "new signing",
                "unveiled",
            )
            if k in blob
        )
        form = sum(
            1
            for k in (
                "hat-trick",
                "hattrick",
                "hat trick",
                "brace",
                "pre-season",
                "preseason",
                "friendly",
                "scored",
                "scores",
                "goals",
                "debut",
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
            k in blob
            for k in ("premier league", "premiership", "fantasy", "chelsea", "arsenal", "liverpool")
        ) else 0
        # Deprioritise listicles / rankings that waste LLM time.
        junk = 1 if any(
            k in blob
            for k in (
                "best male footballers",
                "next generation",
                "100 best",
                "power ranking",
                "quiz:",
                "your views",
                "live blog",
                "team reveals",
                "first draft",
                "no bruno draft",
                "best £",
                "best $",
                "who are the best",
                "club-by-club guide",
                "lips out",
                "eagle!",
                "st helens",
                "super league",
                "national bank open",
                "wimbledon",
                "falklands",
                "wild card for",
                "italy head coach",
                "fifa action",
                "fifa opens",
            )
        ) else 0
        # Rumour-only transfer gossip without confirmation language.
        rumour = 1 if (
            transfer
            and any(k in blob for k in ("linked with", "interested in", "set to bid", "rumour"))
            and not any(
                k in blob
                for k in ("signed", "joins", "has joined", "completes", "confirmed", "unveiled")
            )
        ) else 0
        newest = -_published_ts(a)
        if prefer_lineups:
            return (junk, rumour, -lineup, -strong, -form, -transfer, -weak, newest)
        return (junk, rumour, -strong, -form, -transfer, -lineup, -weak, newest)

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
    title = (article.title or "").lower()
    # FFS (and similar) footers always mention "Team News" — ignore that boilerplate.
    summary = (article.summary or "").lower().split("appeared first on")[0]
    blob = f"{title} {summary}"
    strong = (
        "starting xi",
        "predicted xi",
        "predicted lineup",
        "predicted line-up",
        "how they line up",
        "likely xi",
        "expected xi",
        "confirmed xi",
        "team sheet",
    )
    if any(k in blob for k in strong):
        return True
    # Weaker "team news" / "lineup" phrases only count when they appear in the title.
    return any(k in title for k in ("team news", "line-up", "lineup", "starting xi"))


def article_is_actionable(article: NewsArticle) -> bool:
    """True when the headline/summary looks like injury/transfer/form/lineup news."""
    title = (article.title or "").lower()
    summary = (article.summary or "").lower().split("appeared first on")[0]
    blob = f"{title} {summary}"
    keys = (
        "injur",
        "doubt",
        "sidelined",
        "ruled out",
        "hamstring",
        "knock",
        "suspend",
        "ban",
        "absent",
        "missed out",
        "misses",
        "months out",
        "groin",
        "hat-trick",
        "hattrick",
        "hat trick",
        "brace",
        "scored",
        "scores",
        "lively",
        "sharp",
        "winner",
        "impressive",
        "signed",
        "sign ",
        "signs ",
        "joins",
        "joined",
        "completes",
        "complete deal",
        "deal for",
        "signing of",
        "transfer",
        "new signing",
        "unveiled",
        "out for",
        "months out",
        "months",
        "starting xi",
        "predicted xi",
        "team news",
        "line-up",
        "lineup",
        "pre-season",
        "preseason",
        "friendly",
        "after",
        "moved to",
        "latest",
    )
    return any(k in blob for k in keys)


def _host_allowlisted(url: str) -> bool:
    from urllib.parse import urlparse

    hosts = {
        str(h).lower().strip()
        for h in (SIGNAL_INGEST.get("article_body_hosts") or [])
        if h
    }
    if not hosts:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return host in hosts or any(host.endswith("." + h) for h in hosts)


def fetch_article_body(url: str, *, timeout: float | None = None, max_chars: int | None = None) -> str:
    """Fetch and lightly clean an allowlisted article HTML body."""
    if not url or not _host_allowlisted(url):
        return ""
    timeout = float(timeout or SIGNAL_INGEST.get("article_body_timeout_seconds") or 12)
    max_chars = int(max_chars or SIGNAL_INGEST.get("article_body_max_chars") or 2800)
    req = Request(
        url,
        headers={
            "User-Agent": "FPL-Robo/1.0 (+local research; contact: local)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - allowlisted hosts
            raw = resp.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError, ValueError):
        return ""
    # Drop scripts/styles then strip tags.
    raw = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", raw)
    # Prefer paragraph text when present.
    paras = re.findall(r"(?is)<p[^>]*>(.*?)</p>", raw)
    if paras:
        text = " ".join(_strip_html(p) for p in paras)
    else:
        text = _strip_html(raw)
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_chars]


def needs_article_body(article: NewsArticle) -> bool:
    """True when RSS snippet is thin but the headline looks worth expanding."""
    if not article.link or article.body:
        return False
    if not article_is_actionable(article):
        return False
    title = (article.title or "").lower()
    # Always expand transfer/injury headlines with thin blurbs.
    interesting = any(
        k in title
        for k in (
            "injur",
            "sign",
            "deal",
            "transfer",
            "hat-trick",
            "miss",
            "absent",
            "groin",
            "latest",
            "out for",
            "ruled out",
        )
    )
    return interesting and (article.summary_is_thin or "latest" in title)


def enrich_articles_with_bodies(
    articles: list[NewsArticle],
    *,
    max_fetches: int | None = None,
) -> list[NewsArticle]:
    """Attach HTML bodies to the most promising thin RSS items."""
    if not bool(SIGNAL_INGEST.get("fetch_article_body", True)):
        return articles
    budget = int(max_fetches if max_fetches is not None else SIGNAL_INGEST.get("article_body_max_fetches") or 8)
    out: list[NewsArticle] = []
    fetched = 0
    for art in articles:
        if fetched < budget and needs_article_body(art):
            body = fetch_article_body(art.link)
            if body:
                fetched += 1
                art = NewsArticle(
                    source=art.source,
                    title=art.title,
                    summary=art.summary,
                    link=art.link,
                    published=art.published,
                    body=body,
                )
        out.append(art)
    return out


def articles_payload(articles: list[NewsArticle], errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(articles),
        "errors": errors,
        "articles": [a.to_dict() for a in articles],
    }
