# Streamlit Technical SEO Auditor
# -----------------------------------------------------------
# A single-file Streamlit app for lightweight technical SEO audits.
# Now includes checks extracted & hard-wired from your CSV:
# - AMP basics, canonical variants, validation bits, security hygiene,
# - analytics detection, hreflang sanity, link hygiene, plus resource-level URLs.
#
# ▶ Run locally:
#   pip install streamlit requests beautifulsoup4 pandas lxml openpyxl
#   streamlit run streamlit_technical_seo_auditor_app.py
# -----------------------------------------------------------

from __future__ import annotations

import json
import re
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup, Doctype
import streamlit as st
import urllib.robotparser as robotparser

# -------------------------------
# Configuration
# -------------------------------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 SEO-Auditor/1.2"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

REQUEST_TIMEOUT = 20  # seconds

TITLE_RECOMMENDED_MIN = 30
TITLE_RECOMMENDED_MAX = 60
META_DESC_RECOMMENDED_MIN = 70
META_DESC_RECOMMENDED_MAX = 160
SLOW_RESPONSE_MS = 1500
LARGE_PAGE_BYTES = 2_000_000  # ~2 MB

# -------------------------------
# Dataclasses
# -------------------------------
@dataclass
class PageAudit:
    url: str
    status: int | None
    redirected: bool
    final_url: str
    response_time_ms: int | None
    page_bytes: int | None

    title: str | None
    title_length: int | None

    meta_description: str | None
    meta_description_length: int | None

    meta_robots: str | None
    x_robots: str | None

    canonical: str | None
    canonical_resolves: bool | None
    canonical_is_self: bool | None
    canonical_count: int | None

    h1_count: int | None
    h2_count: int | None

    images: int | None
    images_missing_alt: int | None

    internal_links: int | None
    external_links: int | None
    nofollow_links: int | None

    hreflang_count: int | None
    structured_data_types: str | None  # comma list

    html_lang: str | None
    viewport: bool

    https: bool
    mixed_content: int | None

    indexable: bool | None
    blocked_by_robots_meta: bool | None
    blocked_by_xrobots: bool | None

    # New technical signals (non-defaults)
    head_count: int | None
    body_count: int | None
    doctype_present: bool | None
    has_amp: bool | None
    has_ga: bool | None
    has_gtm: bool | None
    rel_prev: bool | None
    rel_next: bool | None
    target_blank_no_noopener_count: int | None
    links_empty_text_count: int | None
    hreflang_invalid_count: int | None
    hsts_header: bool | None

    # Defaulted (lists or optional extras) must come last
    images_missing_alt_urls: list[str] | None = None
    mixed_content_urls: list[str] | None = None
    inlinks_internal: int | None = None  # computed post-crawl
    target_blank_no_noopener_urls: list[str] | None = None
    links_empty_text_urls: list[str] | None = None
    hreflang_invalid_values: list[str] | None = None

    notes: str = ""

# -------------------------------
# Utility functions
# -------------------------------
def normalise_url(u: str) -> str:
    parsed = urlparse(u)
    scheme = (parsed.scheme or "https").lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if (":" in netloc):
        host, _, port = netloc.partition(":")
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host
    path = re.sub(r"//+", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, netloc, path, "", parsed.query or "", ""))

def same_host(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.netloc.lower().lstrip("www.") == pb.netloc.lower().lstrip("www.")

def is_http_url(u: str) -> bool:
    try:
        return urlparse(u).scheme in {"http", "https"}
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def fetch_url(url: str):
    start = time.time()
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        content = resp.content
        elapsed = time.time() - start
        return resp, content, None, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return None, None, str(e), elapsed

@st.cache_data(show_spinner=False)
def read_robots_txt(base_url: str) -> robotparser.RobotFileParser:
    parsed = urlparse(base_url)
    robots_url = urlunparse((parsed.scheme or "https", parsed.netloc, "/robots.txt", "", "", ""))
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    return rp

@st.cache_data(show_spinner=False)
def parse_sitemap_urls(base_url: str) -> list[str]:
    urls: list[str] = []
    parsed = urlparse(base_url)
    sitemap_url = urlunparse((parsed.scheme or "https", parsed.netloc, "/sitemap.xml", "", "", ""))

    resp, content, err, _ = fetch_url(sitemap_url)
    if err or not resp or resp.status_code >= 400:
        return urls
    try:
        tree = ET.fromstring(content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        if tree.tag.endswith("sitemapindex"):
            for smap in tree.findall("sm:sitemap", ns):
                loc_el = smap.find("sm:loc", ns)
                if loc_el is not None and loc_el.text:
                    child_url = loc_el.text.strip()
                    r2, c2, e2, _ = fetch_url(child_url)
                    if not e2 and r2 and r2.status_code < 400:
                        try:
                            t2 = ET.fromstring(c2)
                            for url_el in t2.findall(".//sm:url/sm:loc", ns):
                                if url_el.text:
                                    urls.append(url_el.text.strip())
                        except Exception:
                            continue
        else:
            for url_el in tree.findall(".//sm:url/sm:loc", ns):
                if url_el.text:
                    urls.append(url_el.text.strip())
    except Exception:
        pass
    return urls

# -------------------------------
# Page analysis
# -------------------------------
AMP_SCRIPT_RE = re.compile(r"https://cdn\\.ampproject\\.org/v0\\.js")
AMP_HTML_ATTRS = {"amp", "⚡", "amp4ads", "amp4email"}  # allow variants, basic

BCP47_LAX = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")

def detect_doctype(content: bytes | None) -> bool | None:
    if not content:
        return None
    head = content[:2048].decode("latin-1", errors="ignore").lower()
    return "<!doctype html" in head

def analyse_page(url: str, resp, content: bytes | None, err: str | None, elapsed: float) -> PageAudit:
    status = None if not resp else resp.status_code
    redirected = bool(resp.history) if resp else False
    final_url = resp.url if resp else url
    response_time_ms = int(elapsed * 1000)
    page_bytes = len(content) if content else None

    title = None
    meta_description = None
    meta_robots = None
    canonical = None
    canonical_count = 0
    h1_count = h2_count = None
    images = images_missing_alt = None
    images_missing_alt_urls: list[str] = []
    internal_links = external_links = nofollow_links = 0
    hreflang_count = 0
    structured_types: list[str] = []
    html_lang = None
    viewport = False
    https = urlparse(final_url).scheme == "https"
    mixed_content = 0
    mixed_content_urls: list[str] = []

    # New technical signals
    head_count = body_count = 0
    doctype_present = detect_doctype(content)
    has_amp = False
    has_ga = False
    has_gtm = False
    rel_prev = False
    rel_next = False
    target_blank_no_noopener_count = 0
    target_blank_no_noopener_urls: list[str] = []
    links_empty_text_count = 0
    links_empty_text_urls: list[str] = []
    hreflang_invalid_values: list[str] = []
    hsts_header = bool(resp.headers.get("Strict-Transport-Security")) if (resp and resp.headers) else False

    x_robots = None if not resp else resp.headers.get("X-Robots-Tag")

    notes: list[str] = []

    if err:
        notes.append(f"Fetch error: {err}")

    ctype = (resp.headers.get("Content-Type", "") if resp else "").lower()
    if content and resp and ("text/html" in ctype or "application/xhtml+xml" in ctype or ctype.startswith("text/")):
        soup = BeautifulSoup(content, "lxml")

        # Doctype (BeautifulSoup can sometimes expose it)
        # Already detected via prefix scan; leave as-is.

        # Language & viewport
        html = soup.find("html")
        if html:
            if html.has_attr("lang"):
                html_lang = (html.get("lang") or "").strip()
            # AMP attribute?
            for a in AMP_HTML_ATTRS:
                if html.has_attr(a):
                    has_amp = True
                    break
        if soup.find("meta", attrs={"name": "viewport"}):
            viewport = True

        # Head/body counts
        head_count = len(soup.find_all("head"))
        body_count = len(soup.find_all("body"))

        # Title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Meta description / robots
        md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if md and md.has_attr("content"):
            meta_description = md["content"].strip()
        mrobots = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
        if mrobots and mrobots.has_attr("content"):
            meta_robots = mrobots["content"].strip()

        # Canonical(s)
        canon_links = soup.find_all("link", rel=lambda x: x and "canonical" in [v.lower() for v in (x if isinstance(x, list) else [x])])
        canonical_count = len(canon_links)
        if canon_links:
            can = canon_links[0]
            if can and can.has_attr("href"):
                canonical = can["href"].strip()

        # Headings
        h1_count = len(soup.find_all("h1"))
        h2_count = len(soup.find_all("h2"))

        # Images & ALT
        imgs = soup.find_all("img")
        images = len(imgs)
        for i in imgs:
            alt = i.get("alt")
            if not alt or not str(alt).strip():
                src = i.get("src") or ""
                full_src = urljoin(final_url, src)
                images_missing_alt_urls.append(full_src)
        images_missing_alt = len(images_missing_alt_urls)

        # Links
        final_host = urlparse(final_url).netloc
        for a in soup.find_all("a", href=True):
            href_raw = a.get("href", "").strip()
            if href_raw.startswith("mailto:") or href_raw.startswith("tel:"):
                continue
            full = urljoin(final_url, href_raw)
            if not is_http_url(full):
                continue

            # link counts
            if urlparse(full).netloc == final_host:
                internal_links += 1
            else:
                external_links += 1
            rel_vals = [r.lower() for r in (a.get("rel") or (a.get("rel") if isinstance(a.get("rel"), list) else []))]
            if "nofollow" in rel_vals:
                nofollow_links += 1

            # security: target=_blank without rel=noopener
            if (a.get("target") or "").lower() == "_blank":
                if not any(rv in {"noopener", "noreferrer"} for rv in rel_vals):
                    target_blank_no_noopener_count += 1
                    target_blank_no_noopener_urls.append(full)

            # empty/whitespace anchor text
            text = (a.get_text() or "").strip()
            if text == "":
                links_empty_text_count += 1
                links_empty_text_urls.append(full)

        # prev/next
        for link in soup.find_all("link", rel=True, href=True):
            rels = [r.lower() for r in (link.get("rel") if isinstance(link.get("rel"), list) else [link.get("rel")])]
            if "prev" in rels:
                rel_prev = True
            if "next" in rels:
                rel_next = True

        # hreflang
        for link in soup.find_all("link", rel=True, href=True):
            rels = [r.lower() for r in (link.get("rel") if isinstance(link.get("rel"), list) else [link.get("rel")])]
            if "alternate" in rels and link.get("hreflang"):
                hreflang_count += 1
                val = str(link.get("hreflang")).strip()
                if not BCP47_LAX.match(val.lower()) and val.lower() != "x-default":
                    hreflang_invalid_values.append(val)

        # Structured data types
        for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\\+json", re.I)}):
            try:
                data = json.loads(script.text)
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict) and d.get("@type"):
                            structured_types.append(str(d.get("@type")))
                elif isinstance(data, dict) and data.get("@type"):
                    structured_types.append(str(data.get("@type")))
            except Exception:
                continue

        # Analytics detection
        for s in soup.find_all("script", src=True):
            src = (s.get("src") or "").strip()
            if "gtag/js" in src:
                has_ga = True
            if "googletagmanager.com/gtm.js" in src or "googletagmanager.com/gtag/js" in src:
                has_gtm = True
        script_text = " ".join((s.get_text() or "") for s in soup.find_all("script", src=False))[:20000].lower()
        if ("gtag(" in script_text or "google-analytics.com/analytics.js" in script_text):
            has_ga = True
        if "googletagmanager.com/gtm.js" in script_text or "dataLayer" in script_text:
            has_gtm = True

        # Mixed content check
        if https:
            for tag in soup.find_all(src=True):
                src = tag.get("src")
                if isinstance(src, str) and src.startswith("http://"):
                    mixed_content += 1
                    mixed_content_urls.append(urljoin(final_url, src))
            for tag in soup.find_all(href=True):
                href = tag.get("href")
                if isinstance(href, str) and href.startswith("http://"):
                    mixed_content += 1
                    mixed_content_urls.append(urljoin(final_url, href))

    # Derived fields
    title_length = len(title) if title else None
    meta_description_length = len(meta_description) if meta_description else None

    blocked_by_robots_meta = bool(meta_robots and re.search(r"noindex", meta_robots, re.I))
    blocked_by_xrobots = bool(x_robots and re.search(r"noindex", x_robots, re.I))

    # Canonical checks (only for first canonical)
    canonical_resolves = None
    canonical_is_self = None
    if canonical:
        can_abs = urljoin(final_url, canonical)
        try:
            r_can, _, _, _ = fetch_url(can_abs)
            canonical_resolves = bool(r_can and r_can.status_code < 400)
        except Exception:
            canonical_resolves = False
        canonical_is_self = normalise_url(can_abs) == normalise_url(final_url)

    # Indexability (simplified)
    indexable = None
    if status is not None:
        indexable = (200 <= status < 300) and not blocked_by_robots_meta and not blocked_by_xrobots

    # Notes aggregation
    notes_local: list[str] = []
    if title_length is not None and (title_length < TITLE_RECOMMENDED_MIN or title_length > TITLE_RECOMMENDED_MAX):
        notes_local.append("Title length outside typical range")
    if meta_description and (meta_description_length < META_DESC_RECOMMENDED_MIN or meta_description_length > META_DESC_RECOMMENDED_MAX):
        notes_local.append("Meta description length outside typical range")
    if h1_count == 0:
        notes_local.append("Missing H1 heading")
    if images_missing_alt and images_missing_alt > 0:
        notes_local.append("Images missing ALT text present")
    if mixed_content and mixed_content > 0:
        notes_local.append("Mixed content (HTTP resources on HTTPS page)")

    return PageAudit(
        url=url,
        status=status,
        redirected=redirected,
        final_url=final_url,
        response_time_ms=response_time_ms,
        page_bytes=page_bytes,
        title=title,
        title_length=title_length,
        meta_description=meta_description,
        meta_description_length=meta_description_length,
        meta_robots=meta_robots,
        x_robots=x_robots,
        canonical=canonical,
        canonical_resolves=canonical_resolves,
        canonical_is_self=canonical_is_self,
        canonical_count=canonical_count or None,
        h1_count=h1_count,
        h2_count=h2_count,
        images=images,
        images_missing_alt=images_missing_alt,
        internal_links=internal_links,
        external_links=external_links,
        nofollow_links=nofollow_links,
        hreflang_count=hreflang_count,
        structured_data_types=", ".join([str(s) for s in structured_types]) if structured_types else None,
        html_lang=html_lang,
        viewport=viewport,
        https=https,
        mixed_content=mixed_content,
        indexable=indexable,
        blocked_by_robots_meta=blocked_by_robots_meta,
        blocked_by_xrobots=blocked_by_xrobots,
        head_count=head_count or None,
        body_count=body_count or None,
        doctype_present=doctype_present,
        has_amp=has_amp,
        has_ga=has_ga,
        has_gtm=has_gtm,
        rel_prev=rel_prev,
        rel_next=rel_next,
        target_blank_no_noopener_count=target_blank_no_noopener_count or None,
        links_empty_text_count=links_empty_text_count or None,
        hreflang_invalid_count=(len(hreflang_invalid_values) or None),
        hsts_header=hsts_header if https else None,
        images_missing_alt_urls=images_missing_alt_urls or None,
        mixed_content_urls=mixed_content_urls or None,
        target_blank_no_noopener_urls=target_blank_no_noopener_urls or None,
        links_empty_text_urls=links_empty_text_urls or None,
        hreflang_invalid_values=hreflang_invalid_values or None,
        notes="; ".join(notes + notes_local),
    )

# -------------------------------
# Crawler
# -------------------------------
def crawl(start_url: str, max_pages: int = 50, obey_robots: bool = True, include_query_urls: bool = False, use_sitemap_seed: bool = True) -> list[PageAudit]:
    base = normalise_url(start_url)
    rp = read_robots_txt(base)

    visited: set[str] = set()
    audits: list[PageAudit] = []
    q: deque[str] = deque()

    # track internal inlinks between crawled URLs
    inlink_counts: defaultdict[str, int] = defaultdict(int)

    seeds: list[str] = []
    if use_sitemap_seed:
        seeds = [u for u in parse_sitemap_urls(base) if is_http_url(u)]
    if not seeds:
        seeds = [base]

    for s in seeds:
        q.append(normalise_url(s))

    pbar = st.progress(0, text="Crawling…")
    processed = 0

    while q and len(visited) < max_pages:
        current = q.popleft()
        if current in visited:
            continue
        if not same_host(current, base):
            continue
        if not include_query_urls and urlparse(current).query:
            continue
        if obey_robots and not rp.can_fetch(DEFAULT_HEADERS["User-Agent"], current):
            visited.add(current)
            continue

        visited.add(current)

        resp, content, err, elapsed = fetch_url(current)
        audit = analyse_page(current, resp, content, err, elapsed)
        audits.append(audit)
        processed += 1
        pbar.progress(min(int(processed / max_pages * 100), 100), text=f"Crawling… {processed}/{max_pages}")

        # Enqueue and count inlinks
        if resp and content and ("text/html" in (resp.headers.get("Content-Type", ""))):
            soup = BeautifulSoup(content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                    continue
                full = urljoin(audit.final_url, href)
                if is_http_url(full) and same_host(full, base):
                    n = normalise_url(full)
                    inlink_counts[n] += 1
                    if n not in visited and (include_query_urls or not urlparse(n).query):
                        q.append(n)

    pbar.progress(100, text="Crawl complete")

    # annotate inlinks on audits
    for a in audits:
        a.inlinks_internal = int(inlink_counts.get(normalise_url(a.final_url), 0))

    return audits

# -------------------------------
# Reporting helpers
# -------------------------------
def audits_to_df(audits: list[PageAudit]) -> pd.DataFrame:
    rows = [asdict(a) for a in audits]
    df = pd.DataFrame(rows)
    cols_order = [
        "url", "final_url", "status", "redirected", "indexable", "response_time_ms", "page_bytes",
        "title", "title_length", "meta_description", "meta_description_length",
        "meta_robots", "x_robots",
        "canonical", "canonical_resolves", "canonical_is_self", "canonical_count",
        "h1_count", "h2_count",
        "images", "images_missing_alt", "images_missing_alt_urls",
        "internal_links", "external_links", "nofollow_links",
        "hreflang_count", "hreflang_invalid_count", "hreflang_invalid_values", "structured_data_types",
        "html_lang", "viewport",
        "https", "mixed_content", "mixed_content_urls",
        "head_count", "body_count", "doctype_present",
        "has_amp", "has_ga", "has_gtm", "rel_prev", "rel_next",
        "target_blank_no_noopener_count", "target_blank_no_noopener_urls",
        "links_empty_text_count", "links_empty_text_urls",
        "inlinks_internal",
        "hsts_header",
        "blocked_by_robots_meta", "blocked_by_xrobots",
        "notes",
    ]
    existing_cols = [c for c in cols_order if c in df.columns]
    return df[existing_cols]

def summarise_issues(df: pd.DataFrame) -> dict[str, int]:
    issues = {
        "Non-200 status": int(((df["status"].fillna(0).astype(int) < 200) | (df["status"].fillna(0).astype(int) >= 300)).sum()) if "status" in df else 0,
        "Missing or multiple H1": int(((df["h1_count"].fillna(0).astype(int) != 1)).sum()) if "h1_count" in df else 0,
        "Missing H2": int(((df["h2_count"].fillna(0).astype(int) == 0)).sum()) if "h2_count" in df else 0,
        "Title too short/long or missing": int(((df["title_length"].fillna(0) < TITLE_RECOMMENDED_MIN) | (df["title_length"].fillna(0) > TITLE_RECOMMENDED_MAX)).sum()) if "title_length" in df else 0,
        "Meta description suboptimal or missing": int(((df["meta_description_length"].fillna(0) < META_DESC_RECOMMENDED_MIN) | (df["meta_description_length"].fillna(0) > META_DESC_RECOMMENDED_MAX)).sum()) if "meta_description_length" in df else 0,
        "Duplicate titles": int((df["title"].fillna("").duplicated(keep=False)).sum()) if "title" in df else 0,
        "Duplicate meta descriptions": int((df["meta_description"].fillna("").duplicated(keep=False)).sum()) if "meta_description" in df else 0,
        "Noindex (meta or header)": int(((df["blocked_by_robots_meta"].fillna(False)) | (df["blocked_by_xrobots"].fillna(False))).sum()) if "blocked_by_robots_meta" in df and "blocked_by_xrobots" in df else 0,
        "Nofollow (meta or header)": int(((df["meta_robots"].fillna("").str.contains("nofollow", case=False)) | (df["x_robots"].fillna("").str.contains("nofollow", case=False))).sum()) if "meta_robots" in df and "x_robots" in df else 0,
        "Missing canonical": int(((df["canonical"].isna()) | (df["canonical"].astype(str).str.len() == 0)).sum()) if "canonical" in df else 0,
        "Multiple canonicals": int(((df["canonical_count"].fillna(0).astype(int) > 1)).sum()) if "canonical_count" in df else 0,
        "Canonical not self-referential": int(((df["canonical_is_self"].fillna(True) == False)).sum()) if "canonical_is_self" in df else 0,
        "Canonical broken": int(((df["canonical_resolves"].fillna(True) == False)).sum()) if "canonical_resolves" in df else 0,
        "Images missing ALT": int(((df["images_missing_alt"].fillna(0) > 0)).sum()) if "images_missing_alt" in df else 0,
        "Mixed content on HTTPS": int(((df["mixed_content"].fillna(0) > 0)).sum()) if "mixed_content" in df else 0,
        "Target=_blank without rel=noopener": int(((df["target_blank_no_noopener_count"].fillna(0).astype(int) > 0)).sum()) if "target_blank_no_noopener_count" in df else 0,
        "Empty anchor text": int(((df["links_empty_text_count"].fillna(0).astype(int) > 0)).sum()) if "links_empty_text_count" in df else 0,
        "No viewport meta": int(((df["viewport"].fillna(False) == False)).sum()) if "viewport" in df else 0,
        "Missing HTML lang": int(((df["html_lang"].fillna("").astype(str).str.len() == 0)).sum()) if "html_lang" in df else 0,
        "Invalid hreflang syntax": int(((df["hreflang_invalid_count"].fillna(0).astype(int) > 0)).sum()) if "hreflang_invalid_count" in df else 0,
        "Slow responses (>1500ms)": int(((df["response_time_ms"].fillna(0).astype(int) > SLOW_RESPONSE_MS)).sum()) if "response_time_ms" in df else 0,
        "Large pages (>2MB)": int(((df["page_bytes"].fillna(0).astype(int) > LARGE_PAGE_BYTES)).sum()) if "page_bytes" in df else 0,
        "Orphan-ish (0 inlinks in crawl)": int(((df["inlinks_internal"].fillna(0).astype(int) == 0)).sum()) if "inlinks_internal" in df else 0,
        "Missing HSTS header on HTTPS": int(((df["https"].fillna(False)) & (df["hsts_header"].fillna(False) == False)).sum()) if "https" in df and "hsts_header" in df else 0,
        "AMP: Missing <html amp>": int(((df["has_amp"].fillna(False) == False)).sum()) if "has_amp" in df else 0,
        "AMP: Missing runtime script": int(((df["has_amp"].fillna(False)) & (df["notes"].fillna("").str.contains("Missing/Invalid AMP Script", case=False) == False)).sum()) if "has_amp" in df and "notes" in df else 0,
        "Analytics: GA/GTM not detected": int(((df["has_ga"].fillna(False) == False) & (df["has_gtm"].fillna(False) == False)).sum()) if "has_ga" in df and "has_gtm" in df else 0,
        "Missing/invalid doctype": int(((df["doctype_present"].fillna(True) == False)).sum()) if "doctype_present" in df else 0,
        "Missing <head> or <body>": int(((df["head_count"].fillna(0).astype(int) == 0) | (df["body_count"].fillna(0).astype(int) == 0)).sum()) if "head_count" in df and "body_count" in df else 0,
        "Multiple <head>/<body>": int(((df["head_count"].fillna(0).astype(int) > 1) | (df["body_count"].fillna(0).astype(int) > 1)).sum()) if "head_count" in df and "body_count" in df else 0,
    }
    return issues

def _issue_df(df: pd.DataFrame, mask: pd.Series, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    out = df.loc[mask, cols].copy()
    # string-ify lists for display
    for c in ["images_missing_alt_urls", "mixed_content_urls", "target_blank_no_noopener_urls", "links_empty_text_urls", "hreflang_invalid_values"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: "; ".join(v) if isinstance(v, list) else (v or ""))
    if "final_url" not in out.columns and "final_url" in df.columns:
        out.insert(0, "final_url", df.loc[mask, "final_url"])
    return out

def _slugify(text: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return t or "issue"

def colour_badge(v: bool | int | None, good_when: str = "true") -> str:
    if v is None:
        return "⚪"
    if isinstance(v, bool):
        return "🟢" if (v and good_when == "true") or ((not v) and good_when == "false") else "🟠"
    if isinstance(v, int):
        if v == 0:
            return "🟢"
        if v < 5:
            return "🟠"
        return "🔴"
    return "⚪"

def build_issue_catalogue(df: pd.DataFrame) -> list[dict]:
    """Curated issue catalogue (labels mirror the CSV where feasible)."""
    if df.empty:
        return []
    def s(name, default):
        return df[name] if name in df else pd.Series([default]*len(df), index=df.index)

    items: list[dict] = []

    def add(label, mask, description, cols):
        items.append({"label": label, "mask": mask, "description": description, "cols": cols})

    # Response / indexability
    add("Non-200 status", (s("status", 0).astype("Int64").lt(200) | s("status", 0).astype("Int64").ge(300)),
        "Pages that returned redirects, client, or server errors.", ["final_url","status","notes"])
    add("Noindex (meta or header)", (s("blocked_by_robots_meta", False) | s("blocked_by_xrobots", False)),
        "Pages blocked from indexing via meta robots or X-Robots-Tag.", ["final_url","meta_robots","x_robots"])
    add("Nofollow (meta or header)", (s("meta_robots","").astype(str).str.contains("nofollow", case=False) | s("x_robots","").astype(str).str.contains("nofollow", case=False)),
        "Page marked nofollow at page or header level.", ["final_url","meta_robots","x_robots"])

    # Canonicals
    add("Missing canonical", (s("canonical","").astype(str).str.len()==0),
        "No canonical link tag found.", ["final_url","canonical"])
    add("Multiple canonicals", (s("canonical_count",0).fillna(0).astype(int) > 1),
        "More than one canonical tag declared.", ["final_url","canonical_count","canonical"])
    add("Canonical not self-referential", (s("canonical_is_self", True).fillna(True) == False),
        "Canonical points away from the current URL (check it's intentional).", ["final_url","canonical","canonical_is_self"])
    add("Canonical broken (does not resolve)", (s("canonical_resolves", True).fillna(True) == False),
        "Canonical URL appears not to resolve (4xx/5xx).", ["final_url","canonical","canonical_resolves"])

    # Titles / Descriptions
    add("Title too short/long or missing", (s("title_length",0).fillna(0) < TITLE_RECOMMENDED_MIN) | (s("title_length",0).fillna(0) > TITLE_RECOMMENDED_MAX),
        "Title length outside typical range.", ["final_url","title","title_length"])
    add("Meta description suboptimal or missing", (s("meta_description_length",0).fillna(0) < META_DESC_RECOMMENDED_MIN) | (s("meta_description_length",0).fillna(0) > META_DESC_RECOMMENDED_MAX),
        "Meta description outside usual length range or missing.", ["final_url","meta_description","meta_description_length"])
    add("Duplicate titles", (s("title","").fillna("").duplicated(keep=False) & s("title","").fillna("").ne("")),
        "Same <title> used by multiple pages.", ["final_url","title"])
    add("Duplicate meta descriptions", (s("meta_description","").fillna("").duplicated(keep=False) & s("meta_description","").fillna("").ne("")),
        "Same meta description used by multiple pages.", ["final_url","meta_description"])

    # Headings
    add("Missing or multiple H1", (s("h1_count",0).fillna(0).astype(int) != 1),
        "Each page should generally have one clear H1.", ["final_url","h1_count","title"])
    add("Missing H2", (s("h2_count",0).fillna(0).astype(int) == 0),
        "No H2 headings found.", ["final_url","h2_count"])

    # Images (resource-level)
    add("Images missing ALT", (s("images_missing_alt",0).fillna(0).astype(int) > 0),
        "Images without alt text found on the page.", ["final_url","images","images_missing_alt","images_missing_alt_urls"])

    # Links
    add("Empty anchor text", (s("links_empty_text_count",0).fillna(0).astype(int) > 0),
        "Anchor tags whose visible text is empty/whitespace.", ["final_url","links_empty_text_count","links_empty_text_urls"])
    add("Target=_blank without rel=noopener", (s("target_blank_no_noopener_count",0).fillna(0).astype(int) > 0),
        "Security best practice: add rel='noopener' (or 'noreferrer').", ["final_url","target_blank_no_noopener_count","target_blank_no_noopener_urls"])

    # Mobile / viewport
    add("No viewport meta", (s("viewport",False) == False),
        "Missing responsive viewport meta tag.", ["final_url","viewport"])

    # Hreflang
    add("Invalid hreflang syntax", (s("hreflang_invalid_count",0).fillna(0).astype(int) > 0),
        "Hreflang values that don't match basic BCP-47 (or x-default).", ["final_url","hreflang_invalid_count","hreflang_invalid_values"])

    # Security
    add("Mixed content on HTTPS", (s("mixed_content",0).fillna(0).astype(int) > 0),
        "HTTPS pages referencing http:// assets (upgrade to HTTPS).", ["final_url","mixed_content","mixed_content_urls"])
    add("Missing HSTS header on HTTPS", (s("https",False) & (s("hsts_header",False) == False)),
        "Consider setting 'Strict-Transport-Security' for HTTPS.", ["final_url","hsts_header"])

    # AMP essentials
    add("AMP: Missing <html amp>", (s("has_amp",False) == False),
        "AMP page not detected (no <html amp|⚡>).", ["final_url","has_amp"])
    # Note: deep AMP validity isn't feasible without a validator; we keep this minimal.

    # Performance / size / crawl graph
    add("Slow responses (>1500ms)", (s("response_time_ms",0).fillna(0).astype(int) > SLOW_RESPONSE_MS),
        "Server response time above threshold.", ["final_url","response_time_ms"])
    add("Large pages (>2MB)", (s("page_bytes",0).fillna(0).astype(int) > LARGE_PAGE_BYTES),
        "HTML response bytes exceed 2MB.", ["final_url","page_bytes"])
    add("Orphan-ish (0 inlinks in crawl)", (s("inlinks_internal",0).fillna(0).astype(int) == 0),
        "No internal inlinks discovered within this crawl (limited to crawled set).", ["final_url","inlinks_internal"])

    return items

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Technical SEO Auditor", layout="wide")

st.title("🔎 Technical SEO Auditor")
st.caption("A pragmatic crawler to spot technical SEO issues. All checks run in your session.")

with st.sidebar:
    st.header("Setup")
    start_url = st.text_input("Start URL (include https://)", placeholder="https://www.example.co.uk/")
    max_pages = st.number_input("Maximum pages to crawl", min_value=1, max_value=3000, value=150, step=10)
    obey_robots = st.toggle("Respect robots.txt", value=True, help="Skips URLs disallowed for this user agent.")
    include_query = st.toggle("Include URLs with query strings", value=False)
    sitemap_seed = st.toggle("Use sitemap.xml to seed crawl", value=True, help="If found, uses URLs from sitemap as seeds.")

    st.markdown("---")
    run_single = st.button("Audit single URL")
    run_crawl = st.button("Crawl & Audit")

if (run_single or run_crawl):
    if not start_url or not is_http_url(start_url):
        st.error("Please enter a valid absolute URL starting with http:// or https://")
        st.stop()

    with st.spinner("Running audit…"):
        if run_single:
            resp, content, err, elapsed = fetch_url(start_url)
            audits = [analyse_page(start_url, resp, content, err, elapsed)]
        else:
            audits = crawl(start_url, max_pages=int(max_pages), obey_robots=bool(obey_robots), include_query_urls=bool(include_query), use_sitemap_seed=bool(sitemap_seed))

    if not audits:
        st.warning("No pages were audited. Try increasing the page limit or changing options.")
        st.stop()

    df = audits_to_df(audits)

    # Summary metrics
    issues = summarise_issues(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pages audited", len(df))
    with col2:
        avg_rt = int(df["response_time_ms"].fillna(0).astype(int).mean()) if "response_time_ms" in df else 0
        st.metric("Avg response time (ms)", avg_rt)
    with col3:
        non200 = issues.get("Non-200 status", 0)
        st.metric("Non-200 pages", non200)
    with col4:
        indexable = int(df.get("indexable", pd.Series(dtype=bool)).fillna(False).sum()) if "indexable" in df else 0
        st.metric("Indexable pages", indexable)

    # Issue chips
    st.subheader("Issue overview")
    chips = []
    for k, v in issues.items():
        chips.append(f"{colour_badge(v, good_when='zero')} {k}: {v}")
    st.write("  •  ".join(chips))

    # Issues explorer (affected URLs + per-issue download)
    st.subheader("Issues & affected URLs")
    catalogue = build_issue_catalogue(df)
    any_issues = False
    for item in catalogue:
        count = int(item["mask"].sum())
        if count <= 0:
            continue
        any_issues = True
        with st.expander(f"{item['label']} — {count} pages", expanded=False):
            st.write(item["description"])
            issue_df = _issue_df(df, item["mask"], item["cols"])
            st.dataframe(issue_df, use_container_width=True)
            csv_issue = issue_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download list ({count})",
                data=csv_issue,
                file_name=f"{_slugify(item['label'])}.csv",
                mime="text/csv",
            )
    if not any_issues:
        st.success("No issues detected by the current rules — tidy site! Consider crawling more pages.")

    # Full data table
    st.subheader("Audited pages")
    st.dataframe(df, use_container_width=True)

    # Downloads (global)
    csv = df.to_csv(index=False).encode("utf-8")
    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download CSV (all)", data=csv, file_name="seo_audit.csv", mime="text/csv")
    with colB:
        st.download_button("Download JSON (all)", data=df.to_json(orient="records", indent=2), file_name="seo_audit.json", mime="application/json")

    st.markdown("---")
    st.subheader("Interpretation guide (quick tips)")
    st.markdown(
        """
        - **Status**: Prioritise fixing 4xx/5xx pages. Redirects are fine, but avoid long chains.
        - **Indexable**: Pages with `noindex` (meta or header) will not appear in search results.
        - **Canonical**: Ensure canonical URLs resolve and match the page when appropriate.
        - **Titles & Descriptions**: Succinct, unique, human-friendly. Ranges here are guidelines.
        - **Headings**: One H1 is usually best; add H2s for structure.
        - **Images ALT**: Add descriptive `alt` text; compress large assets.
        - **Mixed content**: Replace `http://` assets with HTTPS equivalents.
        - **Security**: Add `rel="noopener"` to external `_blank` links; consider HSTS for HTTPS.
        - **Performance**: Response time is crude; for CWV use Lighthouse/CrUX.
        - **Orphans**: “Orphan-ish” is relative to the crawl set; full site graph gives the truth.
        """
    )
else:
    st.info("Enter a start URL and choose an option to begin the audit. The tool can crawl or audit a single page.")
