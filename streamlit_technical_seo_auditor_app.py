# Streamlit Technical SEO Auditor
# -----------------------------------------------------------
# A single-file Streamlit app for lightweight technical SEO audits.
# It crawls a site (respecting robots.txt if you choose), analyses pages,
# and reports on common technical SEO issues.
#
# ▶ Run locally:
#     pip install streamlit requests beautifulsoup4 pandas lxml
#     streamlit run streamlit_technical_seo_auditor_app.py
# -----------------------------------------------------------

from __future__ import annotations

import json
import re
import time
import typing as t
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import urllib.robotparser as robotparser

# -------------------------------
# Configuration
# -------------------------------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 SEO-Auditor/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

REQUEST_TIMEOUT = 20  # seconds

TITLE_RECOMMENDED_MIN = 30
TITLE_RECOMMENDED_MAX = 60
META_DESC_RECOMMENDED_MIN = 70
META_DESC_RECOMMENDED_MAX = 160

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

    notes: str


# -------------------------------
# Utility functions
# -------------------------------

def normalise_url(u: str) -> str:
    """Normalise a URL for consistent comparison.
    - Lowercase scheme/host, strip fragments, remove default ports, collapse //
    - Remove trailing slash except for root
    """
    parsed = urlparse(u)
    scheme = (parsed.scheme or "https").lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Remove default ports
    if (":" in netloc):
        host, _, port = netloc.partition(":")
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host
    path = re.sub(r"//+", "/", parsed.path or "/")
    # Remove trailing slash except for root
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    # Drop fragment
    return urlunparse((scheme, netloc, path, "", parsed.query or "", ""))


def same_host(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    ha = pa.netloc.lower().lstrip("www.")
    hb = pb.netloc.lower().lstrip("www.")
    return ha == hb


def is_http_url(u: str) -> bool:
    try:
        return urlparse(u).scheme in {"http", "https"}
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def fetch_url(url: str) -> tuple[requests.Response | None, bytes | None, str | None, float]:
    """Fetch a URL and return (response, content, error, elapsed_seconds)."""
    start = time.time()
    err = None
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
        # If robots can't be read, default to allowing
        pass
    return rp


@st.cache_data(show_spinner=False)
def parse_sitemap_urls(base_url: str) -> list[str]:
    """Attempt to find and parse sitemap.xml to seed the crawl.
    Supports simple sitemap index and urlset.
    """
    urls: list[str] = []
    parsed = urlparse(base_url)
    sitemap_url = urlunparse((parsed.scheme or "https", parsed.netloc, "/sitemap.xml", "", "", ""))

    resp, content, err, _ = fetch_url(sitemap_url)
    if err or not resp or resp.status_code >= 400:
        return urls
    try:
        tree = ET.fromstring(content)
        ns = {
            "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "xhtml": "http://www.w3.org/1999/xhtml",
            "image": "http://www.google.com/schemas/sitemap-image/1.1",
        }
        if tree.tag.endswith("sitemapindex"):
            for smap in tree.findall("sm:sitemap", ns):
                loc_el = smap.find("sm:loc", ns)
                if loc_el is not None and loc_el.text:
                    child_url = loc_el.text.strip()
                    # Fetch child sitemap
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

def analyse_page(url: str, resp: requests.Response | None, content: bytes | None, err: str | None, elapsed: float) -> PageAudit:
    status = None if not resp else resp.status_code
    redirected = bool(resp.history) if resp else False
    final_url = resp.url if resp else url
    response_time_ms = int(elapsed * 1000)
    page_bytes = len(content) if content else None

    title = None
    meta_description = None
    meta_robots = None
    canonical = None
    h1_count = h2_count = None
    images = images_missing_alt = None
    internal_links = external_links = nofollow_links = 0
    hreflang_count = 0
    structured_types: list[str] = []
    html_lang = None
    viewport = False
    https = urlparse(final_url).scheme == "https"
    mixed_content = 0

    x_robots = None if not resp else resp.headers.get("X-Robots-Tag")

    notes: list[str] = []

    if err:
        notes.append(f"Fetch error: {err}")

    if content and resp and resp.headers.get("Content-Type", "").startswith("text/html"):
        soup = BeautifulSoup(content, "lxml")

        # Language & viewport
        html = soup.find("html")
        if html and html.has_attr("lang"):
            html_lang = (html.get("lang") or "").strip()
        if soup.find("meta", attrs={"name": "viewport"}):
            viewport = True

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

        # Canonical
        can = soup.find("link", rel=lambda x: x and "canonical" in [v.lower() for v in (x if isinstance(x, list) else [x])])
        if can and can.has_attr("href"):
            canonical = can["href"].strip()

        # Headings
        h1_count = len(soup.find_all("h1"))
        h2_count = len(soup.find_all("h2"))

        # Images
        imgs = soup.find_all("img")
        images = len(imgs)
        images_missing_alt = sum(1 for i in imgs if not i.get("alt"))

        # Links
        final_host = urlparse(final_url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("mailto:") or href.startswith("tel:"):
                continue
            full = urljoin(final_url, href)
            if not is_http_url(full):
                continue
            if urlparse(full).netloc == final_host:
                internal_links += 1
            else:
                external_links += 1
            rel = (a.get("rel") or [])
            if any(r.lower() == "nofollow" for r in rel) or ("nofollow" in (a.get("rel") or "")):
                nofollow_links += 1

        # Hreflang
        for link in soup.find_all("link", rel=True, href=True):
            rel = link.get("rel")
            rels = [r.lower() for r in (rel if isinstance(rel, list) else [rel])]
            if "alternate" in rels and link.get("hreflang"):
                hreflang_count += 1

        # Structured data types
        for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
            try:
                data = json.loads(script.text)
                # Could be list or object
                if isinstance(data, list):
                    structured_types.extend([d.get("@type") for d in data if isinstance(d, dict) and d.get("@type")])
                elif isinstance(data, dict) and data.get("@type"):
                    structured_types.append(data.get("@type"))
            except Exception:
                continue

        # Mixed content check: http resources on https page
        if https:
            # Look for http src/href
            mixed_content = 0
            for tag in soup.find_all(src=True):
                src = tag.get("src")
                if isinstance(src, str) and src.startswith("http://"):
                    mixed_content += 1
            for tag in soup.find_all(href=True):
                href = tag.get("href")
                if isinstance(href, str) and href.startswith("http://"):
                    mixed_content += 1

    # Derived fields
    title_length = len(title) if title else None
    meta_description_length = len(meta_description) if meta_description else None

    blocked_by_robots_meta = bool(meta_robots and re.search(r"noindex", meta_robots, re.I))
    blocked_by_xrobots = bool(x_robots and re.search(r"noindex", x_robots, re.I))

    # Canonical checks
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

    # Indexability (simplified): 2xx status, not blocked by robots meta/header
    indexable = None
    if status is not None:
        indexable = (200 <= status < 300) and not blocked_by_robots_meta and not blocked_by_xrobots

    # Notes aggregation
    if title_length is not None and (title_length < TITLE_RECOMMENDED_MIN or title_length > TITLE_RECOMMENDED_MAX):
        notes.append("Title length outside typical range")
    if meta_description and (meta_description_length < META_DESC_RECOMMENDED_MIN or meta_description_length > META_DESC_RECOMMENDED_MAX):
        notes.append("Meta description length outside typical range")
    if h1_count == 0:
        notes.append("Missing H1 heading")
    if images_missing_alt and images_missing_alt > 0:
        notes.append("Images missing ALT text present")
    if mixed_content and mixed_content > 0:
        notes.append("Mixed content (HTTP resources on HTTPS page)")

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
        notes="; ".join(notes),
    )


# -------------------------------
# Crawler
# -------------------------------

def crawl(start_url: str, max_pages: int = 50, obey_robots: bool = True, include_query_urls: bool = False, use_sitemap_seed: bool = True) -> list[PageAudit]:
    base = normalise_url(start_url)
    base_parsed = urlparse(base)
    base_root = urlunparse((base_parsed.scheme, base_parsed.netloc, "/", "", "", ""))

    rp = read_robots_txt(base)

    visited: set[str] = set()
    audits: list[PageAudit] = []
    q: deque[str] = deque()

    seeds: list[str] = []
    if use_sitemap_seed:
        seeds = [u for u in parse_sitemap_urls(base) if is_http_url(u)]
    if not seeds:
        seeds = [base]

    # Start queue
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
            # Skip URLs disallowed by robots.txt
            visited.add(current)
            continue

        visited.add(current)

        # Fetch
        resp, content, err, elapsed = fetch_url(current)
        audit = analyse_page(current, resp, content, err, elapsed)
        audits.append(audit)
        processed += 1
        pbar.progress(min(int(processed / max_pages * 100), 100), text=f"Crawling… {processed}/{max_pages}")

        # Enqueue new links from this page
        if resp and content and resp.headers.get("Content-Type", "").startswith("text/html"):
            soup = BeautifulSoup(content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("#"):
                    continue
                if href.startswith("mailto:") or href.startswith("tel:"):
                    continue
                full = urljoin(audit.final_url, href)
                # Normalise and filter
                if is_http_url(full):
                    n = normalise_url(full)
                    if same_host(n, base) and n not in visited:
                        if include_query_urls or not urlparse(n).query:
                            q.append(n)

    pbar.progress(100, text="Crawl complete")

    return audits


# -------------------------------
# Reporting helpers
# -------------------------------

def audits_to_df(audits: list[PageAudit]) -> pd.DataFrame:
    rows = [asdict(a) for a in audits]
    df = pd.DataFrame(rows)
    # Order columns for clarity
    cols_order = [
        "url", "final_url", "status", "redirected", "indexable", "response_time_ms", "page_bytes",
        "title", "title_length", "meta_description", "meta_description_length",
        "meta_robots", "x_robots",
        "canonical", "canonical_resolves", "canonical_is_self",
        "h1_count", "h2_count",
        "images", "images_missing_alt",
        "internal_links", "external_links", "nofollow_links",
        "hreflang_count", "structured_data_types",
        "html_lang", "viewport",
        "https", "mixed_content",
        "blocked_by_robots_meta", "blocked_by_xrobots",
        "notes",
    ]
    existing_cols = [c for c in cols_order if c in df.columns]
    return df[existing_cols]


def summarise_issues(df: pd.DataFrame) -> dict[str, int]:
    issues = {
        "Non-200 status": int(((df["status"].fillna(0).astype(int) < 200) | (df["status"].fillna(0).astype(int) >= 300)).sum()) if "status" in df else 0,
        "Missing or multiple H1": int(((df["h1_count"].fillna(0).astype(int) != 1)).sum()) if "h1_count" in df else 0,
        "Title too short/long or missing": int(((df["title_length"].fillna(0) < TITLE_RECOMMENDED_MIN) | (df["title_length"].fillna(0) > TITLE_RECOMMENDED_MAX)).sum()) if "title_length" in df else 0,
        "Meta description suboptimal or missing": int(((df["meta_description_length"].fillna(0) < META_DESC_RECOMMENDED_MIN) | (df["meta_description_length"].fillna(0) > META_DESC_RECOMMENDED_MAX)).sum()) if "meta_description_length" in df else 0,
        "Noindex (meta or header)": int(((df["blocked_by_robots_meta"].fillna(False)) | (df["blocked_by_xrobots"].fillna(False))).sum()) if "blocked_by_robots_meta" in df and "blocked_by_xrobots" in df else 0,
        "Canonical not self-referential": int(((df["canonical_is_self"].fillna(True) == False)).sum()) if "canonical_is_self" in df else 0,
        "Canonical broken": int(((df["canonical_resolves"].fillna(True) == False)).sum()) if "canonical_resolves" in df else 0,
        "Images missing ALT": int(((df["images_missing_alt"].fillna(0) > 0)).sum()) if "images_missing_alt" in df else 0,
        "Mixed content on HTTPS": int(((df["mixed_content"].fillna(0) > 0)).sum()) if "mixed_content" in df else 0,
        "No viewport meta": int(((df["viewport"].fillna(False) == False)).sum()) if "viewport" in df else 0,
    }
    return issues


def colour_badge(v: bool | int | None, good_when: str = "true") -> str:
    """Return an emoji badge based on boolean/int quality."""
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


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Technical SEO Auditor", layout="wide")

st.title("🔎 Technical SEO Auditor")
st.caption("A quick, pragmatic crawler to spot common technical SEO issues. All checks run in your local session.")

with st.sidebar:
    st.header("Setup")
    start_url = st.text_input("Start URL (include https://)", placeholder="https://www.example.co.uk/")
    max_pages = st.number_input("Maximum pages to crawl", min_value=1, max_value=2000, value=100, step=10)
    obey_robots = st.toggle("Respect robots.txt", value=True, help="Skips URLs disallowed for this user agent.")
    include_query = st.toggle("Include URLs with query strings", value=False)
    sitemap_seed = st.toggle("Use sitemap.xml to seed crawl", value=True, help="If found, uses URLs from sitemap as seeds.")

    st.markdown("---")
    run_single = st.button("Audit single URL")
    run_crawl = st.button("Crawl & Audit")

placeholder_summary = st.empty()
placeholder_table = st.empty()
placeholder_download = st.empty()

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

    # Data table
    st.subheader("Audited pages")
    st.dataframe(df, use_container_width=True)

    # Downloads
    csv = df.to_csv(index=False).encode("utf-8")
    xlsx = None
    try:
        xlsx = df.to_excel(index=False, engine="openpyxl")  # Will raise if openpyxl missing
    except Exception:
        pass

    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download CSV", data=csv, file_name="seo_audit.csv", mime="text/csv")
    with colB:
        st.download_button("Download JSON", data=df.to_json(orient="records", indent=2), file_name="seo_audit.json", mime="application/json")

    st.markdown("---")
    st.subheader("Interpretation guide (quick tips)")
    st.markdown(
        """
        - **Status**: Prioritise fixing 4xx/5xx pages. Redirects are fine, but avoid long chains.
        - **Indexable**: Pages with `noindex` (meta or header) will not appear in search results.
        - **Canonical**: Ensure canonical URLs resolve and match the page when appropriate (self-referential).
        - **Titles & Descriptions**: Aim for succinct, unique, human-friendly copy. Typical ranges shown here are guidelines, not laws.
        - **Headings**: Exactly one H1 per page is usually best for clarity.
        - **Images ALT**: Add descriptive `alt` text for accessibility and SEO.
        - **Mixed content**: Replace `http://` assets with HTTPS equivalents.
        - **Performance**: Response time is a crude proxy; use Lighthouse or the Chrome UX Report for Core Web Vitals.
        """
    )

else:
    st.info("Enter a start URL and choose an option to begin the audit. The tool can crawl or audit a single page.")
