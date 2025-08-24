# Streamlit Technical SEO Auditor
# -----------------------------------------------------------
# Extended to display all issues grouped by type with affected URLs.
# -----------------------------------------------------------

from __future__ import annotations

import json
import re
import time
import typing as t
from collections import deque
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import urllib.robotparser as robotparser

# (config, dataclasses, utility functions, analyse_page, crawl... unchanged)
# -- Code omitted here for brevity (keep from previous file) --

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


def issue_urls(df: pd.DataFrame) -> dict[str, list[str]]:
    problems: dict[str, list[str]] = {}
    if "status" in df:
        bad = df[(df["status"].fillna(0).astype(int) < 200) | (df["status"].fillna(0).astype(int) >= 300)]
        problems["Non-200 status"] = bad["final_url"].tolist()
    if "h1_count" in df:
        bad = df[df["h1_count"].fillna(0).astype(int) != 1]
        problems["Missing or multiple H1"] = bad["final_url"].tolist()
    if "title_length" in df:
        bad = df[(df["title_length"].fillna(0) < TITLE_RECOMMENDED_MIN) | (df["title_length"].fillna(0) > TITLE_RECOMMENDED_MAX)]
        problems["Title too short/long or missing"] = bad["final_url"].tolist()
    if "meta_description_length" in df:
        bad = df[(df["meta_description_length"].fillna(0) < META_DESC_RECOMMENDED_MIN) | (df["meta_description_length"].fillna(0) > META_DESC_RECOMMENDED_MAX)]
        problems["Meta description suboptimal or missing"] = bad["final_url"].tolist()
    if "blocked_by_robots_meta" in df and "blocked_by_xrobots" in df:
        bad = df[(df["blocked_by_robots_meta"].fillna(False)) | (df["blocked_by_xrobots"].fillna(False))]
        problems["Noindex (meta or header)"] = bad["final_url"].tolist()
    if "canonical_is_self" in df:
        bad = df[df["canonical_is_self"].fillna(True) == False]
        problems["Canonical not self-referential"] = bad["final_url"].tolist()
    if "canonical_resolves" in df:
        bad = df[df["canonical_resolves"].fillna(True) == False]
        problems["Canonical broken"] = bad["final_url"].tolist()
    if "images_missing_alt" in df:
        bad = df[df["images_missing_alt"].fillna(0) > 0]
        problems["Images missing ALT"] = bad["final_url"].tolist()
    if "mixed_content" in df:
        bad = df[df["mixed_content"].fillna(0) > 0]
        problems["Mixed content on HTTPS"] = bad["final_url"].tolist()
    if "viewport" in df:
        bad = df[df["viewport"].fillna(False) == False]
        problems["No viewport meta"] = bad["final_url"].tolist()
    return problems

# -------------------------------
# Streamlit UI (extended)
# -------------------------------
# (existing UI unchanged until after chips)

    # Issue chips
    st.subheader("Issue overview")
    chips = []
    for k, v in issues.items():
        chips.append(f"{colour_badge(v, good_when='zero')} {k}: {v}")
    st.write("  •  ".join(chips))

    # Detailed issue breakdown
    st.subheader("Issues by type with affected URLs")
    problems = issue_urls(df)
    for issue, urls in problems.items():
        if not urls:
            continue
        with st.expander(f"{issue} ({len(urls)})"):
            st.write("\n".join(urls))

    # Data table
    st.subheader("Audited pages")
    st.dataframe(df, use_container_width=True)

# (rest unchanged)
