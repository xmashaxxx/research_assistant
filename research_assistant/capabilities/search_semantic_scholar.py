"""Semantic Scholar search capability."""

from __future__ import annotations

import time

import requests

from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord
from research_assistant.registry import register

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = (
    "paperId,title,authors,abstract,year,"
    "externalIds,publicationDate,openAccessPdf,citationCount"
)


def _s2_request(params: dict) -> dict | None:
    """Make one S2 API request; on 429 wait 5 s and retry once."""
    for attempt in range(2):
        if attempt:
            time.sleep(5)
        try:
            resp = requests.get(_S2_API, params=params, timeout=15)
        except requests.RequestException as exc:
            print(f"[search_s2] Network error: {exc}")
            return None

        if resp.status_code == 429:
            print("[search_s2] Rate-limited (429) — waiting 5 s before retry.")
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"[search_s2] HTTP error: {exc}")
            return None

        try:
            return resp.json()
        except ValueError as exc:
            print(f"[search_s2] JSON parse error: {exc}")
            return None

    print("[search_s2] Giving up after rate-limit retry.")
    return None


def search_semantic_scholar(
    query: str,
    max_results: int = 10,
) -> list[PaperRecord]:
    """Search Semantic Scholar and return a list of PaperRecord objects.

    Papers that are cross-listed on arXiv use the arXiv ID and standard arXiv
    URLs. Papers available only via Semantic Scholar use the S2 paperId as the
    identifier and openAccessPdf.url (if present) as the PDF URL.

    Args:
        query:       Free-text search query.
        max_results: Maximum number of results to return (default 10).

    Returns a list of PaperRecord with source="semantic_scholar".
    """
    if not query.strip():
        return []

    data = _s2_request(
        {
            "query": query,
            "limit": max_results,
            "fields": _S2_FIELDS,
        }
    )
    if data is None:
        return []

    records = []
    for item in data.get("data") or []:
        paper_id = item.get("paperId") or ""
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()

        if not title:
            continue

        authors = [
            a.get("name", "").strip()
            for a in (item.get("authors") or [])
            if a.get("name")
        ]

        # Prefer publicationDate; fall back to year integer.
        pub_date = item.get("publicationDate") or ""
        if not pub_date and item.get("year"):
            pub_date = str(item["year"])

        external_ids = item.get("externalIds") or {}
        arxiv_id_raw = external_ids.get("ArXiv")

        if arxiv_id_raw:
            # Paper is on arXiv — use the canonical arXiv ID and URLs.
            arxiv_id = arxiv_id_raw.split("v")[0]  # strip version suffix
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            # S2-only paper — use the S2 paperId; PDF from openAccessPdf if available.
            arxiv_id = paper_id
            oa_pdf = item.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url") or ""
            arxiv_url = f"https://www.semanticscholar.org/paper/{paper_id}"

        citation_count = item.get("citationCount")

        records.append(
            PaperRecord(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=pub_date,
                pdf_url=pdf_url,
                arxiv_url=arxiv_url,
                categories=[],
                primary_category="",
                source="semantic_scholar",
                citation_count=citation_count if isinstance(citation_count, int) else None,
            )
        )

    return records


@register("search_s2")
def search_s2_papers(context: ResearchContext) -> None:
    """Search Semantic Scholar and populate context.found_papers.

    Uses context.query in general mode, or derives search terms from
    context.project_description in project mode (same distillation as the
    arXiv search capability).
    """
    if context.project_description:
        # Reuse the same query-distillation helper from search.py to keep
        # search terms consistent across both sources.
        from research_assistant.capabilities.search import (
            _build_search_query_from_project,
        )
        query = _build_search_query_from_project(context.project_description)
        print(f"[search_s2] Derived search terms: {query}")
    else:
        query = context.query

    print(f"[search_s2] Querying Semantic Scholar for: {query}")
    results = search_semantic_scholar(query)
    context.found_papers = results
    print(f"[search_s2] Found {len(results)} papers.")
