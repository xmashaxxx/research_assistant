"""arXiv search capability, combined multi-source search, and merge utilities."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import anthropic
import requests

from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord
from research_assistant.registry import register

# Matches both modern arXiv IDs (YYMM.NNNNN[vN]) and old-style (subj/YYYYMMNN).
_ARXIV_ID_RE = re.compile(
    r"^\d{4}\.\d{4,5}(v\d+)?$"
    r"|^[a-z][a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$"
)

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_HAIKU = "claude-haiku-4-5-20251001"


def _build_search_query_from_project(description: str) -> str:
    """Distill a project description into compact arXiv search terms using Claude Haiku.

    A raw pasted abstract is too long and noisy to use as a literal arXiv query.
    This helper extracts the 4-6 most discriminative keywords/phrases and returns
    them as a short search string. Falls back to the first 120 chars of the
    description if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return description[:120]

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=_HAIKU,
        max_tokens=60,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract 4-6 keywords or short phrases for an arXiv paper search "
                    "from the project description below. Return ONLY the search terms "
                    "as a single line with spaces between them — no explanation, no "
                    "punctuation, no bullet points.\n\n"
                    f"Project description:\n{description}"
                ),
            }
        ],
    )
    return response.content[0].text.strip()


def _text(entry: ET.Element, tag: str) -> str:
    el = entry.find(f"{{{_ATOM_NS}}}{tag}")
    return " ".join((el.text or "").split()) if el is not None else ""


def _normalize_title(title: str) -> str:
    """Lowercase, strip non-alphanumeric, collapse whitespace for fuzzy dedup."""
    t = re.sub(r"[^a-z0-9 ]", "", title.lower())
    return re.sub(r" +", " ", t).strip()


def _is_arxiv_id(id_: str) -> bool:
    """Return True if the string looks like a real arXiv ID (not an S2 hash)."""
    return bool(_ARXIV_ID_RE.match(id_))


def _search_arxiv(query: str) -> list[PaperRecord]:
    """Query arXiv Atom API and return a list of PaperRecord objects."""
    try:
        response = requests.get(
            _ARXIV_API,
            params={
                "search_query": f"all:{query}",
                "max_results": 10,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=15,
        )
        response.raise_for_status()
    except requests.Timeout:
        print("[search] Warning: arXiv request timed out.")
        return []
    except requests.HTTPError as exc:
        print(f"[search] Warning: HTTP error from arXiv: {exc}")
        return []
    except requests.RequestException as exc:
        print(f"[search] Warning: network error: {exc}")
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        print(f"[search] Warning: failed to parse arXiv XML: {exc}")
        return []

    papers = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        raw_id = _text(entry, "id")
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

        authors = [
            " ".join((name.text or "").split())
            for author in entry.findall(f"{{{_ATOM_NS}}}author")
            if (name := author.find(f"{{{_ATOM_NS}}}name")) is not None
        ]

        categories = [
            el.get("term", "")
            for el in entry.findall(f"{{{_ATOM_NS}}}category")
            if el.get("term")
        ]

        pc_el = entry.find(f"{{{_ARXIV_NS}}}primary_category")
        primary_category = (
            pc_el.get("term", "") if pc_el is not None
            else (categories[0] if categories else "")
        )

        papers.append(
            PaperRecord(
                arxiv_id=arxiv_id,
                title=_text(entry, "title"),
                authors=authors,
                abstract=_text(entry, "summary"),
                published=_text(entry, "published"),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                categories=categories,
                primary_category=primary_category,
                source="arxiv",
            )
        )

    return papers


def _merge_results(
    arxiv: list[PaperRecord],
    s2: list[PaperRecord],
) -> list[PaperRecord]:
    """Merge arXiv and S2 results, deduplicating by arXiv ID then title.

    Strategy:
    - arXiv results are kept as-is and come first.
    - S2 results are appended only if they are not already represented in
      the arXiv set (matched by arXiv ID or by normalized title).
    - S2 papers that do not have a real arXiv ID are excluded: the current
      fetch stage requires arXiv IDs to retrieve full text.
    """
    seen_ids: set[str] = {p.arxiv_id for p in arxiv}
    seen_titles: set[str] = {_normalize_title(p.title) for p in arxiv}

    s2_additions: list[PaperRecord] = []
    for p in s2:
        if not _is_arxiv_id(p.arxiv_id):
            continue  # S2-only paper — fetch stage cannot handle non-arXiv IDs
        if p.arxiv_id in seen_ids:
            continue  # already in arXiv results
        norm = _normalize_title(p.title)
        if norm in seen_titles:
            continue  # same paper returned by arXiv under a matching title
        seen_ids.add(p.arxiv_id)
        seen_titles.add(norm)
        s2_additions.append(p)

    return arxiv + s2_additions


def search_all(query: str) -> list[PaperRecord]:
    """Search both arXiv and Semantic Scholar, merge, and deduplicate.

    arXiv results appear first; S2-exclusive papers (identified by arXiv ID
    cross-reference) are appended after deduplication. Papers found in both
    sources retain the arXiv version.

    Args:
        query: Search query string (already distilled to keywords if needed).

    Returns a combined, deduplicated list of PaperRecord objects.
    """
    from research_assistant.capabilities.search_semantic_scholar import (
        search_semantic_scholar,
    )

    print(f"[search] Querying arXiv for: {query}")
    arxiv_results = _search_arxiv(query)
    print(f"[search] arXiv: {len(arxiv_results)} results.")

    print(f"[search] Querying Semantic Scholar for: {query}")
    s2_results = search_semantic_scholar(query)
    print(f"[search] S2: {len(s2_results)} results.")

    merged = _merge_results(arxiv_results, s2_results)
    s2_added = len(merged) - len(arxiv_results)
    print(
        f"[search] Merged: {len(merged)} total "
        f"({len(arxiv_results)} arXiv + {s2_added} S2-exclusive)."
    )
    return merged


@register("search")
def search_papers(context: ResearchContext) -> None:
    """Search arXiv and Semantic Scholar and populate context.found_papers.

    In general-query mode (context.query set), uses the query string directly.
    In project mode (context.project_description set), distills the description
    into compact search terms via Claude before querying both sources.
    """
    if context.project_description:
        print("[search] Project description mode — distilling search terms...")
        query = _build_search_query_from_project(context.project_description)
        print(f"[search] Derived search terms: {query}")
    else:
        query = context.query

    context.found_papers = search_all(query)
    print(f"[search] Found {len(context.found_papers)} papers total.")
