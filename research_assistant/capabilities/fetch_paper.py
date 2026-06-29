"""Paper fetch-and-parse capability."""

from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup

from research_assistant.context import ResearchContext
from research_assistant.registry import register

_MAX_TEXT_CHARS = 50_000


def _strip_version(arxiv_id: str) -> str:
    """'2506.06962v3' → '2506.06962' for Semantic Scholar lookups."""
    return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id


def _fetch_full_text(arxiv_id: str) -> str | None:
    """Try the arXiv HTML endpoint; return cleaned body text or None."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "research-assistant/0.1"},
        )
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        container = soup.find("article") or soup.find("body")
        if container is None:
            return None
        text = container.get_text(separator=" ", strip=True)
        return text[:_MAX_TEXT_CHARS] if text else None
    except Exception:
        return None


def _fetch_semantic_scholar(arxiv_id: str) -> tuple[int | None, list | None]:
    """Return (citation_count, references) from Semantic Scholar, or (None, None)."""
    base_id = _strip_version(arxiv_id)
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{base_id}"
    try:
        resp = requests.get(
            url,
            params={"fields": "citationCount,references"},
            timeout=10,
            headers={"User-Agent": "research-assistant/0.1"},
        )
        if resp.status_code == 429:
            print("[fetch] Semantic Scholar rate limit; skipping enrichment.")
            return None, None
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        citation_count = data.get("citationCount")
        refs = [
            {"paperId": r.get("paperId"), "title": r.get("title")}
            for r in (data.get("references") or [])
            if r.get("title")
        ]
        return citation_count, refs or None
    except Exception:
        return None, None


@register("fetch")
def fetch_papers(context: ResearchContext) -> None:
    """Fetch full text and semantic metadata for each paper in context.found_papers.

    For each PaperRecord in context.found_papers:
    - Requests the arXiv HTML endpoint; falls back to the abstract on failure
    - Queries Semantic Scholar for citation count and reference list
    - Mutates the PaperRecord with full_text, citation_count, references
    - Writes context.summaries[arxiv_id] = {"raw_text": ..., "metadata": {...}}
      so the extract stage has a single place to read from
    """
    papers = context.found_papers
    print(f"[fetch] Processing {len(papers)} papers...")

    for i, paper in enumerate(papers, 1):
        arxiv_id = paper.arxiv_id
        short_title = paper.title[:55] + "..." if len(paper.title) > 55 else paper.title
        print(f"[fetch] ({i}/{len(papers)}) {arxiv_id}: {short_title}")

        full_text = _fetch_full_text(arxiv_id)
        if full_text:
            print(f"[fetch]   HTML text: {len(full_text):,} chars")
        else:
            full_text = paper.abstract
            print(f"[fetch]   HTML unavailable — using abstract ({len(full_text):,} chars)")

        citation_count, references = _fetch_semantic_scholar(arxiv_id)
        if citation_count is not None:
            print(f"[fetch]   citations: {citation_count}, references: {len(references or [])}")
        else:
            print("[fetch]   Semantic Scholar: no data")

        paper.full_text = full_text
        paper.citation_count = citation_count
        paper.references = references

        context.summaries[arxiv_id] = {
            "raw_text": full_text,
            "metadata": {
                "citation_count": citation_count,
                "references": references,
            },
        }

        # Avoid hammering Semantic Scholar (unauthenticated limit ~100 req/5 min).
        if i < len(papers):
            time.sleep(1)

    print(f"[fetch] Done. {len(context.summaries)} papers in context.summaries.")
