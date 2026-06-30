"""arXiv search capability."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import anthropic
import requests

from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord
from research_assistant.registry import register

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


@register("search")
def search_papers(context: ResearchContext) -> None:
    """Query arXiv and populate context.found_papers with PaperRecord objects.

    In general-query mode (context.query set), uses the query string directly.
    In project mode (context.project_description set), distills the description
    into compact search terms via Claude before querying arXiv.
    """
    if context.project_description:
        print("[search] Project description mode — distilling search terms...")
        query = _build_search_query_from_project(context.project_description)
        print(f"[search] Derived search terms: {query}")
    else:
        query = context.query
    print(f"[search] Querying arXiv for: {query}")

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
        context.found_papers = []
        return
    except requests.HTTPError as exc:
        print(f"[search] Warning: HTTP error from arXiv: {exc}")
        context.found_papers = []
        return
    except requests.RequestException as exc:
        print(f"[search] Warning: network error: {exc}")
        context.found_papers = []
        return

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        print(f"[search] Warning: failed to parse arXiv XML: {exc}")
        context.found_papers = []
        return

    papers = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        # The <id> element is the canonical arXiv URL; extract just the ID portion.
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

    context.found_papers = papers
    print(f"[search] Found {len(papers)} papers.")
