"""Integration test for the arXiv search capability.

Hits the live arXiv API — requires network access.
"""

import pytest

from research_assistant.capabilities.search import search_papers
from research_assistant.context import ResearchContext

REQUIRED_KEYS = {"arxiv_id", "title", "authors", "abstract", "published", "pdf_url", "arxiv_url"}


def test_search_returns_papers():
    ctx = ResearchContext(query="retrieval augmented generation")
    search_papers(ctx)

    assert len(ctx.found_papers) > 0, "Expected at least one paper from arXiv"

    for paper in ctx.found_papers:
        missing = REQUIRED_KEYS - paper.keys()
        assert not missing, f"Paper missing keys: {missing}\nPaper: {paper}"
        assert paper["title"] is not None and paper["title"] != "", (
            f"Paper has empty title: {paper}"
        )

    first = ctx.found_papers[0]
    print(f"\n[first result] {first['title']}")
    print(f"  authors : {', '.join(first['authors'])}")
    print(f"  arxiv   : {first['arxiv_url']}")
    print(f"  published: {first['published']}")


def test_search_all_papers_have_arxiv_urls():
    ctx = ResearchContext(query="retrieval augmented generation")
    search_papers(ctx)

    for paper in ctx.found_papers:
        assert paper["arxiv_url"].startswith("https://arxiv.org/abs/"), (
            f"Unexpected arxiv_url: {paper['arxiv_url']}"
        )
        assert paper["pdf_url"].startswith("https://arxiv.org/pdf/"), (
            f"Unexpected pdf_url: {paper['pdf_url']}"
        )


def test_search_empty_query_does_not_crash():
    ctx = ResearchContext(query="")
    search_papers(ctx)
    # arXiv returns either papers or an empty feed — both are acceptable.
    assert isinstance(ctx.found_papers, list)
