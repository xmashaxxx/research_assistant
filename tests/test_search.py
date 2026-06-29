"""Integration test for the arXiv search capability.

Hits the live arXiv API — requires network access.
"""

import dataclasses

from research_assistant.capabilities.search import search_papers
from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord

SEARCH_FIELDS = {f.name for f in dataclasses.fields(PaperRecord) if f.default is dataclasses.MISSING}


def test_search_returns_paper_records():
    ctx = ResearchContext(query="retrieval augmented generation")
    search_papers(ctx)

    assert len(ctx.found_papers) > 0, "Expected at least one paper from arXiv"

    for paper in ctx.found_papers:
        assert isinstance(paper, PaperRecord), f"Expected PaperRecord, got {type(paper)}"
        assert paper.title, f"Paper has empty title: {paper}"
        assert paper.arxiv_id, f"Paper has empty arxiv_id: {paper}"

    first = ctx.found_papers[0]
    print(f"\n[first result] {first.title}")
    print(f"  authors  : {', '.join(first.authors)}")
    print(f"  arxiv    : {first.arxiv_url}")
    print(f"  published: {first.published}")


def test_search_all_papers_have_valid_urls():
    ctx = ResearchContext(query="retrieval augmented generation")
    search_papers(ctx)

    for paper in ctx.found_papers:
        assert paper.arxiv_url.startswith("https://arxiv.org/abs/"), (
            f"Unexpected arxiv_url: {paper.arxiv_url}"
        )
        assert paper.pdf_url.startswith("https://arxiv.org/pdf/"), (
            f"Unexpected pdf_url: {paper.pdf_url}"
        )


def test_search_fetch_fields_are_unpopulated():
    """Fetch-stage fields must be None after search — fetch populates them."""
    ctx = ResearchContext(query="retrieval augmented generation")
    search_papers(ctx)

    for paper in ctx.found_papers:
        assert paper.full_text is None
        assert paper.citation_count is None
        assert paper.references is None


def test_search_empty_query_does_not_crash():
    ctx = ResearchContext(query="")
    search_papers(ctx)
    assert isinstance(ctx.found_papers, list)
