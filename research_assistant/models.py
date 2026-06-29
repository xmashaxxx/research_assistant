"""Shared data models for the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperRecord:
    """Represents a single paper throughout the pipeline lifecycle.

    Fields populated by search:
        arxiv_id, title, authors, abstract, published, pdf_url, arxiv_url

    Fields populated by fetch (None until fetch runs):
        full_text       — body text extracted from the arXiv HTML endpoint,
                          falling back to abstract if HTML is unavailable
        citation_count  — total citations reported by Semantic Scholar
        references      — list of {"paperId": str, "title": str} dicts from S2
    """

    # --- search fields ---
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    arxiv_url: str

    # --- fetch fields (optional until fetch stage runs) ---
    full_text: str | None = None
    citation_count: int | None = None
    references: list[dict] | None = None
