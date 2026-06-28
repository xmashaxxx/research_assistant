"""Paper fetch-and-parse capability.

v1 implementation will:
- Iterate over context.found_papers
- For each paper, fetch the HTML abstract page from arXiv
- Attempt PDF text extraction via the arXiv e-print endpoint
- Fall back to abstract-only if PDF extraction fails
- Enrich metadata via the Semantic Scholar API (citation count, reference list)
- Store raw text in context.summaries keyed by arxiv_id:
    context.summaries[arxiv_id] = {"raw_text": str, "metadata": dict}
"""

from research_assistant.context import ResearchContext
from research_assistant.registry import register


@register("fetch")
def fetch_papers(context: ResearchContext) -> None:
    raise NotImplementedError("fetch: not yet implemented")
