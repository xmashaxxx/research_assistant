"""arXiv search capability.

v1 implementation will:
- Accept context.query and optional category filters
- Call the arXiv API (http://export.arxiv.org/api/query)
- Parse the Atom feed with BeautifulSoup
- Populate context.found_papers with a list of dicts:
    {arxiv_id, title, authors, abstract, published, url}
- Respect a configurable max_results limit (default 20)
"""

from research_assistant.context import ResearchContext
from research_assistant.registry import register


@register("search")
def search_papers(context: ResearchContext) -> None:
    raise NotImplementedError("search: not yet implemented")
