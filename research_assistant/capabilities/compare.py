"""Cross-paper comparison capability.

v1 implementation will:
- Collect all extracted summaries from context.summaries
- Build a structured prompt presenting each paper's extracted fields side-by-side
- Call the Claude API requesting a structured cross-paper comparison:
    - agreements: claims supported by multiple papers
    - contradictions: conflicting findings or interpretations
    - gaps: questions raised but not answered by the corpus
- Populate context.comparisons:
    {"agreements": [...], "contradictions": [...], "gaps": [...]}
"""

from research_assistant.context import ResearchContext
from research_assistant.registry import register


@register("compare")
def compare_papers(context: ResearchContext) -> None:
    raise NotImplementedError("compare: not yet implemented")
