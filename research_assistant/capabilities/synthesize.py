"""Synthesis capability.

v1 implementation will:
- Combine context.comparisons and context.summaries into a synthesis prompt
- Call the Claude API requesting a state-of-the-field narrative that:
    - Answers context.query directly
    - Cites specific papers by arxiv_id for each claim
    - Acknowledges open questions and contradictions from context.comparisons
    - Suggests high-value directions for future work
- Set context.synthesis to the returned narrative string
"""

from research_assistant.context import ResearchContext
from research_assistant.registry import register


@register("synthesize")
def synthesize(context: ResearchContext) -> None:
    raise NotImplementedError("synthesize: not yet implemented")
