"""Schema-driven extraction capability.

v1 implementation will:
- Load the active JSON schema from research_assistant/schemas/<context.extraction_schema>.json
- For each paper in context.summaries, prompt the Claude API with the raw text
  and the schema fields, requesting structured JSON output
- Parse and validate the Claude response against the schema
- Merge structured extraction results back into context.summaries[arxiv_id]:
    context.summaries[arxiv_id]["extracted"] = {schema_field: value, ...}
- Default schema: general_cs_paper (research_question, method, dataset, key_results, limitations)
- v2 slot: domain-specific schemas (clinical PICO, social science, economics)
"""

from research_assistant.context import ResearchContext
from research_assistant.registry import register


@register("extract")
def extract_summaries(context: ResearchContext) -> None:
    raise NotImplementedError("extract: not yet implemented")
