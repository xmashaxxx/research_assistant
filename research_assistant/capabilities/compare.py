"""Cross-paper comparison capability — uses Claude tool-use for structured output."""

from __future__ import annotations

import os

import anthropic

from research_assistant.context import ResearchContext
from research_assistant.registry import register

_MODEL = "claude-haiku-4-5-20251001"

_COMPARE_TOOL: dict = {
    "name": "compare_papers",
    "description": (
        "Compare a set of research papers based on their extracted fields. "
        "Identify where papers agree, where they differ (naming which paper "
        "holds which position), and what each paper uniquely contributes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agreements": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Points or claims where two or more papers explicitly align. "
                    "Each entry is a concise statement of the shared position."
                ),
            },
            "disagreements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {
                            "type": "string",
                            "description": "The topic or dimension on which the papers differ.",
                        },
                        "positions": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": (
                                "Maps each arxiv_id to the position or stance that paper takes "
                                "on this point. Include only papers that address the point."
                            ),
                        },
                    },
                    "required": ["point", "positions"],
                },
                "description": (
                    "Points where papers take different positions. Each entry names the "
                    "topic and maps arxiv_id → the stance that paper holds."
                ),
            },
            "unique_contributions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": (
                    "Maps each arxiv_id to a concise statement of what that paper "
                    "contributes that is not addressed by the other papers."
                ),
            },
        },
        "required": ["agreements", "disagreements", "unique_contributions"],
    },
}


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the compare capability."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_compare_prompt(
    extractions: dict[str, dict],
    titles: dict[str, str] | None,
) -> str:
    """Render each paper's extraction as a labelled block for Claude."""
    blocks = []
    for arxiv_id, ext in extractions.items():
        label = (titles or {}).get(arxiv_id, arxiv_id)
        lines = [f"=== {label} (arXiv: {arxiv_id}) ==="]
        for field in ("research_question", "method", "dataset", "key_results", "limitations"):
            value = ext.get(field)
            if value:
                lines.append(f"{field}: {value}")
        related = ext.get("related_work")
        if related:
            lines.append(f"related_work: {', '.join(related)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def compare_extractions(
    extractions: dict[str, dict],
    titles: dict[str, str] | None = None,
) -> dict:
    """Compare structured extractions from two or more papers using Claude tool-use.

    Args:
        extractions: dict mapping arxiv_id → extraction dict (output of extract_paper).
        titles:      optional dict mapping arxiv_id → human-readable title for labelling.

    Returns a dict with keys:
        agreements            — list[str]
        disagreements         — list[{"point": str, "positions": {arxiv_id: str}}]
        unique_contributions  — {arxiv_id: str}
    """
    if len(extractions) < 2:
        raise ValueError(
            f"compare_extractions requires at least 2 papers; got {len(extractions)}."
        )

    client = _get_client()
    paper_block = _build_compare_prompt(extractions, titles)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        tools=[_COMPARE_TOOL],
        tool_choice={"type": "tool", "name": "compare_papers"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Compare the following research papers using the compare_papers tool. "
                    "Use the arxiv_id values exactly as shown when populating "
                    "disagreements.positions and unique_contributions keys.\n\n"
                    + paper_block
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "compare_papers":
            return block.input

    raise RuntimeError(
        f"Claude did not return a compare_papers tool_use block. "
        f"Stop reason: {response.stop_reason}. Content: {response.content}"
    )


@register("compare")
def compare_papers(context: ResearchContext) -> None:
    """Compare all extracted papers in context.summaries.

    Reads context.summaries[arxiv_id]["extraction"] for each paper,
    calls compare_extractions(), and writes the result to
    context.comparisons.
    """
    extractions = {
        arxiv_id: entry["extraction"]
        for arxiv_id, entry in context.summaries.items()
        if "extraction" in entry
    }

    if len(extractions) < 2:
        print(
            f"[compare] Only {len(extractions)} paper(s) with extractions — "
            "need at least 2 to compare. Skipping."
        )
        return

    # Build title map from found_papers for readable prompt labels.
    titles = {p.arxiv_id: p.title for p in context.found_papers}

    print(f"[compare] Comparing {len(extractions)} papers...")
    result = compare_extractions(extractions, titles)
    context.comparisons = result
    print(
        f"[compare] Done — {len(result.get('agreements', []))} agreements, "
        f"{len(result.get('disagreements', []))} disagreements, "
        f"{len(result.get('unique_contributions', {}))} unique contributions."
    )
