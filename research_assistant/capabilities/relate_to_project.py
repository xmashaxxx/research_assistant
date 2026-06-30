"""Relate-to-project capability — maps retrieved papers onto a user's own project."""

from __future__ import annotations

import os

import anthropic

from research_assistant.context import ResearchContext
from research_assistant.registry import register

_MODEL = "claude-haiku-4-5-20251001"

_RELATE_TOOL: dict = {
    "name": "find_related_papers",
    "description": (
        "Given a user's project description and a set of retrieved papers, "
        "identify which papers are most relevant to the project and explain "
        "specifically why each one matters. Then write a short synthesis paragraph "
        "positioning the project within the body of related literature."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "related_papers": {
                "type": "array",
                "description": (
                    "Each retrieved paper that is meaningfully related to the project. "
                    "Omit papers with no substantive connection."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "arxiv_id": {
                            "type": "string",
                            "description": "The arXiv ID of the paper, exactly as provided.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Full title of the paper.",
                        },
                        "relevance_note": {
                            "type": "string",
                            "description": (
                                "1-2 sentences on specifically why this paper relates to the "
                                "user's project. Be concrete: does it use a similar method, "
                                "is it a direct technical precedent, does it present a "
                                "contrasting approach, or does it benchmark the same task?"
                            ),
                        },
                    },
                    "required": ["arxiv_id", "title", "relevance_note"],
                },
            },
            "summary": {
                "type": "string",
                "description": (
                    "A 3-5 sentence paragraph describing how the body of related work "
                    "positions the user's project. What is already established? What gap "
                    "or contribution does the project address? Cite papers by title."
                ),
            },
        },
        "required": ["related_papers", "summary"],
    },
}


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the relate capability."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_relate_prompt(
    project_description: str,
    extractions: dict[str, dict],
    comparisons: dict,
    titles: dict[str, str] | None = None,
) -> str:
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

    papers_block = "\n\n".join(blocks)

    cross_paper_notes = ""
    agreements = comparisons.get("agreements") or []
    if agreements:
        cross_paper_notes += "Agreements across papers:\n" + "\n".join(f"- {a}" for a in agreements) + "\n"
    unique = comparisons.get("unique_contributions") or {}
    if unique:
        cross_paper_notes += "\nUnique contributions:\n" + "\n".join(
            f"- [{k}] {v}" for k, v in unique.items()
        )

    return (
        f"User's project description:\n{project_description}\n\n"
        "Retrieved papers (with extracted fields):\n\n"
        + papers_block
        + (f"\n\nCross-paper context:\n{cross_paper_notes}" if cross_paper_notes else "")
    )


def relate_to_project(
    project_description: str,
    extractions: dict[str, dict],
    comparisons: dict,
    titles: dict[str, str] | None = None,
) -> dict:
    """Identify which retrieved papers relate to the user's project and explain how.

    Args:
        project_description: The user's own project description (short or abstract-length).
        extractions:         dict mapping arxiv_id → extraction dict from extract_paper().
        comparisons:         Cross-paper comparison dict from compare_extractions().
        titles:              Optional dict mapping arxiv_id → paper title for prompt labels.

    Returns a dict with keys:
        related_papers  — list of {arxiv_id, title, relevance_note} dicts
        summary         — short paragraph positioning the project in the literature
    """
    client = _get_client()
    prompt = _build_relate_prompt(project_description, extractions, comparisons, titles)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        tools=[_RELATE_TOOL],
        tool_choice={"type": "tool", "name": "find_related_papers"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the find_related_papers tool to identify which of the retrieved "
                    "papers are relevant to the user's project and summarise how they "
                    "position it within the literature.\n\n"
                    + prompt
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "find_related_papers":
            result = block.input
            seen: set[str] = set()
            deduped = []
            for paper in result.get("related_papers") or []:
                aid = paper.get("arxiv_id")
                if aid and aid not in seen:
                    seen.add(aid)
                    deduped.append(paper)
            result["related_papers"] = deduped
            return result

    raise RuntimeError(
        f"Claude did not return a find_related_papers tool_use block. "
        f"Stop reason: {response.stop_reason}. Content: {response.content}"
    )


@register("relate")
def relate_papers(context: ResearchContext) -> None:
    """Map retrieved papers onto the user's project description.

    Reads context.project_description, context.summaries (extractions), and
    context.comparisons. Writes result to context.related_work_result.
    """
    if not context.project_description:
        print("[relate] No project_description set — skipping.")
        return

    extractions = {
        arxiv_id: entry["extraction"]
        for arxiv_id, entry in context.summaries.items()
        if "extraction" in entry
    }

    titles = {p.arxiv_id: p.title for p in context.found_papers}

    print(f"[relate] Mapping {len(extractions)} papers onto project description...")
    result = relate_to_project(
        context.project_description,
        extractions,
        context.comparisons,
        titles=titles,
    )
    context.related_work_result = result
    print(
        f"[relate] Done — {len(result.get('related_papers', []))} relevant papers identified."
    )
