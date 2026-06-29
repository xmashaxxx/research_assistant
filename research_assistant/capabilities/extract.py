"""Schema-driven extraction capability — uses Claude tool-use for structured output."""

from __future__ import annotations

import os
from pathlib import Path

import anthropic

from research_assistant.context import ResearchContext
from research_assistant.models import PaperRecord
from research_assistant.registry import register

_MODEL = "claude-haiku-4-5-20251001"
_TEXT_LIMIT = 8_000

# Mirrors general_cs_paper.json as a tool input_schema so Claude returns
# structured output without any markdown parsing.
_EXTRACT_TOOL: dict = {
    "name": "extract_paper_fields",
    "description": (
        "Extract structured information from a research paper. "
        "Fill every field based solely on what the paper text states. "
        "Use null for fields the paper does not address."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "research_question": {
                "type": "string",
                "description": "The central research question or problem the paper addresses.",
            },
            "method": {
                "type": "string",
                "description": "The primary method, model, or algorithm proposed or studied.",
            },
            "dataset": {
                "type": ["string", "null"],
                "description": "Datasets used for training, evaluation, or experiments. Null if none mentioned.",
            },
            "key_results": {
                "type": "string",
                "description": "Main empirical findings, benchmark scores, or theoretical results.",
            },
            "limitations": {
                "type": ["string", "null"],
                "description": "Acknowledged limitations, failure modes, or scope restrictions. Null if not stated.",
            },
            "related_work": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Key related works or prior approaches the paper explicitly builds on. Null if not discussed.",
            },
        },
        "required": [
            "research_question",
            "method",
            "dataset",
            "key_results",
            "limitations",
            "related_work",
        ],
    },
}


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the extract capability."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_prompt(paper: PaperRecord) -> str:
    authors = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors += f" et al. ({len(paper.authors)} total)"

    if paper.full_text:
        body = paper.full_text[:_TEXT_LIMIT]
        source_label = f"first {_TEXT_LIMIT:,} chars of full text"
    else:
        body = paper.abstract
        source_label = "abstract only"

    return (
        f"Title: {paper.title}\n"
        f"Authors: {authors}\n"
        f"arXiv: {paper.arxiv_id}\n\n"
        f"Paper text ({source_label}):\n\n{body}"
    )


def extract_paper(paper: PaperRecord) -> dict:
    """Extract structured fields from a single paper using Claude tool-use.

    Forces Claude to call extract_paper_fields via tool_choice so the result
    is always a typed dict — no markdown parsing required.

    Returns a dict with keys: research_question, method, dataset, key_results,
    limitations, related_work. Nullable fields may be None.
    """
    client = _get_client()
    prompt = _build_prompt(paper)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_paper_fields"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract structured information from this research paper "
                    "using the extract_paper_fields tool.\n\n" + prompt
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_paper_fields":
            return block.input

    raise RuntimeError(
        f"Claude did not return a tool_use block for paper {paper.arxiv_id}. "
        f"Stop reason: {response.stop_reason}. Content: {response.content}"
    )


@register("extract")
def extract_summaries(context: ResearchContext) -> None:
    """Extract structured fields from all papers in context.found_papers.

    Calls Claude for each PaperRecord and writes results to
    context.summaries[arxiv_id]["extraction"].
    """
    papers = context.found_papers
    print(f"[extract] Extracting fields from {len(papers)} papers...")

    for i, paper in enumerate(papers, 1):
        short_title = paper.title[:55] + "..." if len(paper.title) > 55 else paper.title
        print(f"[extract] ({i}/{len(papers)}) {paper.arxiv_id}: {short_title}")

        extraction = extract_paper(paper)

        if paper.arxiv_id not in context.summaries:
            context.summaries[paper.arxiv_id] = {}
        context.summaries[paper.arxiv_id]["extraction"] = extraction
        print(f"[extract]   ok — {len([v for v in extraction.values() if v is not None])} fields populated")

    print(f"[extract] Done. {len(papers)} papers extracted.")
