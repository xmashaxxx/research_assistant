"""Synthesis capability — prose answer to the research query using claude-sonnet-4-6."""

from __future__ import annotations

import os

import anthropic

from research_assistant.context import ResearchContext
from research_assistant.registry import register

# Sonnet for synthesis: this is the one step where writing quality and
# multi-source reasoning matter more than throughput.
_MODEL = "claude-sonnet-4-6"


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the synthesize capability."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_synthesis_prompt(
    query: str,
    summaries: dict,
    comparisons: dict,
) -> str:
    """Render the query, per-paper extractions, and comparison into a prompt."""

    # --- per-paper extraction block ---
    paper_blocks = []
    for arxiv_id, entry in summaries.items():
        ext = entry.get("extraction")
        if not ext:
            continue
        lines = [f"[{arxiv_id}]"]
        for field in ("research_question", "method", "key_results", "limitations"):
            val = ext.get(field)
            if val:
                lines.append(f"  {field}: {val}")
        paper_blocks.append("\n".join(lines))
    papers_section = "\n\n".join(paper_blocks) if paper_blocks else "(none)"

    # --- comparison block ---
    comp_lines = []

    agreements = comparisons.get("agreements") or []
    if agreements:
        comp_lines.append("Agreements across papers:")
        for ag in agreements:
            comp_lines.append(f"  - {ag}")

    disagreements = comparisons.get("disagreements") or []
    if disagreements:
        comp_lines.append("\nDisagreements / differing positions:")
        for d in disagreements:
            point = d.get("point", "")
            positions = d.get("positions", {})
            comp_lines.append(f"  Topic: {point}")
            for pid, stance in positions.items():
                comp_lines.append(f"    [{pid}] {stance}")

    unique = comparisons.get("unique_contributions") or {}
    if unique:
        comp_lines.append("\nUnique contributions:")
        for pid, contrib in unique.items():
            comp_lines.append(f"  [{pid}] {contrib}")

    comparison_section = "\n".join(comp_lines) if comp_lines else "(none)"

    return f"""You are a research assistant writing a literature synthesis for an ML practitioner.

Research query: {query}

=== Extracted paper summaries ===
{papers_section}

=== Cross-paper comparison ===
{comparison_section}

=== Instructions ===
Write a cohesive synthesis paragraph (or short set of paragraphs) that:
1. Directly answers the research query based on what these papers show.
2. States clearly what the literature agrees on, citing papers by title when making specific claims.
3. Describes open questions or disagreements, attributing each position to the paper that holds it.
4. Notes what is NOT covered by the reviewed papers — so the reader knows the limits of this synthesis.
5. Is written in clear, precise prose for someone with an ML background but not necessarily expert in this exact subfield. Do not use bullet lists — write in flowing prose like a well-crafted literature review paragraph.
6. Is substantive: aim for 3–5 solid paragraphs."""


def synthesize(query: str, summaries: dict, comparisons: dict) -> str:
    """Produce a prose synthesis answering the research query.

    Reads structured extractions from summaries and the comparison result,
    then asks Claude (Sonnet) to write a cohesive literature-review-style
    answer. Returns the synthesis as a plain string.
    """
    client = _get_client()
    prompt = _build_synthesis_prompt(query, summaries, comparisons)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


@register("synthesize")
def synthesize_capability(context: ResearchContext) -> None:
    """Synthesize a prose answer to context.query from extracted and compared papers.

    Reads context.summaries and context.comparisons, calls synthesize(),
    and writes the result to context.synthesis.
    """
    print(f"[synthesize] Writing synthesis for: {context.query!r}")
    context.synthesis = synthesize(
        query=context.query,
        summaries=context.summaries,
        comparisons=context.comparisons,
    )
    print(f"[synthesize] Done — {len(context.synthesis):,} chars.")
