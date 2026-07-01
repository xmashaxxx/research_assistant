"""Experiment ideation capability — proposes concrete research experiments from a synthesis."""

from __future__ import annotations

import os

import anthropic

from research_assistant.context import ResearchContext
from research_assistant.registry import register

_MODEL = "claude-sonnet-4-6"

_IDEATE_TOOL: dict = {
    "name": "propose_experiments",
    "description": (
        "Propose concrete, actionable experiment ideas that address gaps or "
        "disagreements identified in a literature synthesis. Each idea must be "
        "specific enough that a researcher could implement and run it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment_ideas": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "description": "3 to 5 concrete experiment proposals, varying in difficulty.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short descriptive name for the experiment (≤10 words).",
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "What the experiment tests or expects to find.",
                        },
                        "method": {
                            "type": "string",
                            "description": (
                                "How you would run it — specific dataset(s), model(s), "
                                "baseline(s), and evaluation metric(s)."
                            ),
                        },
                        "gap_addressed": {
                            "type": "string",
                            "description": (
                                "Which specific gap, disagreement, or open question "
                                "from the literature this experiment addresses."
                            ),
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": (
                                "Implementation difficulty: low = a grad student weekend project, "
                                "medium = a few weeks of focused work, "
                                "high = a multi-month research effort."
                            ),
                        },
                    },
                    "required": ["title", "hypothesis", "method", "gap_addressed", "difficulty"],
                },
            },
            "most_promising": {
                "type": "integer",
                "description": (
                    "0-based index into experiment_ideas of the single most promising "
                    "experiment — the one with the best ratio of insight to implementation cost."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "One sentence explaining why the most_promising experiment is the "
                    "highest-value choice given the current state of the literature."
                ),
            },
        },
        "required": ["experiment_ideas", "most_promising", "reasoning"],
    },
}


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the ideate capability."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_ideation_prompt(
    query: str,
    synthesis: str,
    comparisons: dict,
) -> str:
    disagreements = comparisons.get("disagreements") or []
    unique = comparisons.get("unique_contributions") or {}

    gaps_lines = []
    if disagreements:
        gaps_lines.append("Disagreements / open questions across papers:")
        for d in disagreements:
            point = d.get("point", "")
            positions = d.get("positions") or {}
            gaps_lines.append(f"  - {point}")
            for pid, stance in positions.items():
                gaps_lines.append(f"      [{pid}] {stance}")

    if unique:
        gaps_lines.append("\nUnique contributions (each paper's distinct angle):")
        for pid, contrib in unique.items():
            gaps_lines.append(f"  [{pid}] {contrib}")

    gaps_section = "\n".join(gaps_lines) if gaps_lines else "(no structured gaps extracted)"

    return f"""You are a senior ML researcher advising a PhD student on what to work on next.

Research question the literature was surveyed on:
{query}

=== Literature synthesis ===
{synthesis}

=== Identified gaps and disagreements ===
{gaps_section}

=== Your task ===
Read the synthesis and gaps carefully. Use the propose_experiments tool to propose \
3–5 concrete experiment ideas that could meaningfully advance this area.

Rules for good experiment proposals:
- Each must be specific: name the dataset, the model type or scale, the baseline, \
and the metric you would use to evaluate.
- Each must be actionable: a researcher should be able to start running it next week.
- Together they should cover a range of difficulty (at least one low and one high).
- Prioritise experiments that address the sharpest disagreements or most glaring \
gaps in the literature, not incremental variations of existing work.
- Do not propose vague "future work" directions — propose experiments."""


def generate_experiment_ideas(
    query: str,
    synthesis: str,
    comparisons: dict,
) -> dict:
    """Propose concrete experiment ideas grounded in the literature synthesis.

    Args:
        query:       The original research question.
        synthesis:   The prose synthesis produced by the synthesize capability.
        comparisons: The structured comparison dict (agreements, disagreements,
                     unique_contributions) from the compare capability.

    Returns a dict with keys:
        experiment_ideas  — list of {title, hypothesis, method, gap_addressed, difficulty}
        most_promising    — 0-based index of the best idea
        reasoning         — one sentence on why that idea is the best choice
    """
    client = _get_client()
    prompt = _build_ideation_prompt(query, synthesis, comparisons)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        tools=[_IDEATE_TOOL],
        tool_choice={"type": "tool", "name": "propose_experiments"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "propose_experiments":
            return block.input

    raise RuntimeError(
        f"Claude did not return a propose_experiments tool_use block. "
        f"Stop reason: {response.stop_reason}. Content: {response.content}"
    )


@register("ideate")
def ideate_capability(context: ResearchContext) -> None:
    """Generate experiment ideas from the synthesis and comparisons in context.

    Reads context.query, context.synthesis, and context.comparisons.
    Writes results to context.experiment_ideas.
    """
    if not context.synthesis:
        print("[ideate] No synthesis in context — skipping.")
        return

    print(f"[ideate] Generating experiment ideas for: {context.query!r}")
    result = generate_experiment_ideas(
        query=context.query,
        synthesis=context.synthesis,
        comparisons=context.comparisons,
    )
    context.experiment_ideas = result
    n = len(result.get("experiment_ideas") or [])
    best = result.get("most_promising", 0)
    print(f"[ideate] Done — {n} ideas, most promising: #{best}.")
