"""Guardrails layer — query validation, stage output checks, hallucination detection."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from research_assistant.context import ResearchContext


class GuardrailError(Exception):
    """Raised when a guardrail blocks pipeline execution."""


_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_VALIDATE_QUERY_TOOL: dict = {
    "name": "validate_query",
    "description": (
        "Evaluate whether an input is a suitable research question for an academic "
        "literature search pipeline that queries arXiv."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "valid": {
                "type": "boolean",
                "description": "Whether the query should proceed through the pipeline.",
            },
            "issue": {
                "type": ["string", "null"],
                "description": "Brief human-readable explanation of the problem, or null if ok.",
            },
            "severity": {
                "type": "string",
                "enum": ["block", "warn", "ok"],
                "description": (
                    "block = stop the pipeline entirely; "
                    "warn = proceed but show a notice to the user; "
                    "ok = no issues found."
                ),
            },
        },
        "required": ["valid", "issue", "severity"],
    },
}

_GROUNDING_TOOL: dict = {
    "name": "check_grounding",
    "description": (
        "Check whether specific factual claims in an AI-generated synthesis can be "
        "traced back to the provided source paper extractions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific factual claims (numbers, percentages, benchmark scores, "
                    "named techniques) that appear supported by at least one extraction."
                ),
            },
            "ungrounded_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific factual claims that could NOT be traced to any source extraction."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "Overall grounding confidence: "
                    "high = most claims traceable, low = many claims unverifiable."
                ),
            },
            "warning": {
                "type": ["string", "null"],
                "description": (
                    "Human-readable summary if ungrounded claims were found, otherwise null."
                ),
            },
        },
        "required": ["grounded_claims", "ungrounded_claims", "confidence", "warning"],
    },
}

# ---------------------------------------------------------------------------
# Shared client helper
# ---------------------------------------------------------------------------

def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the guardrails."
        )
    return anthropic.Anthropic(api_key=api_key)

# ---------------------------------------------------------------------------
# 1. Safety / Relevance Filter
# ---------------------------------------------------------------------------

def validate_query(query: str) -> dict:
    """Validate a research query before the pipeline starts.

    Short-circuits locally for empty/trivially short queries before calling Claude.

    Args:
        query: The raw user input string.

    Returns a dict with keys:
        valid    — whether to proceed
        issue    — human-readable explanation, or None
        severity — "block" | "warn" | "ok"
    """
    stripped = (query or "").strip()
    if not stripped:
        return {"valid": False, "issue": "Query is empty.", "severity": "block"}
    if len(stripped) < 10:
        return {
            "valid": False,
            "issue": "Query is too short to be a meaningful research question.",
            "severity": "block",
        }

    client = _get_client()
    prompt = (
        f"You are evaluating whether the following input is suitable for an academic "
        f"literature search pipeline that searches arXiv.\n\n"
        f"Input: {stripped!r}\n\n"
        f"Evaluate along three dimensions:\n"
        f"1. Is this a genuine research question or topic? It should be about science, "
        f"technology, medicine, social science, etc. Block: personal data requests, "
        f"harmful instructions, arithmetic, single words, gibberish, or anything that "
        f"is clearly not a research question.\n"
        f"2. Is it in English? Non-English queries should receive severity 'warn' (not block) "
        f"with a note that arXiv search works best in English. Do not block them.\n"
        f"3. Is it specific enough to return useful academic papers? Very vague queries "
        f"(e.g. 'machine learning') warrant a 'warn', not a block.\n\n"
        f"Use the validate_query tool to return your assessment."
    )

    resp = client.messages.create(
        model=_MODEL,
        max_tokens=256,
        tools=[_VALIDATE_QUERY_TOOL],
        tool_choice={"type": "tool", "name": "validate_query"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "validate_query":
            return block.input

    return {"valid": True, "issue": None, "severity": "ok"}


def validate_search_results(query: str, papers: list) -> dict:
    """Heuristic check on search results after Stage 1 (no API call).

    Returns same shape as validate_query: {valid, issue, severity}.
    """
    if len(papers) < 3:
        return {
            "valid": True,
            "issue": (
                f"Only {len(papers)} paper(s) found — results may be thin. "
                "Consider broadening or rephrasing your query."
            ),
            "severity": "warn",
        }

    _STOPWORDS = {
        "a", "an", "the", "of", "in", "for", "on", "with", "and", "or",
        "to", "is", "are", "how", "what", "why", "when", "which", "does",
        "do", "its", "at", "by", "from", "that", "this", "be", "as",
    }
    query_words = {
        w.lower()
        for w in re.findall(r"\w+", query)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }
    if not query_words:
        return {"valid": True, "issue": None, "severity": "ok"}

    on_topic = sum(
        1 for p in papers
        if any(w in (getattr(p, "title", "") or "").lower() for w in query_words)
    )
    if on_topic / len(papers) < 0.3:
        return {
            "valid": True,
            "issue": (
                "Fewer than 30 % of retrieved papers appear to match your query keywords. "
                "Results may be off-topic — consider rephrasing."
            ),
            "severity": "warn",
        }

    return {"valid": True, "issue": None, "severity": "ok"}

# ---------------------------------------------------------------------------
# 2. Stage Output Validation
# ---------------------------------------------------------------------------

def validate_stage_output(stage: str, output: dict, context: ResearchContext) -> dict:
    """Pure heuristic validation of a stage's output. No API calls.

    Args:
        stage:   One of "extract", "compare", "synthesize", "ideate".
        output:  The relevant portion of context for that stage.
        context: Full ResearchContext (used for cross-referencing paper titles etc.).

    Returns a dict with keys:
        passed   — True if no warnings
        warnings — list of specific issues found
        fatal    — if True, pipeline should stop rather than proceed with bad data
    """
    warnings: list[str] = []
    fatal = False

    if stage == "extract":
        total = len(output)
        if total == 0:
            return {"passed": False, "warnings": ["No papers were extracted."], "fatal": True}

        non_null_rq = sum(
            1 for s in output.values()
            if (s.get("extracted") or {}).get("research_question")
        )
        non_null_kr = sum(
            1 for s in output.values()
            if (s.get("extracted") or {}).get("key_results")
        )

        if non_null_rq / total < 0.8:
            warnings.append(
                f"Only {non_null_rq}/{total} papers returned a non-null research_question "
                f"({non_null_rq/total:.0%}). Extraction quality may be low — papers may be "
                f"paywalled or abstract-only."
            )
        if non_null_kr / total < 0.8:
            warnings.append(
                f"Only {non_null_kr}/{total} papers returned non-null key_results "
                f"({non_null_kr/total:.0%}). Extraction quality may be low."
            )

    elif stage == "compare":
        agreements = output.get("agreements") or []
        disagreements = output.get("disagreements") or []
        if not agreements and not disagreements:
            warnings.append(
                "Comparison produced no agreements and no disagreements — "
                "the compare stage may have failed silently."
            )

    elif stage == "synthesize":
        synthesis = output.get("synthesis") or ""

        if len(synthesis) < 500:
            warnings.append(
                f"Synthesis is very short ({len(synthesis)} chars) and may be incomplete."
            )

        # English heuristic: ASCII alpha chars should dominate
        alpha = sum(1 for c in synthesis if c.isalpha())
        ascii_alpha = sum(1 for c in synthesis if c.isascii() and c.isalpha())
        if alpha > 0 and ascii_alpha / alpha < 0.8:
            warnings.append("Synthesis may not be in English.")

        # Citation heuristic: arXiv ID brackets, "et al.", or year-in-parens
        has_citation = bool(
            re.search(r"\[\d{4}\.\d{4,5}\]", synthesis)
            or re.search(r"\bet\s+al\b", synthesis, re.IGNORECASE)
            or re.search(r"\(\d{4}\)", synthesis)
        )
        if not has_citation:
            # Fallback: check if any paper title word (≥7 chars) appears in synthesis
            for paper in (getattr(context, "found_papers", None) or []):
                title = getattr(paper, "title", "") or ""
                if any(
                    w.lower() in synthesis.lower()
                    for w in re.findall(r"\b\w{7,}\b", title)
                ):
                    has_citation = True
                    break
        if not has_citation:
            warnings.append(
                "Synthesis does not appear to reference any source papers by arXiv ID, "
                "'et al.', or title."
            )

    elif stage == "ideate":
        ideas = output.get("experiment_ideas") or []
        most_promising = output.get("most_promising")

        if not ideas:
            warnings.append("No experiment ideas were generated.")

        for i, idea in enumerate(ideas):
            if not (idea.get("hypothesis") or "").strip():
                warnings.append(f"Idea #{i} ('{idea.get('title', '')}') is missing a hypothesis.")
            if not (idea.get("method") or "").strip():
                warnings.append(f"Idea #{i} ('{idea.get('title', '')}') is missing a method.")

        if ideas and most_promising is not None:
            if not isinstance(most_promising, int) or not (0 <= most_promising < len(ideas)):
                warnings.append(
                    f"most_promising={most_promising!r} is not a valid index "
                    f"into {len(ideas)} idea(s)."
                )

    return {"passed": len(warnings) == 0, "warnings": warnings, "fatal": fatal}

# ---------------------------------------------------------------------------
# 3. Hallucination / Grounding Check
# ---------------------------------------------------------------------------

def check_synthesis_grounding(synthesis: str, extractions: dict) -> dict:
    """Use Claude Haiku to check whether specific factual claims in the synthesis
    can be traced back to the source paper extractions.

    This is a probabilistic heuristic, not a guarantee. A claim may be correct
    even if it cannot be located in the extracted snippets.

    Args:
        synthesis:   The prose synthesis produced by the synthesize stage.
        extractions: context.summaries — {arxiv_id: {"extracted": {...}, ...}}

    Returns a dict with keys:
        grounded_claims   — claims traceable to at least one source
        ungrounded_claims — claims that couldn't be traced
        confidence        — "high" | "medium" | "low"
        warning           — human-readable summary, or None
    """
    if not (synthesis or "").strip():
        return {
            "grounded_claims": [],
            "ungrounded_claims": [],
            "confidence": "low",
            "warning": "No synthesis to check.",
        }

    # Build a compact representation of source extractions for the prompt
    source_lines: list[str] = []
    for arxiv_id, summary in extractions.items():
        extracted = (summary.get("extracted") or {}) if isinstance(summary, dict) else {}
        title = extracted.get("title") or arxiv_id
        key_results = (extracted.get("key_results") or "")[:400]
        method = (extracted.get("method") or "")[:200]
        dataset = (extracted.get("dataset") or "")[:100]
        source_lines.append(f"[{arxiv_id}] {title}")
        if key_results:
            source_lines.append(f"  Key results: {key_results}")
        if method:
            source_lines.append(f"  Method: {method}")
        if dataset:
            source_lines.append(f"  Dataset: {dataset}")

    sources_text = "\n".join(source_lines) if source_lines else "(no extractions available)"
    synthesis_excerpt = synthesis[:3000]

    client = _get_client()
    prompt = (
        "You are performing a grounding check on an AI-generated literature synthesis. "
        "Your task: identify specific factual claims in the synthesis and determine "
        "whether each can be traced to the provided source paper extractions.\n\n"
        "Focus ONLY on specific, verifiable claims: numbers, percentages, benchmark scores, "
        "dataset names, model names, named techniques, and quantitative comparisons. "
        "Do NOT flag general statements or interpretations — only hard factual assertions.\n\n"
        f"=== Synthesis ===\n{synthesis_excerpt}\n\n"
        f"=== Source extractions ===\n{sources_text}\n\n"
        "Use the check_grounding tool. Remember: this is probabilistic — a claim may be "
        "correct even if absent from the extractions."
    )

    resp = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        tools=[_GROUNDING_TOOL],
        tool_choice={"type": "tool", "name": "check_grounding"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "check_grounding":
            return block.input

    return {
        "grounded_claims": [],
        "ungrounded_claims": [],
        "confidence": "low",
        "warning": "Grounding check did not complete.",
    }
