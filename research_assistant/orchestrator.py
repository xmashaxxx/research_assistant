"""Orchestrator: runs an ordered list of capabilities against a ResearchContext."""

from __future__ import annotations

from research_assistant.context import ResearchContext
from research_assistant.guardrails import (
    GuardrailError,
    check_synthesis_grounding,
    validate_query,
    validate_search_results,
    validate_stage_output,
)
from research_assistant.registry import registry

_GENERAL_PIPELINE = ["search", "fetch", "extract", "compare", "synthesize", "ideate"]
_PROJECT_PIPELINE = ["search", "fetch", "extract", "compare", "relate"]


def run(context: ResearchContext, capability_names: list[str]) -> ResearchContext:
    """Execute capabilities in order, passing context through each stage.

    Applies guardrails before the first stage (query validation) and after
    each stage (output validation, grounding check after synthesize).

    Raises GuardrailError if a blocking guardrail fires.
    Raises KeyError if a requested capability is not registered.
    Propagates any exception raised by a capability without swallowing it.
    """
    # --- Pre-pipeline: query validation ---
    qv = validate_query(context.query or context.project_description or "")
    if qv["severity"] == "block":
        raise GuardrailError(qv["issue"])
    if qv["severity"] == "warn":
        print(f"[guardrail] Query warning: {qv['issue']}")

    for name in capability_names:
        if name not in registry:
            raise KeyError(f"No capability registered under '{name}'")
        registry[name](context)

        # --- Post-stage output validation ---
        if name == "search":
            sr = validate_search_results(
                context.query or context.project_description or "",
                context.found_papers,
            )
            if sr["severity"] != "ok":
                print(f"[guardrail] Search warning: {sr['issue']}")

        elif name == "extract":
            ev = validate_stage_output("extract", context.summaries, context)
            if not ev["passed"]:
                for w in ev["warnings"]:
                    print(f"[guardrail] Extract warning: {w}")
                if ev["fatal"]:
                    raise GuardrailError(f"Extract stage failed: {ev['warnings'][0]}")

        elif name == "compare":
            cv = validate_stage_output("compare", context.comparisons, context)
            if not cv["passed"]:
                for w in cv["warnings"]:
                    print(f"[guardrail] Compare warning: {w}")

        elif name == "synthesize":
            sv = validate_stage_output(
                "synthesize", {"synthesis": context.synthesis}, context
            )
            if not sv["passed"]:
                for w in sv["warnings"]:
                    print(f"[guardrail] Synthesis warning: {w}")

            # Grounding check — probabilistic hallucination detection
            gc = check_synthesis_grounding(context.synthesis or "", context.summaries)
            context.grounding_check = gc
            if gc.get("warning"):
                print(f"[guardrail] Grounding warning: {gc['warning']}")

        elif name == "ideate":
            iv = validate_stage_output(
                "ideate", context.experiment_ideas or {}, context
            )
            if not iv["passed"]:
                for w in iv["warnings"]:
                    print(f"[guardrail] Ideate warning: {w}")

    return context


def run_pipeline(context: ResearchContext) -> ResearchContext:
    """Run the appropriate pipeline based on context mode.

    If context.project_description is set, runs the "relate" pipeline
    (search → fetch → extract → compare → relate).
    Otherwise runs the standard synthesis pipeline
    (search → fetch → extract → compare → synthesize).
    """
    pipeline = _PROJECT_PIPELINE if context.project_description else _GENERAL_PIPELINE
    return run(context, pipeline)
