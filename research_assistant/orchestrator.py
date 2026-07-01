"""Orchestrator: runs an ordered list of capabilities against a ResearchContext."""

from __future__ import annotations

from research_assistant.context import ResearchContext
from research_assistant.registry import registry

_GENERAL_PIPELINE = ["search", "fetch", "extract", "compare", "synthesize", "ideate"]
_PROJECT_PIPELINE = ["search", "fetch", "extract", "compare", "relate"]


def run(context: ResearchContext, capability_names: list[str]) -> ResearchContext:
    """Execute capabilities in order, passing context through each stage.

    Raises KeyError if a requested capability is not registered.
    Propagates any exception raised by a capability without swallowing it.
    """
    for name in capability_names:
        if name not in registry:
            raise KeyError(f"No capability registered under '{name}'")
        registry[name](context)
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
