"""Orchestrator: runs an ordered list of capabilities against a ResearchContext."""

from __future__ import annotations

from research_assistant.context import ResearchContext
from research_assistant.registry import registry


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
