"""research_assistant — agentic pipeline for scientific literature search, extraction, and synthesis."""

from research_assistant.context import ResearchContext
from research_assistant.registry import register, registry
from research_assistant import orchestrator
import research_assistant.capabilities  # noqa: F401 — triggers self-registration of all capabilities

__all__ = ["ResearchContext", "register", "registry", "orchestrator"]
