"""Skeleton tests: verify package structure and registry wiring."""

import pytest
import research_assistant
from research_assistant.context import ResearchContext
from research_assistant.registry import registry

ALL_CAPABILITIES = {"search", "fetch", "extract", "compare", "synthesize"}

# Update this set as capabilities are implemented.
STUB_CAPABILITIES = {"synthesize"}


def test_registry_contains_all_capabilities():
    assert ALL_CAPABILITIES.issubset(set(registry.keys())), (
        f"Missing capabilities: {ALL_CAPABILITIES - set(registry.keys())}"
    )


def test_research_context_default_instantiation():
    ctx = ResearchContext()
    assert ctx.query == ""
    assert ctx.found_papers == []
    assert ctx.summaries == {}
    assert ctx.comparisons == {}
    assert ctx.synthesis is None
    assert ctx.extraction_schema == "general_cs_paper"


def test_research_context_with_query():
    ctx = ResearchContext(query="What are the best RAG architectures in 2024?")
    assert ctx.query == "What are the best RAG architectures in 2024?"


@pytest.mark.parametrize("capability_name", sorted(STUB_CAPABILITIES))
def test_stub_capabilities_raise_not_implemented(capability_name):
    ctx = ResearchContext(query="test")
    with pytest.raises(NotImplementedError):
        registry[capability_name](ctx)
