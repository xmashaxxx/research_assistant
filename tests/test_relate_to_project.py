"""
tests/test_relate_to_project.py

Integration tests for the relate-to-project capability.

Runs the full pipeline in project-description mode against real arXiv and
Anthropic APIs. Requires ANTHROPIC_API_KEY and network access.

The test project description is intentionally generic and plausible for a
coursework RAG project — specific enough to generate meaningful relevance notes,
not so narrow that no papers match.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.capabilities.relate_to_project import (
    relate_to_project,
    _RELATE_TOOL,
    _MODEL,
    _build_relate_prompt,
)

PROJECT_DESCRIPTION = (
    "I'm building a retrieval-augmented generation pipeline using a quantized "
    "Llama model running in Google Colab, for a coursework project. The system "
    "retrieves relevant passages from a small document corpus using FAISS, then "
    "passes them as context to the language model to answer user questions. I want "
    "to understand what related work exists and how my approach compares."
)

_FAKE_EXTRACTIONS = {
    "2312.10997": {
        "research_question": "How can RAG systems be systematically categorised and improved?",
        "method": "Survey and taxonomy of RAG architectures",
        "dataset": "Multiple NLP benchmarks",
        "key_results": "Three RAG paradigms identified; iterative retrieval outperforms single-pass.",
        "limitations": "Limited to English-language papers through 2023.",
        "related_work": ["Lewis et al. 2020", "DPR", "REALM"],
    },
    "2005.11401": {
        "research_question": "Can retrieval augmentation reduce hallucination in NLP tasks?",
        "method": "RAG with FAISS dense retrieval over Wikipedia passages and BART generation",
        "dataset": "Natural Questions, TriviaQA, WebQuestions",
        "key_results": "RAG outperforms closed-book GPT-2 by up to 18% on open-domain QA.",
        "limitations": "Fixed retrieval corpus; no mechanism to update knowledge after training.",
        "related_work": ["DPR", "BART", "T5"],
    },
}

_FAKE_COMPARISONS = {
    "agreements": ["Both papers use dense retrieval as the core retrieval mechanism."],
    "disagreements": [],
    "unique_contributions": {
        "2312.10997": "Comprehensive taxonomy of RAG paradigms.",
        "2005.11401": "First end-to-end trainable RAG model with FAISS and BART.",
    },
}

_FAKE_RESULT = {
    "related_papers": [
        {
            "arxiv_id": "2312.10997",
            "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "relevance_note": (
                "This survey directly covers the RAG architecture you are building — "
                "it classifies your approach as Naive RAG and discusses FAISS-based "
                "dense retrieval as a standard component."
            ),
        },
        {
            "arxiv_id": "2005.11401",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "relevance_note": (
                "The original RAG paper uses the same FAISS + generator architecture "
                "your project implements, making it the primary technical precedent."
            ),
        },
    ],
    "summary": (
        "Your coursework project implements a classic Naive RAG pipeline — FAISS "
        "retrieval feeding a generative model — which is the architecture that Lewis "
        "et al. (2020) originally proposed and Gao et al. (2023) systematically "
        "categorise. Both works establish that retrieval quality is the primary "
        "determinant of downstream answer quality, which is directly relevant to "
        "your corpus and retrieval design choices."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(result: dict = _FAKE_RESULT) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "find_related_papers"
    block.input = result

    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Unit tests — no network, no API key
# ---------------------------------------------------------------------------

class TestRelateTool:
    def test_tool_name(self):
        assert _RELATE_TOOL["name"] == "find_related_papers"

    def test_required_output_fields(self):
        required = set(_RELATE_TOOL["input_schema"]["required"])
        assert required == {"related_papers", "summary"}

    def test_related_papers_is_array(self):
        prop = _RELATE_TOOL["input_schema"]["properties"]["related_papers"]
        assert prop["type"] == "array"

    def test_paper_item_has_required_keys(self):
        item = _RELATE_TOOL["input_schema"]["properties"]["related_papers"]["items"]
        assert set(item["required"]) == {"arxiv_id", "title", "relevance_note"}

    def test_summary_is_string(self):
        prop = _RELATE_TOOL["input_schema"]["properties"]["summary"]
        assert prop["type"] == "string"


class TestBuildRelatePrompt:
    def test_project_description_appears(self):
        prompt = _build_relate_prompt(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)
        assert "quantized" in prompt
        assert "FAISS" in prompt

    def test_arxiv_ids_appear(self):
        prompt = _build_relate_prompt(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)
        assert "2312.10997" in prompt
        assert "2005.11401" in prompt

    def test_agreements_included(self):
        prompt = _build_relate_prompt(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)
        assert "dense retrieval" in prompt


class TestRelateToProject:
    def test_correct_model_used(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.relate_to_project._get_client", return_value=client):
            relate_to_project(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)
        assert client.messages.create.call_args.kwargs["model"] == _MODEL

    def test_tool_choice_forces_find_related_papers(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.relate_to_project._get_client", return_value=client):
            relate_to_project(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)
        tc = client.messages.create.call_args.kwargs["tool_choice"]
        assert tc == {"type": "tool", "name": "find_related_papers"}

    def test_returns_tool_input_dict(self):
        with patch("research_assistant.capabilities.relate_to_project._get_client",
                   return_value=_make_mock_client()):
            result = relate_to_project(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)
        assert result == _FAKE_RESULT

    def test_deduplicates_repeated_arxiv_ids(self):
        """Regression: model occasionally returns the same arxiv_id twice."""
        duplicate_result = {
            "related_papers": [
                {"arxiv_id": "2404.07220", "title": "Paper A", "relevance_note": "First note."},
                {"arxiv_id": "2312.10997", "title": "Paper B", "relevance_note": "Note B."},
                {"arxiv_id": "2404.07220", "title": "Paper A", "relevance_note": "Duplicate note."},
            ],
            "summary": "Some summary.",
        }
        with patch("research_assistant.capabilities.relate_to_project._get_client",
                   return_value=_make_mock_client(duplicate_result)):
            result = relate_to_project(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)

        ids = [p["arxiv_id"] for p in result["related_papers"]]
        assert ids == list(dict.fromkeys(ids)), f"Duplicate arxiv_ids found: {ids}"
        assert len(result["related_papers"]) == 2
        # First occurrence is kept, not the duplicate
        assert result["related_papers"][0]["relevance_note"] == "First note."

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                relate_to_project(PROJECT_DESCRIPTION, _FAKE_EXTRACTIONS, _FAKE_COMPARISONS)


# ---------------------------------------------------------------------------
# Integration tests — require ANTHROPIC_API_KEY + network
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def require_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live relate test.")


@pytest.fixture(scope="module")
def full_pipeline_result(require_api_key):
    """Run search → fetch → extract → compare → relate for the test project.

    Module-scoped so the full pipeline runs once and all integration tests
    share the result.
    """
    from research_assistant.context import ResearchContext
    from research_assistant.orchestrator import run_pipeline

    ctx = ResearchContext(project_description=PROJECT_DESCRIPTION)
    run_pipeline(ctx)
    return ctx


class TestRelateIntegration:
    def test_related_papers_nonempty(self, full_pipeline_result):
        result = full_pipeline_result.related_work_result
        assert result is not None
        assert isinstance(result.get("related_papers"), list)
        assert len(result["related_papers"]) > 0

    def test_each_paper_has_required_keys(self, full_pipeline_result):
        for paper in full_pipeline_result.related_work_result["related_papers"]:
            assert "arxiv_id" in paper, f"Missing arxiv_id in: {paper}"
            assert "title" in paper, f"Missing title in: {paper}"
            assert "relevance_note" in paper, f"Missing relevance_note in: {paper}"

    def test_relevance_notes_are_substantive(self, full_pipeline_result):
        for paper in full_pipeline_result.related_work_result["related_papers"]:
            assert len(paper["relevance_note"]) > 30, (
                f"Relevance note too short for {paper['arxiv_id']}: {paper['relevance_note']!r}"
            )

    def test_summary_nonempty(self, full_pipeline_result):
        summary = full_pipeline_result.related_work_result.get("summary", "")
        assert isinstance(summary, str)
        assert len(summary) > 200, f"Summary too short ({len(summary)} chars): {summary!r}"

    def test_found_papers_populated(self, full_pipeline_result):
        assert len(full_pipeline_result.found_papers) > 0

    def test_synthesis_not_set_in_project_mode(self, full_pipeline_result):
        assert full_pipeline_result.synthesis is None

    def test_print_full_result(self, full_pipeline_result):
        """Print the complete relate output so we can read it."""
        ctx = full_pipeline_result
        result = ctx.related_work_result

        print(f"\n{'='*65}")
        print("RELATE-TO-PROJECT OUTPUT")
        print(f"{'='*65}")
        print(f"\nProject: {PROJECT_DESCRIPTION}\n")

        print(f"Papers found by search: {len(ctx.found_papers)}")
        for p in ctx.found_papers:
            print(f"  [{p.arxiv_id}] {p.title[:70]}")

        print(f"\n--- RELATED PAPERS ({len(result.get('related_papers', []))}) ---")
        for i, paper in enumerate(result.get("related_papers", []), 1):
            print(f"\n  {i}. [{paper['arxiv_id']}] {paper['title']}")
            print(f"     {paper['relevance_note']}")

        print(f"\n--- SUMMARY ---")
        print(result.get("summary", ""))
        print(f"\n{'='*65}\n")

        assert result  # always passes — exists for its output
