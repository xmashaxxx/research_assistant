"""
tests/test_extract.py

Unit and integration tests for the extract capability.

Unit tests mock the Anthropic client entirely — no API key required.
Integration tests call the live Claude API and require ANTHROPIC_API_KEY.
Integration tests also call the live arXiv API for the paper fixture.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.capabilities.extract import (
    extract_paper,
    _EXTRACT_TOOL,
    _MODEL,
    _TEXT_LIMIT,
)
from research_assistant.models import PaperRecord

RAG_SURVEY_ID = "2312.10997"

_FAKE_EXTRACTION = {
    "research_question": "How can LLMs be augmented with retrieval?",
    "method": "Retrieval-Augmented Generation (RAG)",
    "dataset": "Natural Questions, TriviaQA",
    "key_results": "RAG outperforms closed-book models by 12% on NQ.",
    "limitations": "Retrieval latency increases inference time.",
    "related_work": ["DPR", "BM25", "REALM"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(**kwargs) -> PaperRecord:
    defaults = dict(
        arxiv_id="test.001",
        title="Test Paper: A Survey",
        authors=["Alice Example", "Bob Sample"],
        abstract="This paper surveys retrieval-augmented generation methods.",
        published="2024-01-01T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/test.001",
        arxiv_url="https://arxiv.org/abs/test.001",
    )
    defaults.update(kwargs)
    return PaperRecord(**defaults)


def _make_mock_client(extraction: dict = _FAKE_EXTRACTION) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns a tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_paper_fields"
    block.input = extraction

    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Unit tests — no network, no API key
# ---------------------------------------------------------------------------

class TestToolUsePattern:
    def test_correct_model_used(self):
        paper = _make_paper()
        with patch("research_assistant.capabilities.extract._get_client", return_value=_make_mock_client()):
            extract_paper(paper)

        client = _make_mock_client()
        with patch("research_assistant.capabilities.extract._get_client", return_value=client):
            extract_paper(paper)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == _MODEL

    def test_tool_choice_forces_specific_tool(self):
        paper = _make_paper()
        client = _make_mock_client()
        with patch("research_assistant.capabilities.extract._get_client", return_value=client):
            extract_paper(paper)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "extract_paper_fields"}

    def test_exactly_one_tool_registered(self):
        paper = _make_paper()
        client = _make_mock_client()
        with patch("research_assistant.capabilities.extract._get_client", return_value=client):
            extract_paper(paper)
        call_kwargs = client.messages.create.call_args.kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "extract_paper_fields"

    def test_tool_schema_has_all_required_fields(self):
        required = set(_EXTRACT_TOOL["input_schema"]["required"])
        assert required == {
            "research_question", "method", "dataset",
            "key_results", "limitations", "related_work",
        }

    def test_returns_tool_input_dict(self):
        paper = _make_paper()
        with patch("research_assistant.capabilities.extract._get_client", return_value=_make_mock_client()):
            result = extract_paper(paper)
        assert result == _FAKE_EXTRACTION

    def test_nullable_fields_allowed_in_schema(self):
        props = _EXTRACT_TOOL["input_schema"]["properties"]
        assert "null" in props["dataset"]["type"]
        assert "null" in props["limitations"]["type"]
        assert "null" in props["related_work"]["type"]

    def test_prompt_uses_full_text_when_available(self):
        long_text = "x" * 20_000
        paper = _make_paper(full_text=long_text)
        client = _make_mock_client()
        with patch("research_assistant.capabilities.extract._get_client", return_value=client):
            extract_paper(paper)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        # Should include truncated text, not the full 20k
        assert "x" * _TEXT_LIMIT in user_content
        assert "x" * (_TEXT_LIMIT + 1) not in user_content

    def test_prompt_falls_back_to_abstract(self):
        paper = _make_paper(full_text=None)
        client = _make_mock_client()
        with patch("research_assistant.capabilities.extract._get_client", return_value=client):
            extract_paper(paper)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert paper.abstract in user_content

    def test_prompt_includes_title_and_authors(self):
        paper = _make_paper()
        client = _make_mock_client()
        with patch("research_assistant.capabilities.extract._get_client", return_value=client):
            extract_paper(paper)
        user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert paper.title in user_content
        assert paper.authors[0] in user_content

    def test_missing_api_key_raises_environment_error(self):
        paper = _make_paper()
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if present
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                    extract_paper(paper)


# ---------------------------------------------------------------------------
# Integration test — requires ANTHROPIC_API_KEY and network
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rag_survey_paper():
    """Fetch the RAG survey paper once for integration tests. Skip on arXiv 503."""
    from research_assistant.capabilities.fetch_paper import fetch_papers
    results = fetch_papers([RAG_SURVEY_ID])
    if not results:
        pytest.skip("arXiv returned no results — possible 503, re-run to confirm.")
    return results[0]


class TestExtractIntegration:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set — skipping live API test.")

    def test_extract_returns_dict(self, rag_survey_paper):
        result = extract_paper(rag_survey_paper)
        assert isinstance(result, dict)

    def test_research_question_nonempty(self, rag_survey_paper):
        result = extract_paper(rag_survey_paper)
        assert result.get("research_question"), "research_question should be non-empty"

    def test_method_nonempty(self, rag_survey_paper):
        result = extract_paper(rag_survey_paper)
        assert result.get("method"), "method should be non-empty"

    def test_key_results_nonempty(self, rag_survey_paper):
        result = extract_paper(rag_survey_paper)
        assert result.get("key_results"), "key_results should be non-empty"

    def test_all_required_keys_present(self, rag_survey_paper):
        result = extract_paper(rag_survey_paper)
        missing = set(_EXTRACT_TOOL["input_schema"]["required"]) - result.keys()
        assert not missing, f"Missing keys in extraction: {missing}"

    def test_print_extracted_fields(self, rag_survey_paper):
        """Print extraction so we can inspect what Claude understood."""
        result = extract_paper(rag_survey_paper)
        print(f"\n{'='*60}")
        print(f"Extraction: {rag_survey_paper.title}")
        print(f"{'='*60}")
        for key, value in result.items():
            if isinstance(value, list):
                print(f"\n{key}:")
                for item in (value or []):
                    print(f"  - {item}")
            else:
                print(f"\n{key}:\n  {value}")
        print(f"{'='*60}\n")
        assert result  # always passes — this test exists for its print output
