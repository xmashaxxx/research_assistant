"""
tests/test_compare.py

Unit and integration tests for the compare capability.

Unit tests mock the Anthropic client — no API key or network required.
Integration tests fetch and extract both RAG papers via the live APIs,
then compare the extractions. Require ANTHROPIC_API_KEY and network.

Reference papers
----------------
RAG_SURVEY_ID   = "2312.10997"  Gao et al. 2023 — RAG survey
ORIGINAL_RAG_ID = "2005.11401"  Lewis et al. 2020 — original RAG paper
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.capabilities.compare import (
    compare_extractions,
    _COMPARE_TOOL,
    _MODEL,
    _build_compare_prompt,
)

RAG_SURVEY_ID   = "2312.10997"
ORIGINAL_RAG_ID = "2005.11401"

_FAKE_EXTRACTIONS = {
    RAG_SURVEY_ID: {
        "research_question": "How can RAG systems be systematically categorised and improved?",
        "method": "Survey and taxonomy of RAG architectures",
        "dataset": "Multiple NLP benchmarks",
        "key_results": "Identifies three RAG paradigms; dense retrieval outperforms sparse on open-domain QA.",
        "limitations": "Survey scope limited to English-language papers through 2023.",
        "related_work": ["Lewis et al. 2020", "DPR", "REALM"],
    },
    ORIGINAL_RAG_ID: {
        "research_question": "Can retrieval augmentation reduce hallucination in knowledge-intensive NLP tasks?",
        "method": "RAG with FAISS dense retrieval over Wikipedia passages",
        "dataset": "Natural Questions, TriviaQA, WebQuestions, CuratedTrec",
        "key_results": "RAG outperforms closed-book GPT-2 by up to 18% on open-domain QA.",
        "limitations": "Fixed retrieval corpus; no mechanism to update knowledge after training.",
        "related_work": ["DPR", "BART", "T5"],
    },
}

_FAKE_COMPARISON = {
    "agreements": [
        "Both papers agree that dense retrieval improves factual accuracy over closed-book models.",
        "Both treat Wikipedia as a primary external knowledge source.",
    ],
    "disagreements": [
        {
            "point": "Scope and purpose",
            "positions": {
                RAG_SURVEY_ID: "Provides a broad taxonomy of RAG paradigms across many works.",
                ORIGINAL_RAG_ID: "Proposes and empirically validates a specific RAG architecture.",
            },
        }
    ],
    "unique_contributions": {
        RAG_SURVEY_ID: "Comprehensive taxonomy classifying RAG into Naive, Advanced, and Modular paradigms.",
        ORIGINAL_RAG_ID: "First end-to-end trainable RAG model combining FAISS retrieval with BART generation.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(comparison: dict = _FAKE_COMPARISON) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "compare_papers"
    block.input = comparison

    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Unit tests — no network, no API key
# ---------------------------------------------------------------------------

class TestCompareTool:
    def test_tool_name(self):
        assert _COMPARE_TOOL["name"] == "compare_papers"

    def test_required_output_fields(self):
        required = set(_COMPARE_TOOL["input_schema"]["required"])
        assert required == {"agreements", "disagreements", "unique_contributions"}

    def test_agreements_is_array_of_strings(self):
        prop = _COMPARE_TOOL["input_schema"]["properties"]["agreements"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_disagreements_items_have_point_and_positions(self):
        items = _COMPARE_TOOL["input_schema"]["properties"]["disagreements"]["items"]
        assert "point" in items["properties"]
        assert "positions" in items["properties"]
        assert set(items["required"]) == {"point", "positions"}

    def test_unique_contributions_uses_additional_properties(self):
        prop = _COMPARE_TOOL["input_schema"]["properties"]["unique_contributions"]
        assert prop["type"] == "object"
        assert prop["additionalProperties"]["type"] == "string"


class TestCompareExtractions:
    def test_raises_on_fewer_than_two_papers(self):
        with pytest.raises(ValueError, match="at least 2"):
            compare_extractions({RAG_SURVEY_ID: _FAKE_EXTRACTIONS[RAG_SURVEY_ID]})

    def test_correct_model_used(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.compare._get_client", return_value=client):
            compare_extractions(_FAKE_EXTRACTIONS)
        assert client.messages.create.call_args.kwargs["model"] == _MODEL

    def test_tool_choice_forces_compare_papers(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.compare._get_client", return_value=client):
            compare_extractions(_FAKE_EXTRACTIONS)
        tc = client.messages.create.call_args.kwargs["tool_choice"]
        assert tc == {"type": "tool", "name": "compare_papers"}

    def test_both_arxiv_ids_appear_in_prompt(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.compare._get_client", return_value=client):
            compare_extractions(_FAKE_EXTRACTIONS)
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert RAG_SURVEY_ID in content
        assert ORIGINAL_RAG_ID in content

    def test_titles_used_when_provided(self):
        titles = {RAG_SURVEY_ID: "RAG Survey", ORIGINAL_RAG_ID: "Original RAG"}
        client = _make_mock_client()
        with patch("research_assistant.capabilities.compare._get_client", return_value=client):
            compare_extractions(_FAKE_EXTRACTIONS, titles=titles)
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "RAG Survey" in content
        assert "Original RAG" in content

    def test_returns_tool_input_dict(self):
        with patch("research_assistant.capabilities.compare._get_client",
                   return_value=_make_mock_client()):
            result = compare_extractions(_FAKE_EXTRACTIONS)
        assert result == _FAKE_COMPARISON

    def test_missing_api_key_raises_environment_error(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                compare_extractions(_FAKE_EXTRACTIONS)


class TestBuildComparePrompt:
    def test_includes_all_arxiv_ids(self):
        prompt = _build_compare_prompt(_FAKE_EXTRACTIONS, titles=None)
        assert RAG_SURVEY_ID in prompt
        assert ORIGINAL_RAG_ID in prompt

    def test_titles_appear_when_provided(self):
        titles = {RAG_SURVEY_ID: "Survey", ORIGINAL_RAG_ID: "Original"}
        prompt = _build_compare_prompt(_FAKE_EXTRACTIONS, titles=titles)
        assert "Survey" in prompt
        assert "Original" in prompt

    def test_key_fields_included(self):
        prompt = _build_compare_prompt(_FAKE_EXTRACTIONS, titles=None)
        assert "research_question" in prompt
        assert "method" in prompt
        assert "key_results" in prompt


# ---------------------------------------------------------------------------
# Integration tests — require ANTHROPIC_API_KEY + network
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_paper_extractions():
    """Fetch and extract both RAG papers. Returns (extractions, titles) dicts."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live compare test.")

    from research_assistant.capabilities.fetch_paper import fetch_papers
    from research_assistant.capabilities.extract import extract_paper

    papers = fetch_papers([RAG_SURVEY_ID, ORIGINAL_RAG_ID])
    if len(papers) < 2:
        pytest.skip("arXiv returned fewer than 2 papers — possible 503, re-run to confirm.")

    extractions: dict[str, dict] = {}
    titles: dict[str, str] = {}
    for paper in papers:
        print(f"\n[fixture] Extracting {paper.arxiv_id}...")
        extractions[paper.arxiv_id] = extract_paper(paper)
        titles[paper.arxiv_id] = paper.title

    return extractions, titles


@pytest.fixture(scope="module")
def live_comparison(two_paper_extractions):
    """Run compare_extractions once; share result across all integration tests."""
    extractions, titles = two_paper_extractions
    print("\n[fixture] Running comparison...")
    return compare_extractions(extractions, titles)


class TestCompareIntegration:
    def test_result_is_dict(self, live_comparison):
        assert isinstance(live_comparison, dict)

    def test_has_all_three_top_level_keys(self, live_comparison):
        assert "agreements" in live_comparison
        assert "disagreements" in live_comparison
        assert "unique_contributions" in live_comparison

    def test_agreements_nonempty(self, live_comparison):
        assert isinstance(live_comparison["agreements"], list)
        assert len(live_comparison["agreements"]) > 0

    def test_disagreements_nonempty(self, live_comparison):
        assert isinstance(live_comparison["disagreements"], list)
        assert len(live_comparison["disagreements"]) > 0

    def test_disagreements_have_point_and_positions(self, live_comparison):
        for d in live_comparison["disagreements"]:
            assert "point" in d, f"Missing 'point' in disagreement: {d}"
            assert "positions" in d, f"Missing 'positions' in disagreement: {d}"
            assert isinstance(d["positions"], dict)

    def test_unique_contributions_is_dict(self, live_comparison):
        assert isinstance(live_comparison["unique_contributions"], dict)

    def test_unique_contributions_nonempty(self, live_comparison):
        assert len(live_comparison["unique_contributions"]) > 0

    def test_print_full_comparison(self, live_comparison, two_paper_extractions):
        """Print the complete comparison so we can read Claude's analysis."""
        _, titles = two_paper_extractions
        print(f"\n{'='*65}")
        print("CROSS-PAPER COMPARISON")
        print(f"{'='*65}")

        print("\n--- AGREEMENTS ---")
        for i, ag in enumerate(live_comparison["agreements"], 1):
            print(f"  {i}. {ag}")

        print("\n--- DISAGREEMENTS ---")
        for i, d in enumerate(live_comparison["disagreements"], 1):
            print(f"  {i}. {d['point']}")
            for arxiv_id, stance in d.get("positions", {}).items():
                label = titles.get(arxiv_id, arxiv_id)
                print(f"       [{arxiv_id}] {label[:50]}")
                print(f"       → {stance}")

        print("\n--- UNIQUE CONTRIBUTIONS ---")
        for arxiv_id, contrib in live_comparison["unique_contributions"].items():
            label = titles.get(arxiv_id, arxiv_id)
            print(f"  [{arxiv_id}] {label[:50]}")
            print(f"  → {contrib}")

        print(f"{'='*65}\n")
        assert live_comparison  # always passes — exists for its output
