"""
tests/test_synthesize.py

Unit and integration tests for the synthesize capability.

Unit tests mock the Anthropic client — no API key or network required.
Integration test runs fetch → extract → compare → synthesize on the two
canonical RAG papers and prints the full synthesis text.

Reference papers
----------------
RAG_SURVEY_ID   = "2312.10997"  Gao et al. 2023
ORIGINAL_RAG_ID = "2005.11401"  Lewis et al. 2020
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.capabilities.synthesize import (
    synthesize,
    _MODEL,
    _build_synthesis_prompt,
)

RAG_SURVEY_ID   = "2312.10997"
ORIGINAL_RAG_ID = "2005.11401"
QUERY = "What is the current state of retrieval-augmented generation for large language models?"

_FAKE_SUMMARIES = {
    RAG_SURVEY_ID: {
        "extraction": {
            "research_question": "How can RAG systems be categorised and improved?",
            "method": "Survey and taxonomy of RAG paradigms",
            "key_results": "Three paradigms identified; dense retrieval outperforms sparse on open-domain QA.",
            "limitations": "Scope limited to English papers through 2023.",
        }
    },
    ORIGINAL_RAG_ID: {
        "extraction": {
            "research_question": "Can retrieval augmentation reduce hallucination?",
            "method": "RAG with FAISS dense retrieval over Wikipedia",
            "key_results": "RAG outperforms closed-book GPT-2 by up to 18% on open-domain QA.",
            "limitations": "Fixed retrieval corpus; no mechanism to update knowledge after training.",
        }
    },
}

_FAKE_COMPARISONS = {
    "agreements": [
        "Dense retrieval over Wikipedia improves factual accuracy.",
        "Both treat external memory as complementary to parametric model knowledge.",
    ],
    "disagreements": [
        {
            "point": "Scope",
            "positions": {
                RAG_SURVEY_ID: "Broad taxonomy across many RAG variants.",
                ORIGINAL_RAG_ID: "Specific end-to-end trainable architecture.",
            },
        }
    ],
    "unique_contributions": {
        RAG_SURVEY_ID: "Three-paradigm taxonomy: Naive, Advanced, Modular RAG.",
        ORIGINAL_RAG_ID: "RAG-Sequence and RAG-Token decoding strategies.",
    },
}

_FAKE_SYNTHESIS = (
    "Retrieval-augmented generation has emerged as a leading approach to reducing "
    "hallucination in large language models by grounding generation in retrieved evidence."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(text: str = _FAKE_SYNTHESIS) -> MagicMock:
    content_block = MagicMock()
    content_block.text = text

    response = MagicMock()
    response.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Unit tests — no network, no API key
# ---------------------------------------------------------------------------

class TestSynthesizeUnit:
    def test_correct_model_used(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.synthesize._get_client", return_value=client):
            synthesize(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert client.messages.create.call_args.kwargs["model"] == _MODEL

    def test_model_is_sonnet(self):
        assert "sonnet" in _MODEL.lower()

    def test_no_tool_use(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.synthesize._get_client", return_value=client):
            synthesize(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        kwargs = client.messages.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_returns_string(self):
        with patch("research_assistant.capabilities.synthesize._get_client",
                   return_value=_make_mock_client()):
            result = synthesize(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert isinstance(result, str)
        assert result == _FAKE_SYNTHESIS

    def test_missing_api_key_raises_environment_error(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                synthesize(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)

    def test_query_appears_in_prompt(self):
        prompt = _build_synthesis_prompt(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert QUERY in prompt

    def test_prompt_includes_extraction_fields(self):
        prompt = _build_synthesis_prompt(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert "key_results" in prompt
        assert "method" in prompt

    def test_prompt_includes_agreements(self):
        prompt = _build_synthesis_prompt(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert _FAKE_COMPARISONS["agreements"][0] in prompt

    def test_prompt_includes_unique_contributions(self):
        prompt = _build_synthesis_prompt(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert _FAKE_COMPARISONS["unique_contributions"][RAG_SURVEY_ID] in prompt

    def test_prompt_includes_both_arxiv_ids(self):
        prompt = _build_synthesis_prompt(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert RAG_SURVEY_ID in prompt
        assert ORIGINAL_RAG_ID in prompt

    def test_prompt_instructs_prose_not_bullets(self):
        prompt = _build_synthesis_prompt(QUERY, _FAKE_SUMMARIES, _FAKE_COMPARISONS)
        assert "bullet" in prompt.lower() or "flowing prose" in prompt.lower()

    def test_empty_comparisons_handled_gracefully(self):
        client = _make_mock_client()
        with patch("research_assistant.capabilities.synthesize._get_client", return_value=client):
            result = synthesize(QUERY, _FAKE_SUMMARIES, {})
        assert result == _FAKE_SYNTHESIS


# ---------------------------------------------------------------------------
# Integration test — requires ANTHROPIC_API_KEY + network
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_pipeline_synthesis():
    """Run fetch → extract → compare → synthesize on both RAG papers."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live synthesis test.")

    from research_assistant.capabilities.fetch_paper import fetch_papers
    from research_assistant.capabilities.extract import extract_paper
    from research_assistant.capabilities.compare import compare_extractions

    print(f"\n[fixture] Fetching papers...")
    papers = fetch_papers([RAG_SURVEY_ID, ORIGINAL_RAG_ID])
    if len(papers) < 2:
        pytest.skip("arXiv returned fewer than 2 papers — possible 503, re-run to confirm.")

    summaries: dict = {}
    titles: dict = {}
    for paper in papers:
        print(f"[fixture] Extracting {paper.arxiv_id}...")
        summaries[paper.arxiv_id] = {"extraction": extract_paper(paper)}
        titles[paper.arxiv_id] = paper.title

    print("[fixture] Comparing...")
    extractions = {aid: s["extraction"] for aid, s in summaries.items()}
    comparisons = compare_extractions(extractions, titles)

    print("[fixture] Synthesizing...")
    result = synthesize(QUERY, summaries, comparisons)
    return result, titles


class TestSynthesizeIntegration:
    def test_synthesis_is_string(self, full_pipeline_synthesis):
        text, _ = full_pipeline_synthesis
        assert isinstance(text, str)

    def test_synthesis_is_substantive(self, full_pipeline_synthesis):
        text, _ = full_pipeline_synthesis
        assert len(text) > 500, (
            f"Synthesis too short ({len(text)} chars) — expected substantive prose."
        )

    def test_synthesis_references_rag(self, full_pipeline_synthesis):
        text, _ = full_pipeline_synthesis
        assert "retrieval" in text.lower() or "RAG" in text, (
            "Synthesis should mention retrieval or RAG."
        )

    def test_synthesis_references_both_papers(self, full_pipeline_synthesis):
        text, titles = full_pipeline_synthesis
        # Check that at least one identifying term from each paper appears.
        survey_terms = ["survey", "taxonomy", "Gao", "2312"]
        original_terms = ["Lewis", "knowledge-intensive", "RAG-Sequence", "RAG-Token", "2005"]
        assert any(t.lower() in text.lower() for t in survey_terms), (
            f"Synthesis should reference the RAG survey. Text:\n{text[:300]}"
        )
        assert any(t.lower() in text.lower() for t in original_terms), (
            f"Synthesis should reference the original RAG paper. Text:\n{text[:300]}"
        )

    def test_print_full_synthesis(self, full_pipeline_synthesis):
        """Print the complete synthesis — the capstone output of the pipeline."""
        text, _ = full_pipeline_synthesis
        bar = "=" * 65
        print(f"\n{bar}")
        print(f"SYNTHESIS: {QUERY}")
        print(bar)
        print(text)
        print(f"{bar}\n")
        assert text  # always passes — exists for its print output
