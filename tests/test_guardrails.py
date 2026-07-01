"""
tests/test_guardrails.py

Unit and integration tests for the guardrails layer.

Unit tests mock _get_client — no API key or network required.
Integration tests call the live API — require ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.context import ResearchContext
from research_assistant.guardrails import (
    GuardrailError,
    check_synthesis_grounding,
    validate_query,
    validate_search_results,
    validate_stage_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_validate_query_mock(valid: bool, issue: str | None, severity: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "validate_query"
    block.input = {"valid": valid, "issue": issue, "severity": severity}

    resp = MagicMock()
    resp.content = [block]

    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def _make_grounding_mock(
    grounded: list[str],
    ungrounded: list[str],
    confidence: str,
    warning: str | None,
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "check_grounding"
    block.input = {
        "grounded_claims": grounded,
        "ungrounded_claims": ungrounded,
        "confidence": confidence,
        "warning": warning,
    }

    resp = MagicMock()
    resp.content = [block]

    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def _make_paper(title: str, arxiv_id: str = "0000.00000") -> MagicMock:
    p = MagicMock()
    p.title = title
    p.arxiv_id = arxiv_id
    return p


# ---------------------------------------------------------------------------
# Realistic test data for integration tests
# ---------------------------------------------------------------------------

RAG_SYNTHESIS_SNIPPET = """
Retrieval-Augmented Generation (RAG) has achieved significant improvements across benchmarks.
EvoR achieved a 76x accuracy improvement over single-source baselines on code generation tasks.
FAIR-RAG reported an 8.3-point F1 improvement over iterative baselines on HotpotQA.
The EHR-RAGp system achieved an F1 of 0.453 on clinical prediction benchmarks.
Across studies, retrieval quality is consistently identified as the primary bottleneck, with
oracle retrieval yielding roughly $0.02 per query in cost savings compared to dynamic retrieval.
Most papers evaluate on Natural Questions and TriviaQA as standard benchmarks.
"""

RAG_EXTRACTIONS = {
    "2402.12317": {
        "extracted": {
            "title": "EvoR: Evolving Retrieval for Code Generation",
            "key_results": "EvoR achieves 76x accuracy improvement over single-source baselines by combining heterogeneous knowledge sources.",
            "method": "Iterative retrieval from library docs, web search, execution traces, and LLM-generated snippets.",
            "dataset": "HumanEval, MBPP, Natural Questions",
        }
    },
    "2510.22344": {
        "extracted": {
            "title": "FAIR-RAG: Faithful Adaptive Iterative Refinement for RAG",
            "key_results": "8.3-point F1 improvement over iterative baselines on HotpotQA via Structured Evidence Assessment (SEA).",
            "method": "SEA module decomposes multi-hop queries into required sub-findings.",
            "dataset": "HotpotQA, Natural Questions, TriviaQA",
        }
    },
    "2605.12335": {
        "extracted": {
            "title": "EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model",
            "key_results": "F1=0.453 on clinical prediction. Prototype-guided retrieval outperforms dense retrieval by 12 points.",
            "method": "Prototype-guided retrieval with EHR foundation model fine-tuning.",
            "dataset": "MIMIC-III, eICU",
        }
    },
}


# ---------------------------------------------------------------------------
# 1. validate_query — local short-circuits (no API)
# ---------------------------------------------------------------------------

class TestValidateQueryLocal:
    def test_empty_query_is_blocked(self):
        result = validate_query("")
        assert result["severity"] == "block"
        assert result["valid"] is False

    def test_whitespace_only_is_blocked(self):
        result = validate_query("   \n\t  ")
        assert result["severity"] == "block"

    def test_too_short_is_blocked(self):
        result = validate_query("ML")
        assert result["severity"] == "block"
        assert "short" in (result["issue"] or "").lower()

    def test_exactly_9_chars_is_blocked(self):
        result = validate_query("a" * 9)
        assert result["severity"] == "block"

    def test_exactly_10_chars_does_not_short_circuit(self, monkeypatch):
        # Should call Claude (not be blocked locally), so we mock the API
        mock_client = _make_validate_query_mock(True, None, "ok")
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = validate_query("a" * 10)
        assert mock_client.messages.create.called


# ---------------------------------------------------------------------------
# 2. validate_query — mocked Claude responses
# ---------------------------------------------------------------------------

class TestValidateQueryMocked:
    def test_returns_ok_for_valid_query(self):
        mock_client = _make_validate_query_mock(True, None, "ok")
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = validate_query("What is the current state of retrieval-augmented generation?")
        assert result["valid"] is True
        assert result["severity"] == "ok"
        assert result["issue"] is None

    def test_returns_block_for_harmful_query(self):
        mock_client = _make_validate_query_mock(
            False, "This is a harmful request, not a research question.", "block"
        )
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = validate_query("How do I make a bomb at home step by step")
        assert result["valid"] is False
        assert result["severity"] == "block"

    def test_returns_warn_for_vague_query(self):
        mock_client = _make_validate_query_mock(
            True, "Query is very vague — consider narrowing your topic.", "warn"
        )
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = validate_query("machine learning")
        assert result["severity"] == "warn"
        assert result["valid"] is True

    def test_returns_warn_for_non_english(self):
        mock_client = _make_validate_query_mock(
            True, "arXiv search works best in English.", "warn"
        )
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = validate_query("Quelles sont les avancées en apprentissage automatique?")
        assert result["severity"] == "warn"
        assert result["valid"] is True  # non-English should NOT be blocked

    def test_api_called_once(self):
        mock_client = _make_validate_query_mock(True, None, "ok")
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            validate_query("What are the recent advances in transformer architectures?")
        assert mock_client.messages.create.call_count == 1

    def test_result_has_required_keys(self):
        mock_client = _make_validate_query_mock(True, None, "ok")
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = validate_query("What are transformers used for in NLP research?")
        assert {"valid", "issue", "severity"} <= set(result.keys())

    def test_raises_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            validate_query("What is the state of transformers in NLP?")


# ---------------------------------------------------------------------------
# 3. validate_search_results — pure heuristic, no API
# ---------------------------------------------------------------------------

class TestValidateSearchResults:
    def test_zero_papers_warns(self):
        result = validate_search_results("retrieval augmented generation", [])
        assert result["severity"] == "warn"

    def test_one_paper_warns(self):
        result = validate_search_results("retrieval augmented generation", [_make_paper("RAG paper")])
        assert result["severity"] == "warn"

    def test_three_papers_ok_when_on_topic(self):
        papers = [
            _make_paper("Retrieval-Augmented Generation for LLMs"),
            _make_paper("Retrieval mechanisms in RAG"),
            _make_paper("Augmented Generation techniques"),
        ]
        result = validate_search_results("retrieval augmented generation", papers)
        assert result["valid"] is True

    def test_off_topic_results_warn(self):
        papers = [
            _make_paper("Completely Unrelated Paper About Cooking"),
            _make_paper("Another Unrelated Topic on Gardening"),
            _make_paper("Something About Weather Prediction"),
            _make_paper("Unrelated Chemistry Paper"),
            _make_paper("Unrelated Biology Paper"),
        ]
        result = validate_search_results("retrieval augmented generation", papers)
        assert result["severity"] == "warn"

    def test_always_returns_valid_true(self):
        # validate_search_results never blocks, only warns
        result = validate_search_results("retrieval augmented generation", [])
        assert result["valid"] is True

    def test_result_has_required_keys(self):
        result = validate_search_results("rag", [_make_paper("RAG paper")] * 3)
        assert {"valid", "issue", "severity"} <= set(result.keys())


# ---------------------------------------------------------------------------
# 4. validate_stage_output — extract stage
# ---------------------------------------------------------------------------

class TestValidateStageExtract:
    def _make_extraction(self, rq=True, kr=True):
        return {"extracted": {
            "research_question": "What?" if rq else None,
            "key_results": "Results." if kr else None,
        }}

    def test_all_populated_passes(self):
        summaries = {str(i): self._make_extraction() for i in range(5)}
        ctx = ResearchContext(query="test")
        result = validate_stage_output("extract", summaries, ctx)
        assert result["passed"] is True
        assert result["warnings"] == []

    def test_empty_summaries_fatal(self):
        ctx = ResearchContext(query="test")
        result = validate_stage_output("extract", {}, ctx)
        assert result["passed"] is False
        assert result["fatal"] is True

    def test_low_rq_coverage_warns(self):
        summaries = {
            "0": self._make_extraction(rq=True),
            "1": self._make_extraction(rq=False),
            "2": self._make_extraction(rq=False),
            "3": self._make_extraction(rq=False),
            "4": self._make_extraction(rq=False),
        }
        ctx = ResearchContext(query="test")
        result = validate_stage_output("extract", summaries, ctx)
        assert result["passed"] is False
        assert any("research_question" in w for w in result["warnings"])

    def test_low_kr_coverage_warns(self):
        summaries = {str(i): self._make_extraction(kr=False) for i in range(5)}
        ctx = ResearchContext(query="test")
        result = validate_stage_output("extract", summaries, ctx)
        assert any("key_results" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# 5. validate_stage_output — compare stage
# ---------------------------------------------------------------------------

class TestValidateStageCompare:
    def test_populated_compare_passes(self):
        comparisons = {
            "agreements": ["Papers agree on X."],
            "disagreements": [{"point": "Y", "positions": {"A": "yes", "B": "no"}}],
        }
        ctx = ResearchContext(query="test")
        result = validate_stage_output("compare", comparisons, ctx)
        assert result["passed"] is True

    def test_empty_compare_warns(self):
        ctx = ResearchContext(query="test")
        result = validate_stage_output("compare", {}, ctx)
        assert result["passed"] is False
        assert result["warnings"]


# ---------------------------------------------------------------------------
# 6. validate_stage_output — synthesize stage
# ---------------------------------------------------------------------------

class TestValidateStageSynthesize:
    def _make_synthesis_output(self, text: str):
        return {"synthesis": text}

    def test_good_synthesis_passes(self):
        text = (
            "Retrieval-Augmented Generation (RAG) has matured significantly. "
            "Multiple papers (Smith et al. 2024) demonstrate that retrieval quality "
            "is the primary bottleneck. Recent work including EvoR [2402.12317] shows "
            "significant gains. " * 15
        )
        ctx = ResearchContext(query="test")
        result = validate_stage_output("synthesize", self._make_synthesis_output(text), ctx)
        assert result["passed"] is True

    def test_short_synthesis_warns(self):
        ctx = ResearchContext(query="test")
        result = validate_stage_output("synthesize", self._make_synthesis_output("Short."), ctx)
        assert any("short" in w.lower() for w in result["warnings"])

    def test_no_citation_warns(self):
        ctx = ResearchContext(query="test")
        long_text = "This is a synthesis without any citations or references. " * 20
        result = validate_stage_output("synthesize", self._make_synthesis_output(long_text), ctx)
        assert any("reference" in w.lower() or "citation" in w.lower() for w in result["warnings"])

    def test_arxiv_id_counts_as_citation(self):
        text = (
            "RAG has matured significantly. EvoR [2402.12317] achieves improvements. " * 20
        )
        ctx = ResearchContext(query="test")
        result = validate_stage_output("synthesize", self._make_synthesis_output(text), ctx)
        citation_warnings = [
            w for w in result["warnings"]
            if "reference" in w.lower() or "citation" in w.lower()
        ]
        assert citation_warnings == []

    def test_et_al_counts_as_citation(self):
        text = "Smith et al. (2024) demonstrated significant improvements in RAG. " * 20
        ctx = ResearchContext(query="test")
        result = validate_stage_output("synthesize", self._make_synthesis_output(text), ctx)
        citation_warnings = [
            w for w in result["warnings"]
            if "reference" in w.lower() or "citation" in w.lower()
        ]
        assert citation_warnings == []


# ---------------------------------------------------------------------------
# 7. validate_stage_output — ideate stage
# ---------------------------------------------------------------------------

class TestValidateStageIdeate:
    def _good_idea(self, title="Idea", diff="medium"):
        return {
            "title": title,
            "hypothesis": "If we do X, Y will happen.",
            "method": "Run experiment on dataset Z with model M.",
            "gap_addressed": "The gap is G.",
            "difficulty": diff,
        }

    def test_good_ideas_pass(self):
        ideas_data = {
            "experiment_ideas": [self._good_idea("A"), self._good_idea("B")],
            "most_promising": 0,
            "reasoning": "A is better.",
        }
        ctx = ResearchContext(query="test")
        result = validate_stage_output("ideate", ideas_data, ctx)
        assert result["passed"] is True

    def test_no_ideas_warns(self):
        ctx = ResearchContext(query="test")
        result = validate_stage_output("ideate", {"experiment_ideas": []}, ctx)
        assert not result["passed"]

    def test_missing_hypothesis_warns(self):
        idea = self._good_idea()
        idea["hypothesis"] = ""
        ideas_data = {"experiment_ideas": [idea], "most_promising": 0}
        ctx = ResearchContext(query="test")
        result = validate_stage_output("ideate", ideas_data, ctx)
        assert any("hypothesis" in w.lower() for w in result["warnings"])

    def test_missing_method_warns(self):
        idea = self._good_idea()
        idea["method"] = "   "
        ideas_data = {"experiment_ideas": [idea], "most_promising": 0}
        ctx = ResearchContext(query="test")
        result = validate_stage_output("ideate", ideas_data, ctx)
        assert any("method" in w.lower() for w in result["warnings"])

    def test_invalid_most_promising_index_warns(self):
        ideas_data = {
            "experiment_ideas": [self._good_idea()],
            "most_promising": 5,  # out of range
        }
        ctx = ResearchContext(query="test")
        result = validate_stage_output("ideate", ideas_data, ctx)
        assert any("most_promising" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# 8. check_synthesis_grounding — mocked
# ---------------------------------------------------------------------------

class TestCheckSynthesisGroundingMocked:
    def test_returns_required_keys(self):
        mock_client = _make_grounding_mock(["Claim A"], [], "high", None)
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = check_synthesis_grounding("Some synthesis text.", RAG_EXTRACTIONS)
        assert {"grounded_claims", "ungrounded_claims", "confidence", "warning"} <= set(result.keys())

    def test_empty_synthesis_skips_api(self):
        mock_client = _make_grounding_mock([], [], "low", None)
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = check_synthesis_grounding("", RAG_EXTRACTIONS)
        assert mock_client.messages.create.call_count == 0
        assert result["confidence"] == "low"

    def test_whitespace_only_synthesis_skips_api(self):
        mock_client = _make_grounding_mock([], [], "low", None)
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = check_synthesis_grounding("   \n\t  ", RAG_EXTRACTIONS)
        assert mock_client.messages.create.call_count == 0

    def test_api_called_once_for_real_synthesis(self):
        mock_client = _make_grounding_mock(["76x improvement"], [], "high", None)
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            check_synthesis_grounding(RAG_SYNTHESIS_SNIPPET, RAG_EXTRACTIONS)
        assert mock_client.messages.create.call_count == 1

    def test_ungrounded_claims_trigger_warning(self):
        mock_client = _make_grounding_mock(
            ["76x improvement"],
            ["$0.02 per query cost claim"],
            "medium",
            "1 claim could not be traced to any source extraction.",
        )
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = check_synthesis_grounding(RAG_SYNTHESIS_SNIPPET, RAG_EXTRACTIONS)
        assert result["warning"] is not None
        assert len(result["ungrounded_claims"]) == 1

    def test_all_grounded_warning_is_none(self):
        mock_client = _make_grounding_mock(
            ["76x improvement", "8.3-point F1", "F1=0.453"], [], "high", None
        )
        with patch("research_assistant.guardrails._get_client", return_value=mock_client):
            result = check_synthesis_grounding(RAG_SYNTHESIS_SNIPPET, RAG_EXTRACTIONS)
        assert result["warning"] is None
        assert result["confidence"] == "high"

    def test_raises_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            check_synthesis_grounding("Some synthesis.", RAG_EXTRACTIONS)


# ---------------------------------------------------------------------------
# Integration tests — require ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_query_valid():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live guardrail tests.")
    return validate_query("What is the current state of retrieval-augmented generation for LLMs?")


@pytest.fixture(scope="module")
def live_query_blocked():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live guardrail tests.")
    return validate_query("2 + 2")


@pytest.fixture(scope="module")
def live_grounding():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping live guardrail tests.")
    return check_synthesis_grounding(RAG_SYNTHESIS_SNIPPET, RAG_EXTRACTIONS)


class TestValidateQueryLive:
    def test_research_question_is_valid(self, live_query_valid):
        assert live_query_valid["valid"] is True

    def test_research_question_is_ok_or_warn(self, live_query_valid):
        assert live_query_valid["severity"] in ("ok", "warn")

    def test_has_all_keys(self, live_query_valid):
        assert {"valid", "issue", "severity"} <= set(live_query_valid.keys())

    def test_print_result(self, live_query_valid):
        print(f"\n[live validate_query] {live_query_valid}")
        assert True


class TestValidateQueryBlockedLive:
    def test_arithmetic_blocked_or_warned(self, live_query_blocked):
        # "2 + 2" may be blocked (too short short-circuits first, actually)
        # but the fixture passes after the local short-circuit with len < 10
        # The local short-circuit catches this so it should already be "block"
        assert live_query_blocked["severity"] in ("block", "warn")

    def test_print_result(self, live_query_blocked):
        print(f"\n[live validate_query blocked] {live_query_blocked}")
        assert True


class TestGroundingLive:
    def test_returns_required_keys(self, live_grounding):
        assert {"grounded_claims", "ungrounded_claims", "confidence", "warning"} <= set(live_grounding.keys())

    def test_grounded_claims_is_list(self, live_grounding):
        assert isinstance(live_grounding["grounded_claims"], list)

    def test_ungrounded_claims_is_list(self, live_grounding):
        assert isinstance(live_grounding["ungrounded_claims"], list)

    def test_confidence_is_valid_enum(self, live_grounding):
        assert live_grounding["confidence"] in ("high", "medium", "low")

    def test_76x_claim_is_grounded(self, live_grounding):
        all_grounded = " ".join(live_grounding["grounded_claims"]).lower()
        # The "76x" claim should be traceable to EvoR extraction
        assert "76" in all_grounded or any(
            "76" in c for c in live_grounding["grounded_claims"]
        ), f"Expected '76x' claim in grounded_claims, got: {live_grounding['grounded_claims']}"

    def test_print_full_output(self, live_grounding):
        print(f"\n{'='*70}")
        print("GROUNDING CHECK — Live output")
        print(f"{'='*70}")
        print(f"Confidence      : {live_grounding['confidence']}")
        print(f"Grounded claims : {len(live_grounding['grounded_claims'])}")
        for c in live_grounding["grounded_claims"]:
            print(f"  ✓ {c}")
        print(f"Ungrounded      : {len(live_grounding['ungrounded_claims'])}")
        for c in live_grounding["ungrounded_claims"]:
            print(f"  ? {c}")
        if live_grounding["warning"]:
            print(f"Warning         : {live_grounding['warning']}")
        print(f"{'='*70}\n")
        assert True
